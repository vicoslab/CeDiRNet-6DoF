"""Config regressions using real imports in isolated CPU-only subprocesses.

Run with the project environment: python -m unittest discover -s tests -v.
No dataset, pretrained weights, network access, or accelerator is needed.
"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "crop_rgb": ("robot", "off"),
    "nocrop_rgb": ("off", "off"),
    "crop_rgbd": ("robot", "3d_arm_plane"),
}
CONFIG_ENV = (
    "CROP_MODE", "CLOTH_COMPETITION_CROP_MODE", "DEPTH_ENCODING",
    "CLOTH_COMPETITION_BACKBONE", "CLOTH_COMPETITION_ICRA2024_DATASET",
    "OUTPUT_DIR", "PRETRAINED_MODEL_DIR", "COCO_BG_DIR", "COCO_BG_IMAGES",
)


def run_probe(code, **overrides):
    env = {key: value for key, value in os.environ.items() if key not in CONFIG_ENV}
    env.update(
        PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1", OUTPUT_DIR=str(ROOT / "exp"),
    )
    env.update(overrides)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], cwd=ROOT,
        env=env, capture_output=True, text=True, timeout=60,
    )
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    return json.loads(result.stdout.splitlines()[-1])


def load_configs(**overrides):
    return run_probe("""
        import json
        from config import get_config_args
        result = {}
        for mode, dataset_key in [('train', 'train_dataset'), ('test', 'dataset')]:
            args = get_config_args('cloth_6dof', mode)
            dataset = args[dataset_key]['kwargs']
            result[mode] = {
                'save_dir': args['save_dir'],
                'checkpoint_path': args.get('checkpoint_path'),
                'crop_mode': args.get('crop_mode'),
                'crop_size': dataset['crop_by_robot_positions'],
                'segment_cloth': dataset['segment_cloth'],
                'segment_cloth_optional': dataset['segment_cloth_optional'],
                'samples': dataset['valid_sample_names'],
            }
        print(json.dumps(result))
    """, **overrides)


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"),
    "requires the project torch/torchvision environment",
)
class TrainingConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = {
            name: load_configs(CROP_MODE=crop, DEPTH_ENCODING=depth)
            for name, (crop, depth) in VARIANTS.items()
        }

    def test_training_outputs_are_distinct_for_all_released_variants(self):
        paths = [cfg['train']['save_dir'] for cfg in self.configs.values()]
        self.assertEqual(len(set(paths)), len(VARIANTS), paths)

    def test_evaluation_checkpoints_are_distinct_for_all_released_variants(self):
        paths = [cfg['test']['checkpoint_path'] for cfg in self.configs.values()]
        self.assertEqual(len(set(paths)), len(VARIANTS), paths)

    def test_evaluation_outputs_are_distinct_for_all_released_variants(self):
        paths = [cfg['test']['save_dir'] for cfg in self.configs.values()]
        self.assertEqual(len(set(paths)), len(VARIANTS), paths)

    def test_each_evaluation_uses_its_own_training_checkpoint(self):
        for name, cfg in self.configs.items():
            with self.subTest(variant=name):
                train_dir = Path(cfg['train']['save_dir'])
                self.assertEqual(Path(cfg['test']['checkpoint_path']), train_dir / 'checkpoint.pth')
                self.assertEqual(Path(cfg['test']['save_dir']), train_dir / 'test_results')

    def test_paths_record_the_normalized_crop_mode(self):
        for name, cfg in self.configs.items():
            crop, depth = VARIANTS[name]
            for mode in ('train', 'test'):
                with self.subTest(variant=name, mode=mode):
                    self.assertEqual(cfg[mode]['crop_mode'], crop)
                    self.assertIn('crop=' + crop, Path(cfg[mode]['save_dir']).parts)
                    self.assertIn('depth=' + depth, Path(cfg[mode]['save_dir']).parts)
                    expected_size = [768, 768] if mode == 'train' else [1024, 1024]
                    self.assertEqual(cfg[mode]['crop_size'], expected_size if crop == 'robot' else False)

    def test_legacy_crop_environment_and_aliases_use_the_same_paths(self):
        cases = (
            ({'CLOTH_COMPETITION_CROP_MODE': 'workspace'}, 'crop_rgb'),
            ({'CLOTH_COMPETITION_CROP_MODE': 'none'}, 'nocrop_rgb'),
            ({'CROP_MODE': 'FALSE', 'CLOTH_COMPETITION_CROP_MODE': 'robot'}, 'nocrop_rgb'),
        )
        for env, name in cases:
            with self.subTest(environment=env):
                self.assertEqual(load_configs(**env), self.configs[name])

    def test_train_and_evaluation_require_segmentation_masks(self):
        for name, cfg in self.configs.items():
            for mode in ('train', 'test'):
                with self.subTest(variant=name, mode=mode):
                    self.assertTrue(cfg[mode]['segment_cloth'])
                    self.assertFalse(cfg[mode]['segment_cloth_optional'])

    def test_missing_masks_fail_in_the_real_dataset_loader(self):
        errors = run_probe("""
            import json
            from pathlib import Path
            import tempfile
            from PIL import Image
            from config import get_config_args
            from datasets.ClothCompetitionICRA2024Dataset import ClothCompetitionICRA2024Dataset
            errors = {}
            with tempfile.TemporaryDirectory() as root:
                observation = Path(root) / 'fixture' / 'sample_000000' / 'observation_start'
                observation.mkdir(parents=True)
                Image.new('RGB', (8, 8)).save(observation / 'image_left.png')
                for mode, key in [('train', 'train_dataset'), ('test', 'dataset')]:
                    opts = get_config_args('cloth_6dof', mode)[key]['kwargs']
                    # Isolate mask validation from unrelated depth/pose/augmentation inputs.
                    opts.update(root_dir=root, subfolders=['fixture'], valid_sample_names=None,
                                use_depth=False, segement_bg_by_depth_thr=False,
                                crop_by_robot_positions=False, transform=None)
                    dataset = ClothCompetitionICRA2024Dataset(**opts)
                    dataset.remove_out_of_bounds_centers = False
                    try:
                        dataset[0]
                    except AssertionError as error:
                        errors[mode] = str(error)
                    else:
                        errors[mode] = ''
            print(json.dumps(errors))
        """)
        for mode, error in errors.items():
            with self.subTest(mode=mode):
                self.assertIn('Missing segmentation file', error)
                self.assertIn('cloth_segmentation.png', error)

    def test_selected_split_is_disjoint_and_covers_all_503_samples(self):
        for name, cfg in self.configs.items():
            with self.subTest(variant=name):
                train = cfg['train']['samples']
                test = cfg['test']['samples']
                self.assertEqual(len(train), 440)
                self.assertEqual(len(test), 63)
                self.assertEqual(len(set(train)), len(train))
                self.assertEqual(len(set(test)), len(test))
                self.assertFalse(set(train) & set(test))
                expected = {
                    f'cloth_competition_dataset_{subset:04}/sample_{index:06}'
                    for subset, size in ((0, 125), (1, 378)) for index in range(size)
                }
                self.assertEqual(set(train) | set(test), expected)


if __name__ == '__main__':
    unittest.main()
