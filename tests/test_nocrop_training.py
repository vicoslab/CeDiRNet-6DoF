"""Fixed-size training regression, plus opt-in real masked dataset batching.

Set CEDIRNET_BATCH_DATASET to the dataset root to enable the real-data test.
"""
import copy
import os
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from utils.transforms import get_transform
from utils.utils import variable_len_collate

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [('off', 'off', 3), ('robot', 'off', 3), ('robot', '3d_arm_plane', 4)]


def config(crop, depth, backgrounds=False):
    env = dict(os.environ, CROP_MODE=crop, DEPTH_ENCODING=depth)
    env.pop('COCO_BG_DIR', None)
    env.pop('COCO_BG_IMAGES', None)
    if backgrounds:
        env.update(COCO_BG_DIR='unused', COCO_BG_IMAGES='unused')
    with patch.dict(os.environ, env, clear=True):
        return runpy.run_path(str(ROOT / 'src/config/cloth_6dof/train.py'))['get_args']()


class NoCropTrainingTests(unittest.TestCase):
    def test_nocrop_two_samples_collate_with_aligned_targets(self):
        opts = config('off', 'off')['train_dataset']['kwargs']
        transform = get_transform(opts['transform'])
        samples = []
        for seed in (1337, 42):
            h, w = 90, 160
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[30:60, 60:100] = 1
            sample = {k: mask.copy() for k in
                      ('instance', 'label', 'ignore', 'mask', 'segmentation_mask', 'bg_mask_by_depth')}
            # Float masks mirror the loader and avoid uint8 ToTensor's /255.
            sample['segmentation_mask'] = mask.astype(np.float32)
            sample.update(image=np.repeat(mask[..., None], 3, axis=2),
                          orientation=np.repeat(mask[..., None], 3, axis=2).astype(np.float32),
                          center=np.array([[80., 45.], [0., 0.]], dtype=np.float32),
                          K=np.array([[100., 0., 80.], [0., 100., 45.], [0., 0., 1.]]))
            samples.append(transform(sample, np.random.default_rng(seed)))
        batch = variable_len_collate(samples)
        self.assertEqual(tuple(batch['image'].shape), (2, 3, 768, 768))
        for sample in samples:
            for key in ('instance', 'label', 'ignore', 'orientation', 'mask', 'segmentation_mask', 'bg_mask_by_depth'):
                self.assertEqual(tuple(sample[key].shape[-2:]), (768, 768), key)
            np.testing.assert_allclose(sample['center'][0], [384., 384.])
            np.testing.assert_allclose(sample['center'][1], [0., 0.])
            np.testing.assert_allclose(sample['K'][:2, 2], [384., 384.])
            self.assertEqual(set(torch.unique(sample['segmentation_mask']).tolist()), {0., 1.})
            self.assertEqual(float(sample['segmentation_mask'][..., 384, 384]), 1.)
            self.assertTrue(torch.all(sample['orientation'][..., 384, 384] == 1))

    def test_provenance_augmentation_order_and_robot_crop_preserved(self):
        for crop, depth, _ in VARIANTS:
            with self.subTest(crop=crop, depth=depth):
                opts = config(crop, depth, backgrounds=True)['train_dataset']['kwargs']
                names = [t['name'] for t in opts['transform']]
                expected = (['ToTensor', 'Resize', 'Background', 'Background', 'ColorJitter']
                            if crop == 'off' else
                            ['ToTensor', 'Background', 'Background', 'RandomResize', 'ColorJitter'])
                self.assertEqual(names, expected)
                self.assertEqual(opts['crop_by_robot_positions'], False if crop == 'off' else (768, 768))
                self.assertTrue(opts['segment_cloth'])
                self.assertFalse(opts['segment_cloth_optional'])

    def test_arm_plane_depth_normalization_matches_published_training(self):
        fpn = config('robot', '3d_arm_plane')['model']['kwargs']['fpn_args']
        self.assertEqual(fpn['depth_mean'], 0)
        self.assertEqual(fpn['depth_std'], 1)

    @unittest.skipUnless(os.environ.get('CEDIRNET_BATCH_DATASET'), 'set CEDIRNET_BATCH_DATASET for real samples')
    def test_real_masked_two_sample_batches_all_variants(self):
        from datasets.ClothCompetitionICRA2024Dataset import ClothCompetitionICRA2024Dataset
        for crop, depth, channels in VARIANTS:
            with self.subTest(crop=crop, depth=depth):
                opts = config(crop, depth)['train_dataset']['kwargs']
                opts['root_dir'] = os.environ['CEDIRNET_BATCH_DATASET']
                dataset = ClothCompetitionICRA2024Dataset(**copy.deepcopy(opts))
                self.assertGreaterEqual(len(dataset), 2)
                batch = variable_len_collate([dataset[0], dataset[1]])
                self.assertEqual(tuple(batch['image'].shape), (2, channels, 768, 768))
                self.assertEqual(tuple(batch['segmentation_mask'].shape[-2:]), (768, 768))
                self.assertTrue(torch.isfinite(batch['image']).all())
                self.assertTrue(torch.isfinite(batch['orientation']).all())
                self.assertTrue(torch.all(batch['segmentation_mask'].flatten(1).sum(1) > 0))
                print('REAL_BATCH', crop, depth, tuple(batch['image'].shape),
                      'mask_pixels', batch['segmentation_mask'].flatten(1).sum(1).tolist(), flush=True)


if __name__ == '__main__':
    unittest.main()
