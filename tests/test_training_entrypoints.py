"""Offline regressions for the real training/evaluation entrypoints."""
import importlib
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def config_args(mode, root):
    """Keep real components, but use an offline backbone and temporary output."""
    env = dict(os.environ, CROP_MODE='robot', DEPTH_ENCODING='off',
               CLOTH_COMPETITION_BACKBONE='resnet18')
    env.pop('COCO_BG_DIR', None)
    env.pop('COCO_BG_IMAGES', None)
    with patch.dict(os.environ, env, clear=True):
        args = runpy.run_path(str(ROOT / f'src/config/cloth_6dof/{mode}.py'))['get_args']()
    args.update(cuda=False, save_dir=str(root), pretrained_model_path=None,
                pretrained_center_model_path=None)
    args['model']['kwargs']['pretrained'] = False
    args['visualizer'].setdefault('opts', {})['tensorboard_dir'] = None
    key = 'dataset' if mode == 'test' else 'train_dataset'
    args[key]['workers'] = 0
    # Construction enumerates samples but never reads or trains on this fixture.
    (root / 'fixture/sample_000000').mkdir(parents=True, exist_ok=True)
    args[key]['kwargs'].update(root_dir=str(root), subfolders=['fixture'],
                              valid_sample_names=None)
    return args


class TrainingEntrypointTests(unittest.TestCase):
    def test_default_evaluation_constructs_real_processing_pipeline(self):
        from models import get_model
        from test_polar import Evaluator
        from inference.post_processing.patch_split import PatchSplitPostProcesser

        with tempfile.TemporaryDirectory() as directory:
            args = config_args('test', Path(directory))
            model = get_model(args['model']['name'], args['model']['kwargs'])
            args['checkpoint_path'] = str(Path(directory) / 'model.pth')
            torch.save({'model_state_dict': torch.nn.DataParallel(model).state_dict()},
                       args['checkpoint_path'])
            del model
            evaluator = Evaluator(args)
            evaluator.initialize()
            self.assertIs(args['display'], False)
            self.assertEqual(len(evaluator.dataset_it.dataset), 1)
            self.assertIsInstance(evaluator.processed_image_iter, PatchSplitPostProcesser)
            self.assertTrue(evaluator.processed_image_iter.only_essential_output)
            self.assertEqual(evaluator.device, torch.device('cpu'))

    def test_training_module_imports_with_hard_sampler(self):
        module = importlib.import_module('train_polar')
        self.assertTrue(callable(module.Trainer))

    def test_hard_sampler_preserves_seed_locking_through_gt_wrapper(self):
        from datasets.LockableSeedRandomAccess import LockableSeedRandomAccess
        from datasets.CenterDirGroundtruthDataset import CenterDirGroundtruthDataset
        from utils.hard_sampler import HardExamplesBatchSampler

        class SeededDataset(torch.utils.data.Dataset, LockableSeedRandomAccess):
            def __len__(self):
                return 4

            def lock_samples_seed(self, index_list):
                self.locked = list(index_list)

        dataset = SeededDataset()
        wrapped = CenterDirGroundtruthDataset(dataset, None)
        sampler = HardExamplesBatchSampler(
            wrapped, torch.utils.data.SequentialSampler(wrapped),
            batch_size=2, hard_sample_size=1, drop_last=True,
        )
        sampler.update_difficulty_score({'index': torch.arange(4)},
                                        torch.tensor([0., 0., 0., 10.]))
        batches = list(sampler)
        self.assertEqual(dataset.locked, [3])
        self.assertEqual(batches, [[3, 0], [3, 1], [3, 2], [3, 3]])
        with self.assertRaises(NotImplementedError):
            LockableSeedRandomAccess().lock_samples_seed([])


class CheckpointEntrypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from train_polar import Trainer

        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        args = config_args('train', cls.root)
        args['save'] = False
        trainer = Trainer(0, 0, 1, args, use_distributed_data_parallel=False)
        trainer.initialize_data_parallel()
        trainer.initialize()
        # Use the real Logger and checkpoint writer, including NumPy's loss scalar.
        trainer.logger.add('train', np.float64(0.125))
        state = dict(epoch=3, model_state_dict=trainer.model.state_dict(),
                     center_model_state_dict=trainer.center_model.state_dict(),
                     criterion_state_dict=trainer.criterion.state_dict(),
                     optim_state_dict=trainer.optimizer.state_dict(),
                     center_optim_state_dict=trainer.center_optimizer.state_dict(),
                     logger_data=trainer.logger.data,
                     sample_loss_history={0: np.float64(0.25)})
        trainer.save_checkpoint(state)
        cls.checkpoint = str(cls.root / 'checkpoint.pth')
        cls.weights_only_checkpoint = str(cls.root / 'weights.pth')
        torch.save({'model_state_dict': state['model_state_dict']}, cls.weights_only_checkpoint)
        cls.model_key = next(iter(state['model_state_dict']))
        cls.model_value = state['model_state_dict'][cls.model_key].clone()
        cls.center_key = next(iter(state['center_model_state_dict']))
        cls.center_value = state['center_model_state_dict'][cls.center_key].clone()
        trainer.cleanup()

    def initialize_trainer(self, checkpoint_key):
        from train_polar import Trainer

        args = config_args('train', self.root)
        args['save'] = False
        args[checkpoint_key] = self.checkpoint
        args['center_model']['use_learnable_center_estimation'] = True
        trainer = Trainer(0, 0, 1, args, use_distributed_data_parallel=False)
        self.addCleanup(trainer.cleanup)
        trainer.initialize_data_parallel()
        trainer.initialize()
        return trainer

    def assert_model_restored(self, model):
        torch.testing.assert_close(model.state_dict()[self.model_key], self.model_value)
        self.assertTrue(all(t.device.type == 'cpu' for t in model.state_dict().values()))

    def assert_center_restored(self, model):
        torch.testing.assert_close(model.state_dict()[self.center_key], self.center_value)

    def test_resume_roundtrips_numpy_logger_and_state(self):
        trainer = self.initialize_trainer('resume_path')
        self.assertEqual(trainer.start_epoch, 4)
        self.assertEqual(trainer.logger.data, {'train': [np.float64(0.125)]})
        self.assertIsInstance(trainer.logger.data['train'][0], np.float64)
        self.assertEqual(trainer.sample_loss_history, {0: np.float64(0.25)})
        self.assert_model_restored(trainer.model)
        self.assert_center_restored(trainer.center_model)

    def test_pretrained_model_accepts_numpy_logger_checkpoint(self):
        trainer = self.initialize_trainer('pretrained_model_path')
        self.assert_model_restored(trainer.model)
        self.assert_center_restored(trainer.center_model)

    def test_pretrained_center_accepts_numpy_logger_checkpoint(self):
        trainer = self.initialize_trainer('pretrained_center_model_path')
        self.assert_center_restored(trainer.center_model)

    def initialize_evaluator(self, separate_center):
        from test_polar import Evaluator

        args = config_args('test', self.root)
        args['checkpoint_path'] = self.weights_only_checkpoint if separate_center else self.checkpoint
        if separate_center:
            args['center_checkpoint_path'] = self.checkpoint
        evaluator = Evaluator(args)
        evaluator.initialize()
        processor = evaluator.processed_image_iter.main_processer
        self.assert_model_restored(processor.model)
        self.assert_center_restored(processor.get_center_model_list()[0]['model'])

    def test_evaluator_accepts_numpy_logger_checkpoint(self):
        self.initialize_evaluator(separate_center=False)

    def test_evaluator_separate_center_accepts_numpy_logger_checkpoint(self):
        self.initialize_evaluator(separate_center=True)

    def test_trusted_loader_explicitly_sets_policy_and_device(self):
        from utils.checkpoint import load_trusted_checkpoint

        real_load = torch.load
        calls = []

        def current_load(path, *, map_location, weights_only):
            calls.append((map_location, weights_only))
            return real_load(path, map_location=map_location, weights_only=weights_only)

        device = torch.device('cpu')
        with patch.object(torch, 'load', current_load):
            state = load_trusted_checkpoint(self.checkpoint, map_location=device)
        self.assertEqual(calls, [(device, False)])
        self.assertIsInstance(state['logger_data']['train'][0], np.float64)

    def test_trusted_loader_supports_torch_without_weights_only_parameter(self):
        from utils.checkpoint import load_trusted_checkpoint

        real_load = torch.load
        calls = []

        def legacy_load(path, map_location=None, **pickle_load_args):
            self.assertEqual(pickle_load_args, {})
            calls.append(map_location)
            # Simulate the older signature, but deserialize the real file.
            return real_load(path, map_location=map_location, weights_only=False)

        with patch.object(torch, 'load', legacy_load):
            state = load_trusted_checkpoint(self.checkpoint, map_location='cpu')
        self.assertEqual(calls, ['cpu'])
        self.assertIsInstance(state['logger_data']['train'][0], np.float64)

    def test_trusted_loader_does_not_retry_deserialization_type_errors(self):
        from utils.checkpoint import load_trusted_checkpoint

        calls = []

        def broken_load(path, *, map_location, weights_only):
            calls.append(path)
            raise TypeError('invalid checkpoint payload')

        with patch.object(torch, 'load', broken_load):
            with self.assertRaisesRegex(TypeError, 'invalid checkpoint payload'):
                load_trusted_checkpoint(self.checkpoint, map_location='cpu')
        self.assertEqual(calls, [self.checkpoint])


if __name__ == '__main__':
    unittest.main()
