"""Released checkpoint preprocessing must agree with its recorded model config."""
import argparse
import importlib
import os
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'src'))
import infer_observation


class ModelPreprocessingTests(unittest.TestCase):
    def test_released_variants_match_recorded_preprocessing(self):
        cli = argparse.Namespace(center_threshold=0.1, device='cpu', top_k=1,
                                 pregrasp_approach_depth=0.025)
        for variant, env in infer_observation.VARIANTS.items():
            with self.subTest(variant=variant):
                recorded = runpy.run_path(str(ROOT / 'models/configs' / ('cedirnet6dof_' + variant + '.py')))['get_args']()
                built = infer_observation._build_inference_args(cli, env, 'checkpoint.pth')
                self.assertEqual(built['model']['kwargs']['fpn_args'], recorded['model']['kwargs']['fpn_args'])
                self.assertEqual(built['model']['kwargs']['in_channels'], recorded['model']['kwargs']['in_channels'])
                self.assertEqual(built['size'], recorded['size'])
                self.assertEqual(built['enable_crop_by_robot_position'], recorded.get('enable_crop_by_robot_position', False))
                self.assertFalse(built['cuda'])

    def test_rgbd_training_and_evaluation_match_published_std(self):
        with patch.dict(os.environ, {'DEPTH_ENCODING': '3d_arm_plane', 'CROP_MODE': 'robot'}):
            for name in ('config.cloth_6dof.train', 'config.cloth_6dof.test'):
                with self.subTest(name=name):
                    module = importlib.reload(importlib.import_module(name))
                    self.assertEqual(module.get_args()['model']['kwargs']['fpn_args']['depth_std'], 1)


if __name__ == '__main__':
    unittest.main()
