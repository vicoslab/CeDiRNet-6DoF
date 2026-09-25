"""Pose JSON and debug-image contracts for the public inference helpers."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import infer_observation as helper


class InferenceOutputTests(unittest.TestCase):
    def test_nonfinite_position_or_score_is_not_exported(self):
        valid = np.eye(4)
        invalid = valid.copy()
        invalid[0, 3] = np.nan
        self.assertEqual(helper._json_pose_list([invalid], [0.5]), [])
        self.assertEqual(helper._json_pose_list([valid], [np.inf]), [])
        self.assertEqual(len(helper._json_pose_list([valid], [0.5])), 1)

    def test_debug_image_preserves_inference_bgr_channels(self):
        inference = helper.Inference.__new__(helper.Inference)
        inference.C_K = inference.C_K_inv = inference.X_W_C = np.eye(3)
        inference.detect_grasp_points = lambda *args: ([], [], [], [], None)
        rgb = np.zeros((32, 48, 3), dtype=np.uint8)
        rgb[..., 0] = 240
        bgr, _, _ = inference.detect_return_in_3d_and_plot_grasp_points(rgb, np.ones((32, 48)))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'debug.jpg'
            helper._write_debug_image(target, bgr)
            decoded = cv2.imread(str(target))
            self.assertGreater(float(decoded[..., 2].mean()), 230)
            self.assertLess(float(decoded[..., 0].mean()), 10)

    def test_debug_image_write_failure_is_reported(self):
        with patch.object(cv2, 'imwrite', return_value=False):
            with self.assertRaises(OSError):
                helper._write_debug_image(Path('debug.jpg'), np.zeros((2, 2, 3), dtype=np.uint8))


if __name__ == '__main__':
    unittest.main()
