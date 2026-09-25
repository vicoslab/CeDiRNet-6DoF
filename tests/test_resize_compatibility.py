"""Tensor resize must preserve the historical pre-antialias checkpoint input."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cloth_unfolding.infer_docker import Inference


class StopAfterInput(Exception):
    pass


class InputRecorder(torch.nn.Module):
    def forward(self, image):
        self.input = image
        raise StopAfterInput()


class ResizeCompatibilityTests(unittest.TestCase):
    def test_actual_inference_uses_historical_non_antialiased_resize(self):
        inference = Inference.__new__(Inference)
        inference.args = {'threshold': 0.1, 'size': [(11, 13)]}
        inference.input_crop = None
        inference.use_depth = False
        inference.device = torch.device('cpu')
        inference.model = InputRecorder()
        inference.center_model = Mock()
        image = np.random.default_rng(0).integers(0, 256, (37, 43, 3), dtype=np.uint8)
        expected = F.interpolate(ToTensor()(image)[None], size=(11, 13), mode='bilinear', align_corners=False)
        with self.assertRaises(StopAfterInput):
            inference.detect_grasp_points(image)
        torch.testing.assert_close(inference.model.input, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
