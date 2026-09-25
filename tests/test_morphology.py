"""Regression contracts for the two mask-expansion consumers (CPU, no data).

The reference computes individual zero-padded neighborhoods directly, without
using a morphology implementation, pooling, or unfold.
"""
import ast
import math
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from models.center_groundtruth import CenterDirGroundtruth
from models.localization.centers import Conv1dMultiscaleLocalization
import test_polar


def neighborhood_reference(image, kernel, soft=False):
    result = torch.empty_like(image)
    before = (kernel - 1) // 2
    for b in range(image.shape[0]):
        for c in range(image.shape[1]):
            for y in range(image.shape[2]):
                for x in range(image.shape[3]):
                    values = []
                    for dy in range(kernel):
                        for dx in range(kernel):
                            yy, xx = y + dy - before, x + dx - before
                            values.append(float(image[b, c, yy, xx])
                                          if 0 <= yy < image.shape[2] and 0 <= xx < image.shape[3]
                                          else 0.0)
                    maximum = max(values)
                    result[b, c, y, x] = (maximum + math.log(sum(
                        math.exp(20 * (v - maximum)) for v in values)) / 20
                        if soft else maximum)
    return result


def expected_instances(instances, kernel, soft=False):
    result = instances.clone()
    background = instances == 0
    for label in instances.unique():
        if label > 0:
            mask = (instances == label)[None, None].float()
            dilated = neighborhood_reference(mask, kernel, soft)[0, 0]
            result[background & (dilated > (1 if soft else 0))] = label
    return result


class MaskExpansionTests(unittest.TestCase):
    def test_copied_module_and_imports_are_removed(self):
        self.assertFalse((ROOT / "src/utils/morphology.py").exists())
        for path in (ROOT / "src").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotEqual(node.module, "utils.morphology", str(path))
                elif isinstance(node, ast.Import):
                    self.assertNotIn("utils.morphology", [a.name for a in node.names], str(path))

    def test_hard_expansion_shapes_borders_kernels_and_label_dtypes(self):
        for dtype in (torch.uint8, torch.int16, torch.int32, torch.int64):
            for shape in ((1, 1), (1, 7), (7, 1), (5, 8)):
                for kernel in (1, 2, 3, 4, 9):
                    with self.subTest(dtype=dtype, shape=shape, kernel=kernel):
                        instances = torch.zeros(shape, dtype=dtype)
                        instances[0, 0] = 1
                        instances[-1, -1] = 2
                        op = CenterDirGroundtruth(extend_instance_mask_weights=kernel)
                        sample = {"instance": instances[None, None]}
                        out = op._generate_extented_instance_mask(sample, 0, None, None, None, None, None)
                        actual = out["gt_extended_instances"]
                        self.assertEqual(actual.shape, instances.shape)
                        self.assertEqual(actual.dtype, torch.int16)
                        self.assertTrue(torch.equal(actual, expected_instances(instances, kernel).short()))
                        self.assertEqual(out["gt_extend_instance_mask_weights"], kernel)
                        self.assertTrue(torch.equal(sample["instance"][0, 0], instances))

    def test_hard_disabled_empty_and_closest_center_modes(self):
        instances = torch.zeros((4, 6), dtype=torch.int64)
        for kernel in (0, 3):
            for closest in (False, True):
                op = CenterDirGroundtruth(extend_instance_mask_weights=kernel,
                                         ignore_instance_mask_and_use_closest_center=closest)
                for populated in (False, True):
                    mask = instances.clone()
                    if populated:
                        mask[1, 2] = 4
                    out = op._generate_extented_instance_mask(
                        {"instance": mask[None, None]}, 0, None, None, None, None, None)
                    if not kernel:
                        self.assertEqual(out, {})
                    else:
                        expected = mask if closest else expected_instances(mask, kernel)
                        self.assertTrue(torch.equal(out["gt_extended_instances"], expected.short()))

    def test_soft_scores_shapes_borders_dtypes_and_kernels(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((2, 1, 3, 5), (1, 1, 1, 4)):
                for kernel in (1, 2, 3, 4, 7):
                    with self.subTest(dtype=dtype, shape=shape, kernel=kernel):
                        x = torch.linspace(-2, 2, math.prod(shape), dtype=dtype).reshape(shape)
                        original = x.clone()
                        result = test_polar._soft_dilate_gt(x, kernel)
                        self.assertEqual(result.shape, x.shape)
                        self.assertEqual(result.dtype, dtype)
                        self.assertEqual(result.device, x.device)
                        self.assertTrue(torch.allclose(result, neighborhood_reference(x, kernel, True), atol=2e-7))
                        self.assertTrue(torch.equal(x, original))

    def test_soft_constant_and_gradient(self):
        x = torch.zeros((1, 1, 3, 5), dtype=torch.float64, requires_grad=True)
        result = test_polar._soft_dilate_gt(x, 3)
        self.assertTrue(torch.allclose(result, torch.full_like(x, math.log(9) / 20)))
        result.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue((x.grad > 0).all())

    def test_evaluator_soft_threshold_and_overlap(self):
        # The >1 threshold relies on SOFT scores; hard pooling would erase all labels.
        instances = torch.zeros((5, 8), dtype=torch.int32)
        instances[0, :2] = 1
        instances[1:3, 2:4] = 2
        difficult = (instances > 0)[None, None]
        evaluator = test_polar.Evaluator.__new__(test_polar.Evaluator)
        for kernel in (2, 3, 4, 9):
            evaluator.dilated_gt = kernel
            actual, actual_difficult = evaluator.dilate_gt_op(instances, difficult)
            self.assertTrue(torch.equal(actual, expected_instances(instances, kernel, True)))
            self.assertEqual(actual.dtype, instances.dtype)
            expected = neighborhood_reference(difficult.float(), kernel, True) > 1
            self.assertTrue(torch.equal(actual_difficult, expected))
            self.assertTrue(actual_difficult.any())
            self.assertEqual(evaluator.dilate_gt_op(None, None), (None, None))

    def test_evaluator_accepts_actual_2d_difficult_masks_and_thin_images(self):
        evaluator = test_polar.Evaluator.__new__(test_polar.Evaluator)
        evaluator.dilated_gt = 3
        for shape in ((5, 8), (7, 1), (1, 7), (1, 1)):
            instances = torch.ones(shape, dtype=torch.int64)
            difficult = torch.ones(shape, dtype=torch.bool)
            actual, actual_difficult = evaluator.dilate_gt_op(instances, difficult)
            self.assertEqual(actual.shape, instances.shape)
            self.assertTrue(torch.equal(actual, instances))
            self.assertEqual(actual_difficult.shape, difficult.shape)
            expected = neighborhood_reference(difficult[None, None].float(), 3, True)[0, 0] > 1
            self.assertTrue(torch.equal(actual_difficult, expected))

    def test_evaluator_disabled_is_identity(self):
        evaluator = test_polar.Evaluator.__new__(test_polar.Evaluator)
        instances, difficult = torch.ones((3, 4)), torch.zeros((1, 1, 3, 4))
        for kernel in (0, 1):
            evaluator.dilated_gt = kernel
            actual, actual_difficult = evaluator.dilate_gt_op(instances, difficult)
            self.assertIs(actual, instances)
            self.assertIs(actual_difficult, difficult)

    def test_localization_still_detects_peaks_without_erosion(self):
        op = Conv1dMultiscaleLocalization(exclude_border_px=0)
        response = torch.zeros((1, 1, 5, 8))
        response[0, 0, 0, 0] = 2
        response[0, 0, 3, 6] = 3
        result = op._get_local_max_indexes(response, min_distance=1, threshold_abs=1, exclude_border=0)
        self.assertTrue(torch.equal(result, torch.tensor([[0., 0., 0., 2.], [0., 6., 3., 3.]])))


if __name__ == "__main__":
    unittest.main()
