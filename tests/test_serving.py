"""Folder transport tests use fake inference; no weights, GPU or network required."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]


class ServingTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "tools" / "serve_observations.py"
        self.assertTrue(path.is_file(), "portable folder server is missing")
        spec = importlib.util.spec_from_file_location("serve_observations", path)
        assert spec is not None and spec.loader is not None
        self.server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.server)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "input"
        self.root.mkdir()
        self.predict = Mock(return_value=([{
            "position_in_meters": {"x": 0.1, "y": 0.2, "z": 0.3},
            "rotation_euler_xyz_in_radians": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "score": 0.9,
        }], b"fake JPEG for transport test"))

    def sample(self, name="sample", variant="nocrop_rgb", requested=None):
        sample = self.root / name
        obs = sample / "observation_start"
        obs.mkdir(parents=True)
        for filename in self.server.required_files(variant):
            (obs / filename).write_bytes(b"placeholder for injected inference")
        if requested is not None:
            (obs / "requested_model.json").write_text(json.dumps({"requested_model": requested}))
        return sample

    def scan(self, **kwargs):
        return self.server.serve_once(self.root, self.predict, default_variant="nocrop_rgb", **kwargs)

    def test_historical_layout_and_success_marker_are_preserved(self):
        sample = self.sample()
        result = self.scan()
        self.assertEqual(result, {"processed": 1, "skipped": 0, "deferred": 0, "failed": 0})
        self.predict.assert_called_once_with(sample, "nocrop_rgb")
        out = sample / "grasp_predicted"
        self.assertEqual(json.loads((out / "grasp_pose.json").read_text()), self.predict.return_value[0])
        self.assertEqual((out / "frontal_image_grasp.jpg").read_bytes(), self.predict.return_value[1])
        self.assertFalse(list(out.glob("*.tmp")))
        self.assertEqual(self.scan()["skipped"], 1)
        self.assertEqual(self.predict.call_count, 1)

    def test_success_marker_is_published_after_debug_image(self):
        from unittest.mock import patch
        sample = self.sample()
        out = sample / "grasp_predicted"
        write = self.server._atomic_write
        published = []

        def checked_write(path, payload):
            self.assertFalse((out / "grasp_pose.json").exists())
            write(path, payload)
            published.append(path.name)

        with patch.object(self.server, "_atomic_write", side_effect=checked_write):
            self.assertEqual(self.scan()["processed"], 1)
        self.assertEqual(published, ["frontal_image_grasp.jpg", "grasp_pose.json"])

    def test_debug_write_failure_cannot_publish_success(self):
        from unittest.mock import patch
        sample = self.sample()
        write = self.server._atomic_write

        def failing_write(path, payload):
            if path.suffix == ".jpg":
                raise OSError("test disk failure")
            write(path, payload)

        with patch.object(self.server, "_atomic_write", side_effect=failing_write):
            self.assertEqual(self.scan()["failed"], 1)
        self.assertFalse((sample / "grasp_predicted" / "grasp_pose.json").exists())
        self.assertTrue((sample / "grasp_predicted" / "error.json").exists())

    def test_nested_output_root_is_rejected(self):
        self.sample()
        with self.assertRaisesRegex(ValueError, "nested"):
            self.scan(output_root=self.root / "results")
        self.predict.assert_not_called()

    def test_robot_request_selects_only_a_released_variant(self):
        sample = self.sample(variant="crop_rgbd", requested="crop_rgbd")
        self.assertEqual(self.scan()["processed"], 1)
        self.predict.assert_called_once_with(sample, "crop_rgbd")

    def test_competition_fixed_model_ignores_request(self):
        sample = self.sample(requested="old_private_model")
        self.assertEqual(self.scan(allow_requested_model=False)["processed"], 1)
        self.predict.assert_called_once_with(sample, "nocrop_rgb")

    def test_empty_historical_request_uses_default(self):
        sample = self.sample(requested="")
        self.scan()
        self.predict.assert_called_once_with(sample, "nocrop_rgb")

    def test_unknown_model_is_an_error_not_empty_success(self):
        sample = self.sample(requested="config.eval.unreleased")
        self.assertEqual(self.scan()["failed"], 1)
        self.predict.assert_not_called()
        self.assertFalse((sample / "grasp_predicted" / "grasp_pose.json").exists())
        error = json.loads((sample / "grasp_predicted" / "error.json").read_text())
        self.assertIn("Unknown", error["message"])

    def test_malformed_request_does_not_stop_other_samples(self):
        sample = self.sample("a")
        (sample / "observation_start" / "requested_model.json").write_text("[]")
        self.sample("b")
        result = self.scan()
        self.assertEqual((result["failed"], result["processed"]), (1, 1))

    def test_incomplete_observation_is_deferred_until_complete(self):
        sample = self.sample()
        depth = sample / "observation_start" / "depth_map.tiff"
        depth.unlink()
        self.assertEqual(self.scan()["deferred"], 1)
        self.predict.assert_not_called()
        self.assertFalse((sample / "grasp_predicted" / "grasp_pose.json").exists())
        depth.write_bytes(b"ready")
        self.assertEqual(self.scan()["processed"], 1)

    def test_crop_requires_all_four_robot_pose_files(self):
        files = self.server.required_files("crop_rgb")
        self.assertIn("arm_left_tcp_pose_in_world.json", files)
        self.assertIn("arm_right_tcp_pose_in_world.json", files)
        self.assertEqual(len(files), 8)
        self.assertEqual(len(self.server.required_files("nocrop_rgb")), 4)

    def test_inference_failure_is_retryable_and_not_success(self):
        sample = self.sample()
        self.predict.side_effect = RuntimeError("test failure")
        self.assertEqual(self.scan()["failed"], 1)
        out = sample / "grasp_predicted"
        self.assertTrue((out / "error.json").exists())
        self.assertFalse((out / "grasp_pose.json").exists())
        self.predict.side_effect = None
        self.assertEqual(self.scan()["processed"], 1)
        self.assertFalse((out / "error.json").exists())

    def test_no_detections_is_valid_success(self):
        sample = self.sample()
        self.predict.return_value = ([], None)
        self.assertEqual(self.scan()["processed"], 1)
        self.assertEqual(json.loads((sample / "grasp_predicted" / "grasp_pose.json").read_text()), [])

    def test_nonfinite_pose_is_not_published(self):
        sample = self.sample()
        self.predict.return_value[0][0]["score"] = float("nan")
        self.assertEqual(self.scan()["failed"], 1)
        self.assertFalse((sample / "grasp_predicted" / "grasp_pose.json").exists())

    def test_separate_output_root_keeps_input_unchanged(self):
        sample = self.sample()
        output = Path(self.temp.name) / "output"
        self.assertEqual(self.scan(output_root=output)["processed"], 1)
        self.assertFalse((sample / "grasp_predicted").exists())
        self.assertTrue((output / "sample" / "grasp_predicted" / "grasp_pose.json").is_file())
        self.assertEqual(self.scan(output_root=output)["skipped"], 1)

    def test_hidden_staging_and_symlink_samples_are_ignored(self):
        self.sample(".uploading")
        external = Path(self.temp.name) / "external"
        external.mkdir()
        (self.root / "linked").symlink_to(external, target_is_directory=True)
        self.assertEqual(self.scan()["processed"], 0)
        self.predict.assert_not_called()

    def test_output_symlink_is_rejected_without_external_writes(self):
        sample = self.sample()
        external = Path(self.temp.name) / "external"
        external.mkdir()
        (sample / "grasp_predicted").symlink_to(external, target_is_directory=True)
        self.assertEqual(self.scan()["failed"], 1)
        self.assertEqual(list(external.iterdir()), [])
        self.predict.assert_not_called()

    def test_cli_help_does_not_import_torch(self):
        result = subprocess.run([sys.executable, "-S", str(ROOT / "tools" / "serve_observations.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--serve-folder", result.stdout)
        self.assertIn("--model-root", result.stdout)

    def test_two_images_are_inference_only_reconstructions(self):
        for name in ("competition", "robot"):
            dockerfile = ROOT / "deploy" / name / "Dockerfile"
            self.assertTrue(dockerfile.is_file(), f"missing {name} Docker recipe")
            text = dockerfile.read_text()
            self.assertIn("ARG BASE_IMAGE=cedirnet-6dof:latest", text)
            self.assertIn("serve_observations.py", text)
            self.assertIn('"--model-root", "/models"', text)
            self.assertIn('"--serve-folder", "/data"', text)
            self.assertEqual("--ignore-requested-model" in text, name == "competition")
            for forbidden in ("git clone", "COPY . ", "--privileged", "--network host", "roslaunch", "*.pth"):
                self.assertNotIn(forbidden, text)

    def test_cli_rejects_invalid_poll_and_top_k(self):
        for extra in (["--poll-interval", "0"], ["--poll-interval", "nan"], ["--top-k", "0"]):
            with self.assertRaises(SystemExit):
                self.server.parse_args(["--serve-folder", str(self.root)] + extra)

    def test_cli_once_reports_success_and_incomplete_work(self):
        from unittest.mock import patch
        sample = self.sample()
        with patch.object(self.server, "ReleasePredictor", return_value=self.predict):
            code = self.server.main(["--serve-folder", str(self.root), "--variant", "nocrop_rgb", "--once"])
        self.assertEqual(code, 0)
        (sample / "grasp_predicted" / "grasp_pose.json").unlink()
        (sample / "observation_start" / "depth_map.tiff").unlink()
        with patch.object(self.server, "ReleasePredictor", return_value=self.predict):
            code = self.server.main(["--serve-folder", str(self.root), "--variant", "nocrop_rgb", "--once"])
        self.assertEqual(code, 1)

    def test_missing_mounted_checkpoint_never_downloads(self):
        args = self.server.parse_args(["--serve-folder", str(self.root), "--model-root", str(self.root), "--variant", "crop_rgb"])
        helper = Mock()
        helper.VARIANTS = {"crop_rgb": {"CROP_MODE": "robot", "DEPTH_ENCODING": "off"}}
        helper._load_model_definitions.return_value = {"crop_rgb": {"variant": "crop_rgb", "checkpoint": "models/missing.pth", "url": "https://example.invalid/never-download"}}
        predictor = self.server.ReleasePredictor(args, helpers=helper)
        with self.assertRaisesRegex(FileNotFoundError, "mounted"):
            predictor(self.sample(variant="crop_rgb"), "crop_rgb")
        helper._maybe_download_checkpoint.assert_not_called()
        helper._build_inference.assert_not_called()

    def test_predictor_reuses_loaded_model_and_calibrates_each_request(self):
        import numpy as np
        args = self.server.parse_args(["--serve-folder", str(self.root), "--model-root", str(self.root), "--variant", "nocrop_rgb", "--no-debug-image"])
        (self.root / "test.pth").write_bytes(b"fake weights validated by injected helper")
        helper = Mock()
        helper.VARIANTS = {"nocrop_rgb": {"CROP_MODE": "off", "DEPTH_ENCODING": "off"}}
        helper._load_model_definitions.return_value = {"nocrop_rgb": {"variant": "nocrop_rgb", "checkpoint": "models/test.pth"}}
        helper._read_rgb.return_value = np.zeros((8, 8, 3), dtype=np.uint8)
        helper._read_depth.return_value = np.ones((8, 8), dtype=np.float32)
        helper.load_camera_intrinsics.return_value = (np.eye(3), np.eye(3))
        helper.load_camera_world.return_value = np.eye(4)
        model = Mock()
        model.normalize_depth_by_plane = False
        model.get_enabled_crop_by_robot_position.return_value = False
        model.get_postprocessing_kwargs.return_value = {}
        model.return_value = (helper._read_rgb.return_value, [], [])
        helper._build_inference.return_value = model
        helper.FolderServerProcessing.return_value.filter_unreachable_and_add_pregrasp_depth.return_value = ([], [])
        helper._json_pose_list.return_value = []
        predictor = self.server.ReleasePredictor(args, helpers=helper)
        for name in ("a", "b"):
            self.assertEqual(predictor(self.sample(name), "nocrop_rgb"), ([], None))
        self.assertEqual(helper._build_inference.call_count, 1)
        self.assertEqual(helper._validate_checkpoint.call_count, 1)
        self.assertEqual(helper.load_camera_intrinsics.call_count, 2)
        self.assertEqual(model.call_count, 2)
        self.assertIn("cam_K", model.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
