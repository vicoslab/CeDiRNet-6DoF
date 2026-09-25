#!/usr/bin/env python3
"""Portable, inference-only reconstruction of the competition/robot folder service.

Preserves observation_start/ -> grasp_predicted/grasp_pose.json. It is not the
historical competition model or robot controller. See deploy/README.md.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("crop_rgb", "crop_rgbd", "nocrop_rgb")


def required_files(variant):
    """Depth/calibration are required even for RGB-only 3D lifting."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown released model: {variant!r}; choose from {VARIANTS}")
    files = ["image_left.png", "depth_map.tiff", "camera_intrinsics.json", "camera_pose_in_world.json"]
    if variant != "nocrop_rgb":
        files += [f"arm_{side}{part}_pose_in_world.json" for side in ("left", "right") for part in ("", "_tcp")]
    return files


def _requested_variant(observation, default_variant, enabled):
    request = observation / "requested_model.json"
    if not enabled or not request.exists():
        return default_variant
    data = json.loads(request.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("requested_model"), str):
        raise ValueError("requested_model.json must contain a string 'requested_model'")
    return data["requested_model"] or default_variant


def _atomic_write(path, data):
    """Publish on the same filesystem, never expose a partially written result."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _output_folder(sample, output_root):
    base = output_root if output_root is not None else sample
    target = base / sample.name / "grasp_predicted" if output_root is not None else sample / "grasp_predicted"
    # The caller-selected root is trusted; sample-controlled subdirectories are not.
    current = target
    while current != base:
        if current.is_symlink():
            raise ValueError(f"Refusing symlink output directory: {current}")
        current = current.parent
    return target


def serve_once(serve_folder, predict, default_variant="crop_rgb", output_root=None, allow_requested_model=True):
    """Process one sorted scan, with an injectable (sample, variant) predictor.

The predictor returns (JSON pose list, JPEG bytes or None). Single consumer only:
producers must publish complete, immutable sample directories by atomic rename.
"""
    required_files(default_variant)
    root = Path(serve_folder).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve() if output_root is not None else None
    if output_root is not None and root in output_root.parents:
        raise ValueError("--output-root must not be nested inside --serve-folder")
    counts = dict(processed=0, skipped=0, deferred=0, failed=0)
    for sample in sorted(root.iterdir()):
        if sample.name.startswith(".") or sample.is_symlink() or not sample.is_dir():
            continue
        out = None
        try:
            out = _output_folder(sample, output_root)
            if (out / "grasp_pose.json").exists():
                counts["skipped"] += 1
                continue
            observation = sample / "observation_start"
            variant = _requested_variant(observation, default_variant, allow_requested_model)
            needed = required_files(variant)
            if not all((observation / name).is_file() for name in needed):
                counts["deferred"] += 1
                continue
            poses, debug_jpeg = predict(sample, variant)
            if not isinstance(poses, list):
                raise ValueError("Inference must return a JSON pose list")
            payload = json.dumps(poses, indent=2, allow_nan=False).encode("utf-8")
            out.mkdir(parents=True, exist_ok=True)
            if debug_jpeg is not None:
                _atomic_write(out / "frontal_image_grasp.jpg", debug_jpeg)
            error = out / "error.json"
            if error.exists():
                error.unlink()
            # Success marker LAST: consumers may immediately read the result.
            _atomic_write(out / "grasp_pose.json", payload)
            counts["processed"] += 1
        except Exception as exc:
            counts["failed"] += 1
            print(f"{sample.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            if out is not None:
                try:
                    out.mkdir(parents=True, exist_ok=True)
                    _atomic_write(out / "error.json", json.dumps({
                        "type": type(exc).__name__, "message": str(exc),
                    }, indent=2).encode("utf-8"))
                except OSError as output_error:
                    print(f"Could not write error.json: {output_error}", file=sys.stderr)
    return counts


class ReleasePredictor:
    """Use released inference helpers, not the removed config.eval CLI path.

Only one model is resident at a time. No automatic checkpoint downloads or
arbitrary Python config imports; mounted checkpoints are manifest-validated.
"""
    def __init__(self, args, helpers=None):
        if helpers is None:
            tools = str(REPO_ROOT / "tools")
            if tools not in sys.path:
                sys.path.insert(0, tools)
            helpers = importlib.import_module("infer_observation")
        self.helpers = helpers
        self.args = args
        self.definitions = helpers._load_model_definitions(Path(args.model_definitions))
        self.variant = None
        self.inference = None

    def _get_model(self, variant, observation):
        if self.variant == variant and self.inference is not None:
            return self.inference
        required_files(variant)
        entry = self.definitions.get(variant)
        if not isinstance(entry, dict) or entry.get("variant", variant) != variant:
            raise ValueError(f"No matching released manifest entry for {variant!r}")
        if entry.get("available") is False:
            raise ValueError(f"Released model {variant!r} is unavailable")
        checkpoint_name = entry.get("checkpoint")
        if not isinstance(checkpoint_name, str) or not checkpoint_name:
            raise ValueError(f"Missing checkpoint in manifest for {variant!r}")
        # Released manifest paths are repo-relative. A model mount replaces only
        # their directory; no request can supply paths or config/module names.
        checkpoint = (Path(self.args.model_root).expanduser() / Path(checkpoint_name).name
                      if self.args.model_root else REPO_ROOT / checkpoint_name)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing mounted checkpoint: {checkpoint}; provide --model-root")
        self.helpers._validate_checkpoint(checkpoint, entry)
        if self.inference is not None:
            self.inference.unload_model()
        self.inference = None
        self.variant = None
        model = self.helpers._build_inference(
            self.args, self.helpers.VARIANTS[variant], str(checkpoint),
            observation / "camera_intrinsics.json", observation / "camera_pose_in_world.json",
        )
        self.inference, self.variant = model, variant
        return model

    def __call__(self, sample, variant):
        import numpy as np

        sample = Path(sample)
        observation = sample / "observation_start"
        h = self.helpers
        infer = self._get_model(variant, observation)
        frame = h._read_rgb(observation / "image_left.png")
        depth = h._read_depth(observation / "depth_map.tiff")
        cam_K, cam_K_inv = h.load_camera_intrinsics(str(observation / "camera_intrinsics.json"))
        cam_in_3d = h.load_camera_world(str(observation / "camera_pose_in_world.json"))
        helper = h.FolderServerProcessing({"selected": infer}, str(sample), load_all_models=True)
        extra = {}
        if infer.normalize_depth_by_plane:
            from utils.utils_3d import read_competition_pose
            left, _ = read_competition_pose(str(observation / "arm_left_pose_in_world.json"))
            right, _ = read_competition_pose(str(observation / "arm_right_pose_in_world.json"))
            extra["normalize_by_depth_plane_points"] = (right, left)
        crop = infer.get_enabled_crop_by_robot_position()
        if crop:
            frame, depth, cam_K = helper.crop_data_by_robot_position(frame, depth, str(sample), cam_K, cam_in_3d, crop)
            cam_K_inv = np.linalg.inv(cam_K)
        debug, poses, scores = infer(frame, depth, cam_K=cam_K, cam_K_inv=cam_K_inv, cam_in_3d=cam_in_3d, **extra)
        poses, scores = helper.filter_unreachable_and_add_pregrasp_depth(
            poses, scores, str(sample), **(infer.get_postprocessing_kwargs() or {})
        )
        jpeg = None
        if self.args.save_debug_image:
            import cv2
            # Inference returns an OpenCV/BGR debug image (the input is RGB).
            ok, encoded = cv2.imencode(".jpg", debug)
            if not ok:
                raise ValueError("Could not encode debug JPEG")
            jpeg = encoded.tobytes()
        return h._json_pose_list(poses, scores), jpeg


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve-folder", required=True, help="Parent of immutable sample folders")
    parser.add_argument("--output-root", help="Separate output parent; default writes beside each observation")
    parser.add_argument("--model-definitions", default=str(REPO_ROOT / "models" / "model_definitions.json"))
    parser.add_argument("--model-root", help="Mounted folder containing manifest checkpoint basenames (offline only)")
    parser.add_argument("--variant", choices=VARIANTS, default="crop_rgb")
    parser.add_argument("--ignore-requested-model", action="store_true", help="Fixed-model competition adapter")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--center-threshold", type=float, default=0.1)
    parser.add_argument("--pregrasp-approach-depth", type=float, default=0.025, help="Local grasp Z offset in metres")
    parser.add_argument("--no-debug-image", dest="save_debug_image", action="store_false", default=True)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--once", action="store_true", help="One scan; exit nonzero on errors or incomplete samples")
    args = parser.parse_args(argv)
    if not math.isfinite(args.poll_interval) or args.poll_interval <= 0:
        parser.error("--poll-interval must be positive and finite")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if not math.isfinite(args.center_threshold) or args.center_threshold < 0:
        parser.error("--center-threshold must be nonnegative and finite")
    if not math.isfinite(args.pregrasp_approach_depth):
        parser.error("--pregrasp-approach-depth must be finite")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not Path(args.serve_folder).expanduser().is_dir():
        raise FileNotFoundError(f"Serve folder does not exist: {args.serve_folder}")
    predictor = ReleasePredictor(args)
    try:
        while True:
            counts = serve_once(args.serve_folder, predictor, default_variant=args.variant,
                                output_root=args.output_root, allow_requested_model=not args.ignore_requested_model)
            if args.once or counts["processed"] or counts["failed"]:
                print(json.dumps(counts), flush=True)
            if args.once:
                return int(bool(counts["failed"] or counts["deferred"]))
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
