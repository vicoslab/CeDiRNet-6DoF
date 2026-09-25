#!/usr/bin/env python3
"""Run CeDiRNet-6DoF on one ICRA-style observation/sample folder.

Input can be either:
  * a sample folder containing observation_start/, or
  * the observation_start/ folder itself.

The output is an ICRA-compatible grasp_predicted/grasp_pose.json plus a debug image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Tuple
from urllib.request import urlretrieve

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cloth_unfolding.infer_docker import Inference, load_camera_intrinsics, load_camera_world, FolderServerProcessing  # noqa: E402

VARIANTS: Dict[str, Dict[str, str]] = {
    # Public names: RGB = depth is used only for 3D lifting; RGB-D = depth is also a network input.
    "crop_rgb": {"DEPTH_ENCODING": "off", "CROP_MODE": "robot"},
    "nocrop_rgb": {"DEPTH_ENCODING": "off", "CROP_MODE": "off"},
    "crop_rgbd": {"DEPTH_ENCODING": "3d_arm_plane", "CROP_MODE": "robot"},
}


def _cuda_enabled(device: str) -> bool:
    device = device.lower()
    if device == "auto":
        return torch.cuda.is_available()
    if device == "cuda":
        return True
    if device == "cpu":
        return False
    raise ValueError(f"Unsupported device {device!r}")


def _build_inference_args(cli_args: argparse.Namespace, variant_env: Dict[str, str], checkpoint: str) -> Dict[str, Any]:
    depth_encoding = variant_env["DEPTH_ENCODING"]
    crop_mode = variant_env["CROP_MODE"]
    use_arm_base_plane_depth = depth_encoding == "3d_arm_plane"
    use_depth = depth_encoding != "off"
    in_channels = 3 + (1 if use_depth else 0)
    num_fields = 9  # center directions/log-distance + 6 Euler orientation channels
    test_size = (1024, 1024)

    return {
        "threshold": float(cli_args.center_threshold),
        "use_segmnetation": False,
        "seg_threshold": 0.99,
        "use_depth": use_depth,
        "norm_depth": False if use_arm_base_plane_depth else True,
        "normalize_depth_by_plane": use_arm_base_plane_depth,
        "size": [test_size],
        "cuda": _cuda_enabled(cli_args.device),
        "top_k_predictions": int(cli_args.top_k),
        "nms_radius": 40,
        "enable_6dof_from_eulers": True,
        "enable_crop_by_robot_position": test_size if crop_mode == "robot" else False,
        "checkpoint_path": checkpoint,
        "center_checkpoint_path": checkpoint,
        "postprocessing_kwargs": {
            "robot_distance_thr": None,
            "pregrasp_aproach_depth": float(cli_args.pregrasp_approach_depth),
        },
        "model": {
            "name": "fpn",
            "kwargs": {
                "backbone": "tu-convnext_base",
                "use_depth": use_depth,
                "use_only_depth": False,
                "num_classes": [num_fields, 1, 1],
                "use_custom_fpn": True,
                "add_output_exp": False,
                "in_channels": in_channels,
                "fpn_args": {
                    "decoder_segmentation_head_channels": 64,
                    "classes_grouping": [(0, 1, 2, 9), (3, 6), (4, 7), (5, 8), (10,)],
                    "depth_mean": 0 if use_arm_base_plane_depth else 0.96,
                    "depth_std": 1.0 if use_arm_base_plane_depth else 0.075,
                },
                "init_decoder_gain": 0.1,
                "pretrained": False,
            },
        },
        "center_model": {
            "name": "CenterOrientationEstimator",
            "use_learnable_center_estimation": True,
            "kwargs": {
                "use_centerdir_radii": False,
                "use_magnitude_as_mask": True,
                "local_max_thr": 0.01,
                "mask_thr": 0.01,
                "hough_thr": 5000,
                "local_max_thr_use_abs": True,
                "suppression_by_mask": False,
                "ignore_centerdir_magnitude": True,
                "ignore_cls_prediction": True,
                "use_dilated_nn": True,
                "dilated_nn_args": {
                    "return_sigmoid": False,
                    "inner_ch": 16,
                    "inner_kernel": 3,
                    "dilations": [1, 4, 8, 12],
                    "use_centerdir_radii": False,
                    "use_centerdir_magnitude": False,
                    "use_cls_mask": False,
                },
                "scale_r": 1.0,
                "scale_r_gt": 1,
                "use_log_r": False,
                "use_log_r_base": "10",
                "enable_6dof": True,
            },
        },
        "num_vector_fields": num_fields,
    }


def _build_inference(cli_args: argparse.Namespace, variant_env: Dict[str, str], checkpoint: str,
                     camera_intrinsics_path: Path, camera_world_path: Path) -> Inference:
    infer_args = _build_inference_args(cli_args, variant_env, checkpoint)
    infer = Inference(infer_args, camera_calibration=str(camera_intrinsics_path), camera_in_world=str(camera_world_path))
    infer.load_model()
    return infer


def _load_model_definitions(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _maybe_download_checkpoint(path: Path, entry: Dict[str, Any]) -> None:
    if path.exists():
        return
    url = entry.get("url")
    if not url:
        raise FileNotFoundError(f"Checkpoint not found and no download URL is configured: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading checkpoint: {url} -> {path}", file=sys.stderr)
    tmp = path.with_suffix(path.suffix + ".download")
    try:
        urlretrieve(url, tmp)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _validate_checkpoint(path: Path, entry: Dict[str, Any]) -> None:
    expected_size = entry.get("size_bytes")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise ValueError(f"Checkpoint size mismatch for {path}: expected {expected_size}, got {path.stat().st_size}")
    expected_sha256 = entry.get("sha256")
    if expected_sha256 is not None:
        actual = _sha256(path)
        if actual.lower() != str(expected_sha256).lower():
            raise ValueError(f"Checkpoint sha256 mismatch for {path}: expected {expected_sha256}, got {actual}")


def _resolve_checkpoint(args: argparse.Namespace) -> Tuple[str, str]:
    variant = args.variant
    checkpoint = args.model_checkpoint
    entry: Dict[str, Any] = {}
    if args.model_definitions is not None:
        defs = _load_model_definitions(Path(args.model_definitions))
        if variant not in defs:
            model_variants = [k for k, v in defs.items() if isinstance(v, dict) and v.get("variant", k) in VARIANTS]
            raise KeyError(f"Variant {variant!r} not found in {args.model_definitions}. Available: {sorted(model_variants)}")
        entry = defs[variant]
        if entry.get("available") is False:
            raise ValueError(entry.get("note", f"Variant {variant!r} is not available"))
        checkpoint = checkpoint or entry.get("checkpoint") or entry.get("model")
        variant = entry.get("variant", variant)
    if checkpoint is None:
        raise ValueError("Provide --model-checkpoint or --model-definitions with the selected --variant")
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
    _maybe_download_checkpoint(checkpoint_path, entry)
    _validate_checkpoint(checkpoint_path, entry)
    return str(checkpoint_path), variant


def _resolve_sample_and_observation(path: Path, observation_subdir: str) -> Tuple[Path, Path]:
    path = path.expanduser().resolve()
    if (path / observation_subdir).is_dir():
        return path, path / observation_subdir
    if path.name == observation_subdir:
        return path.parent, path
    # Allow arbitrary one-observation folder without a sample wrapper.
    if (path / "image_left.png").exists():
        return path.parent, path
    raise FileNotFoundError(
        f"Could not find an observation folder. Expected {path}/{observation_subdir}/image_left.png "
        f"or {path}/image_left.png"
    )


def _read_rgb(path: Path) -> np.ndarray:
    frame_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _read_depth(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Depth map is required for 3D pose lifting but was not found: {path}")
    from PIL import Image
    return np.array(Image.open(path))


def _json_pose_list(pose_in_3d_list, scores):
    out = []
    for pose_in_3d, score in zip(pose_in_3d_list, scores):
        if not np.all(np.isfinite(pose_in_3d)) or not np.isfinite(score):
            continue
        euler = Rotation.from_matrix(pose_in_3d[:3, :3]).as_euler("xyz")
        if np.any(np.isnan(euler)):
            continue
        out.append({
            "position_in_meters": {
                "x": float(pose_in_3d[0, 3]),
                "y": float(pose_in_3d[1, 3]),
                "z": float(pose_in_3d[2, 3]),
            },
            "rotation_euler_xyz_in_radians": {
                "roll": float(euler[0]),
                "pitch": float(euler[1]),
                "yaw": float(euler[2]),
            },
            "score": float(score),
        })
    return out


def _write_debug_image(path: Path, debug_image: np.ndarray) -> None:
    # Inference returns BGR after drawing on its RGB input.
    if not cv2.imwrite(str(path), debug_image):
        raise OSError(f"Could not write debug image: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run public CeDiRNet-6DoF inference on one observation folder")
    parser.add_argument("--observation-folder", required=True, help="Sample folder or observation_start folder")
    parser.add_argument("--model-checkpoint", help="Path to a .pth checkpoint for the selected variant; overrides model definitions")
    parser.add_argument("--model-definitions", default=str(REPO_ROOT / "models" / "model_definitions.json"), help="JSON mapping variants to checkpoint paths and download URLs")
    parser.add_argument("--variant", default="crop_rgb", choices=sorted(VARIANTS), help="Released model variant")
    parser.add_argument("--output-dir", help="Output folder. Defaults to <sample>/grasp_predicted")
    parser.add_argument("--observation-subdir", default="observation_start")
    parser.add_argument("--image-name", default="image_left.png")
    parser.add_argument("--depth-name", default="depth_map.tiff")
    parser.add_argument("--camera-intrinsics-name", default="camera_intrinsics.json")
    parser.add_argument("--camera-in-world-name", default="camera_pose_in_world.json")
    parser.add_argument("--top-k", type=int, default=1, help="Maximum number of grasps to write")
    parser.add_argument("--center-threshold", type=float, default=0.1)
    parser.add_argument("--pregrasp-approach-depth", type=float, default=0.025, help="Extra offset along predicted gripper z-axis in meters")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Inference device")
    parser.add_argument("--save-debug-image", action="store_true", default=True)
    parser.add_argument("--no-debug-image", dest="save_debug_image", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint, variant = _resolve_checkpoint(args)
    variant_env = VARIANTS[variant]

    sample_dir, obs_dir = _resolve_sample_and_observation(Path(args.observation_folder), args.observation_subdir)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else sample_dir / "grasp_predicted"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = obs_dir / args.image_name
    depth_path = obs_dir / args.depth_name
    camera_intrinsics_path = obs_dir / args.camera_intrinsics_name
    camera_world_path = obs_dir / args.camera_in_world_name

    if not camera_intrinsics_path.exists():
        raise FileNotFoundError(f"Missing camera intrinsics: {camera_intrinsics_path}")
    if not camera_world_path.exists():
        raise FileNotFoundError(f"Missing camera pose in world: {camera_world_path}")

    frame = _read_rgb(image_path)
    depth = _read_depth(depth_path)
    cam_K, cam_K_inv = load_camera_intrinsics(str(camera_intrinsics_path))
    cam_in_3d = load_camera_world(str(camera_world_path))

    infer = _build_inference(args, variant_env, checkpoint, camera_intrinsics_path, camera_world_path)

    call_fn = infer
    helper = FolderServerProcessing({"selected": infer}, str(sample_dir), load_all_models=True)

    if infer.normalize_depth_by_plane:
        from utils.utils_3d import read_competition_pose
        left_base, _ = read_competition_pose(str(obs_dir / "arm_left_pose_in_world.json"))
        right_base, _ = read_competition_pose(str(obs_dir / "arm_right_pose_in_world.json"))
        from functools import partial
        call_fn = partial(call_fn, normalize_by_depth_plane_points=(right_base, left_base))

    if infer.get_enabled_crop_by_robot_position():
        frame, depth, cam_K = helper.crop_data_by_robot_position(
            frame, depth, str(sample_dir), cam_K, cam_in_3d, infer.get_enabled_crop_by_robot_position()
        )
        cam_K_inv = np.linalg.inv(cam_K)

    debug_image, pose_in_3d_list, scores = call_fn(frame, depth, cam_K=cam_K, cam_K_inv=cam_K_inv, cam_in_3d=cam_in_3d)
    pose_in_3d_list, scores = helper.filter_unreachable_and_add_pregrasp_depth(
        pose_in_3d_list, scores, str(sample_dir), **(infer.get_postprocessing_kwargs() or {})
    )

    pose_json = _json_pose_list(pose_in_3d_list, scores)
    out_json = output_dir / "grasp_pose.json"
    out_json.write_text(json.dumps(pose_json, indent=2, allow_nan=False), encoding="utf-8")

    if args.save_debug_image:
        _write_debug_image(output_dir / "frontal_image_grasp.jpg", debug_image)

    print(json.dumps({
        "variant": variant,
        "checkpoint": checkpoint,
        "input": str(obs_dir),
        "output_json": str(out_json),
        "num_grasps": len(pose_json),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
