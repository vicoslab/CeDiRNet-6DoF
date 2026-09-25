# Competition and robot-experiment deployment

These are **portable reconstructions of the inference services**, not archived
competition binaries and not robot controllers. They reuse the released
`tools/infer_observation.py` model/preprocessing helpers and the existing
`tools/cloth_unfolding/infer_docker.py` geometry helpers. They do **not**
invoke that historical module's CLI, which expects the removed `config.eval`
configuration layout.

| Recipe | Preserved interface | Model behavior |
| --- | --- | --- |
| `competition/Dockerfile` | ICRA 2024 sample-folder polling and grasp JSON | Fixed released variant, ignores `requested_model.json` |
| `robot/Dockerfile` | Robot-experiment sample-folder polling, grasp JSON, optional model request | Selects a released variant per request; keeps one model resident |

Both are headless, mount weights at runtime, and do not download weights. Neither
runs ROS, echolib, a camera, a competition submission client, or motion commands.
No hardware device, privileged mode, host networking, or display mount is needed.
A separate, independently safety-checked robot controller must consume predictions.
No collision checking or reachability guarantee is provided.

## Build and run

From the repository root, first build the public dependency/code image, then the
two service images. The dependency stack is inherited from the root Dockerfile;
these recipes do not try to recreate the older competition environment.

```bash
docker build -t cedirnet-6dof:latest .
docker build -f deploy/competition/Dockerfile -t cedirnet-competition:portable .
docker build -f deploy/robot/Dockerfile -t cedirnet-robot:portable .
```

Use `--build-arg BASE_IMAGE=your-base-tag` to select another compatible public base.
Provision trusted checkpoints listed in `models/model_definitions.json` into a
local `weights/` directory, using their declared filenames. The server checks the
manifest's sizes and hashes before loading each model. Checkpoints may contain
Python pickle data: mount only trusted files and keep the manifest trusted too.
A missing checkpoint fails that request; it does not initiate a download.

Prepare `requests/` with samples as described below. The mounted directory must
be writable by the chosen UID/GID. The image defaults to UID/GID 1000; the examples
explicitly use the invoking user's IDs.

```bash
mkdir -p requests weights
docker run --rm --init --user "$(id -u):$(id -g)" --network none \
  --mount type=bind,src="$PWD/weights",dst=/models,readonly \
  --mount type=bind,src="$PWD/requests",dst=/data \
  cedirnet-competition:portable

# Same mounts, but honors a released model name in requested_model.json:
docker run --rm --init --user "$(id -u):$(id -g)" --network none \
  --mount type=bind,src="$PWD/weights",dst=/models,readonly \
  --mount type=bind,src="$PWD/requests",dst=/data \
  cedirnet-robot:portable
```

Add `--gpus all` before the image name on a configured NVIDIA host. Without it,
`--device auto` uses the CPU. Neither recipe has been Docker-built in the release
preparation environment: no Docker executable was available. Host-side tests do
not establish image buildability, in-container CUDA visibility, or latency.

For **read-only observations and separate output**, use the robot image as follows
(the same flags work with the competition image):

```bash
mkdir -p results
docker run --rm --init --user "$(id -u):$(id -g)" --network none \
  --mount type=bind,src="$PWD/weights",dst=/models,readonly \
  --mount type=bind,src="$PWD/requests",dst=/data,readonly \
  --mount type=bind,src="$PWD/results",dst=/results \
  cedirnet-robot:portable \
  --serve-folder /data --model-root /models --output-root /results \
  --variant crop_rgb --device cpu --once
```

Arguments after the image replace its default command, so include
`--serve-folder` and `--model-root` when overriding options. The competition image
always includes `--ignore-requested-model` in its entrypoint.

Equivalent local invocation:

```bash
python tools/serve_observations.py --serve-folder requests --model-root weights \
  --variant crop_rgb --device cpu --once
python -m unittest discover -s tests -p test_serving.py -v
```

The tests inject fake inference and require no checkpoints or network. They test
the transport and adapter contract, not model accuracy. A separate host-side CPU
smoke test with the manifest-verified `crop_rgb` and `crop_rgbd` checkpoints and a
real ICRA observation produced one finite grasp and a decodable 1024×1024 JPEG
for each variant using Python 3.8/PyTorch 1.9.1. The RGB-D run used
`requested_model.json` to override the default RGB variant. This verifies the
host path only, not the newer container stack.

## Folder protocol

```text
requests/
  sample_000001/
    observation_start/
      image_left.png
      depth_map.tiff
      camera_intrinsics.json
      camera_pose_in_world.json
      arm_left_pose_in_world.json       # required for crop_* variants
      arm_left_tcp_pose_in_world.json   # required for crop_* variants
      arm_right_pose_in_world.json      # required for crop_* variants
      arm_right_tcp_pose_in_world.json  # required for crop_* variants
      requested_model.json             # optional; robot image only
    grasp_predicted/                    # created by server
      frontal_image_grasp.jpg           # unless --no-debug-image
      grasp_pose.json                   # final success marker
```

RGB and depth must be registered to the same camera. Depth is in **metres**, even
for RGB-only variants: the depth map is still used to lift detections into 3D.
Camera intrinsics use `focal_lengths_in_pixels.{fx,fy}` and
`principal_point_in_pixels.{cx,cy}`. Camera/robot poses use the same position and
Euler fields as the output below, without a required `score`.

Robot requests may contain `{"requested_model": "crop_rgbd"}`. Only `crop_rgb`,
`nocrop_rgb`, and `crop_rgbd` are accepted, subject to manifest availability.
A missing request file or `{"requested_model": ""}` selects `--variant` (default
`crop_rgb`). Internal experiment aliases and arbitrary config/module names are
not accepted. The competition adapter ignores this file and uses `--variant`.

`grasp_pose.json` is a JSON list, ordered by prediction score, in **world**
coordinates. By default it contains at most one grasp (`--top-k`):

```json
[
  {
    "position_in_meters": {"x": 0.1, "y": 0.2, "z": 0.3},
    "rotation_euler_xyz_in_radians": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    "score": 0.9
  }
]
```

The above values illustrate the schema; they are not measured predictions.
`--pregrasp-approach-depth` defaults to 0.025 metres along the local predicted
Z axis, matching the public one-observation helper. Set it to `0` if an external
consumer already adds the offset, to avoid applying it twice.

### Producer/consumer rules

- Run **one server per queue**; concurrent consumers are not supported.
- Stage a complete sample outside the queue (on the same filesystem), then
  atomically rename it into the queue. Hidden staging directories are ignored.
  File-presence checks alone cannot detect a file still being uploaded.
- Treat published samples as immutable and use unique names. Existing
  `grasp_pose.json` files are skipped, including on restart. To reprocess a sample,
  remove its success marker after stopping consumers, or publish a new name.
- Debug output is published first, followed by an atomic success JSON rename.
  A valid no-detection result is `[]`. Non-finite numeric output is an error.
- Incomplete samples are deferred. Other failures write `error.json`, **not** a
  fake empty success result; they are retried on the next scan. Repair or remove
  a permanently failing sample to stop retries. A successful retry removes the
  error file. Consumers must watch `error.json` as well as success to avoid waits
  with no explanation. `--once` exits 1 on failed or incomplete samples, 0 otherwise.
- `--output-root` writes `<output-root>/<sample>/grasp_predicted/` instead of
  modifying the input tree. It must not be nested under the input root.
- Symlink sample directories are skipped; symlink output subdirectories are
  rejected. The queue and mounted paths must still be trusted local storage,
  not a hostile multi-user upload service.

## Historical provenance and limits

The original competition detector used an Ubuntu 20.04/CUDA 11.7.1/cuDNN 8
ROS/echolib image with PyTorch 1.13.1 and torchvision 0.14.1. Its selected
configuration was
`v2+seg+cropping_6dof_euler_no_depth_no_rot_rand_bg=coco_epoch=100`: RGB,
ConvNeXt-base FPN, nine center/orientation fields and two extra output channels.
The declared architecture family matches the released RGB helper, **but the
historical competition checkpoint is not one of the three released manifest
entries and checkpoint compatibility/numerical equivalence is not established**.

The historical competition setup used a 768×1024 robot crop, threshold 0.3,
a 2 cm depth offset, and external GUI/controller postprocessing (including a
2.5 cm approach offset and a 25–80 cm reach filter). Its adapter also had different
pixel scaling behavior. The public helper uses a 1024×1024 crop/resize, threshold
0.1 by default, no historical depth offset, and its own pregrasp postprocessing.
Therefore these recipes must **not** be described as reproducing competition
scores or the exact historical grasp policy. We have not silently substituted
modern weights under a historical model name. No competition GUI, submission
client, calibration files, private source checkout, or original image layers are
copied into this release.

The robot-experiment service extended the folder protocol with
`requested_model.json`, dynamic model loading, arm-plane depth normalization,
robot-position cropping and pregrasp postprocessing. An older ROS/CUDA 11.7
recipe and a later Ubuntu 22.04/CUDA 12.8.1/PyTorch 2.7 recipe existed. The portable
robot image reuses the released geometry/preprocessing code, replaces the removed
`config.eval` model loader with the public manifest, and omits unreleased model
aliases, preemptive/all-model loading, and the optional `input_space.jpg` robot-axis
diagnostic. Hardware integration and exact historical environment replication
remain outside this release.
