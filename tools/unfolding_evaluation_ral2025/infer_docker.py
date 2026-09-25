#!/usr/bin/python
import os, time
import cv2
import csv
import sys
import inspect

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

import numpy as np

import torch
from torchvision import transforms
from models import get_model, get_center_model

AXIS_COLOR = [(255,0,0), (0,255,0), (0,0,255)]

import json
from utils.utils_3d import get_homogeneous_matrix


_TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY = "weights_only" in inspect.signature(torch.load).parameters


def torch_load_compat(path, **kwargs):
    if _TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY:
        kwargs.setdefault("weights_only", False)

    return torch.load(path, **kwargs)

def load_camera_intrinsics(filename):

    with open(filename, 'r') as f:
        data = json.load(f)

        # Create camera matrix
        C_K = np.array([[data['focal_lengths_in_pixels']['fx'], 0, data['principal_point_in_pixels']['cx']],
                        [0, data['focal_lengths_in_pixels']['fy'], data['principal_point_in_pixels']['cy']],
                        [0, 0, 1]])

        C_K_inv = np.linalg.inv(C_K)

    return C_K, C_K_inv

def load_camera_world(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

        cam_in_3d = dict(pos=np.array([data['position_in_meters']['x'], data['position_in_meters']['y'], data['position_in_meters']['z']]),
                                rot=np.array([data['rotation_euler_xyz_in_radians']['roll'],
                                            data['rotation_euler_xyz_in_radians']['pitch'],
                                            data['rotation_euler_xyz_in_radians']['yaw']]))

    return get_homogeneous_matrix(cam_in_3d["pos"], cam_in_3d["rot"])


class Inference:
    def __init__(self, args, camera_calibration = None, camera_in_world = None):
        if args.get('cudnn_benchmark'):
            torch.backends.cudnn.benchmark = True

        self.args = args
        self.show_regression_output = args.get('show_regression_output')
        self.input_crop = args.get('input_crop')
        self.output_crop = args.get('crop') if args.get('crop') else args.get('output_crop')

        self.top_k_predictions = args.get('top_k_predictions')

        self.enable_6dof = args['center_model']['kwargs'].get('enable_6dof')
        self.enable_6dof_from_eulers = args.get('enable_6dof_from_eulers')
        self.enable_orinetation_in_world_coord = args.get('enable_world_pose')

        self.enabled_confidence_score = args['center_model']['kwargs'].get('use_orientation_confidence_score')
        self.use_only_confidence_prediction_score = args.get('use_only_confidence_prediction_score')

        if self.use_only_confidence_prediction_score:
            assert self.enabled_confidence_score, 'Missing confidence score setting in center model'

        self.use_depth = args['use_depth']
        self.norm_depth = args['norm_depth'] if 'norm_depth' in args else True
        self.normalize_depth_by_plane = args['normalize_depth_by_plane'] if 'normalize_depth_by_plane' in args else False

        self.enable_crop_by_robot_position = args.get('enable_crop_by_robot_position')
        # depth offset is in centimeters - should be in meters for
        self.depth_offset = (args['depth_offset']) / 100 if 'depth_offset' in args else 0
        print("using depth offset: ", self.depth_offset)

        # set device
        self.device = torch.device("cuda:0" if args['cuda'] else "cpu")
        print("device: ", self.device)

        self.C_K, self.C_K_inv, self.X_W_C = None, None, None

        if camera_calibration is not None:
            assert camera_calibration is not None, "Missing camera-in-world path."
            self.C_K, self.C_K_inv = load_camera_intrinsics(camera_calibration)

        if camera_in_world is not None:
            assert camera_in_world is not None, "Missing camera-in-world path."
            self.X_W_C = load_camera_world(camera_in_world)

        self.postprocessing_kwargs = args.get('postprocessing_kwargs')

        self.model, self.center_model = None, None


    def load_model(self):
        self.model, self.center_model = self._construct_dataset_and_processing(self.args, self.device)

    def unload_model(self):
        self.model = None
        self.center_model = None

    def is_model_loaded(self):
        return self.model is not None and self.center_model is not None

    def get_enabled_crop_by_robot_position(self):
        return self.enable_crop_by_robot_position

    def get_postprocessing_kwargs(self):
        return self.postprocessing_kwargs

    def __call__(self, *args, **kwargs):
        return self.detect_return_in_3d_and_plot_grasp_points(*args, **kwargs)

    def _construct_dataset_and_processing(self, args, device):

        ###################################################################################################
        # load model
        model = get_model(args['model']['name'], args['model']['kwargs'])
        model.init_output(args['num_vector_fields'])
        model = torch.nn.DataParallel(model).to(device)

        # prepare center_model and center_estimator based on number of center_checkpoint_path that will need to be processed

        center_checkpoint_path = args.get('center_checkpoint_path')

        center_model = get_center_model(args['center_model']['name'], args['center_model']['kwargs'],
                                        is_learnable=args['center_model'].get('use_learnable_center_estimation'),
                                        use_fast_estimator=False)

        center_model.init_output(args['num_vector_fields'])
        center_model = torch.nn.DataParallel(center_model).to(device)

        ###################################################################################################
        # load snapshot
        if os.path.exists(args['checkpoint_path']):
            print('Loading from "%s"' % args['checkpoint_path'])
            state = torch_load_compat(args['checkpoint_path'], map_location=device)
            if 'model_state_dict' in state: model.load_state_dict(state['model_state_dict'], strict=True)
            if not args.get('center_checkpoint_path') and 'center_model_state_dict' in state and args['center_model'].get('use_learnable_center_estimation'):
                print('Loading center model from "%s"' % args['checkpoint_path'])
                center_model.load_state_dict(state['center_model_state_dict'], strict=False)
        else:
            raise Exception('checkpoint_path {} does not exist!'.format(args['checkpoint_path']))

        if args['center_model'].get('use_learnable_center_estimation'):
            if os.path.exists(center_checkpoint_path):
                print('Loading center model from "%s"' % center_checkpoint_path)
                state = torch_load_compat(center_checkpoint_path, map_location=device)
                if 'center_model_state_dict' in state:
                    if 'module.instance_center_estimator.conv_start.0.weight' in state['center_model_state_dict']:
                        checkpoint_input_weights = state['center_model_state_dict']['module.instance_center_estimator.conv_start.0.weight']
                        center_input_weights = center_model.module.instance_center_estimator.conv_start[0].weight
                        if checkpoint_input_weights.shape != center_input_weights.shape:
                            state['center_model_state_dict']['module.instance_center_estimator.conv_start.0.weight'] = checkpoint_input_weights[:,:2,:,:]

                            print('WARNING: #####################################################################################################')
                            print('WARNING: center input shape mismatch - will load weights for only the first two channels, is this correct ?!!!')
                            print('WARNING: #####################################################################################################')

                    center_model.load_state_dict(state['center_model_state_dict'], strict=False)
            else:
                raise Exception('checkpoint_path {} does not exist!'.format(center_checkpoint_path))

        return model, center_model

    def detect_grasp_points(self, input_image, input_depth = None, normalize_by_depth_plane_points=None):
        to_tensor = transforms.ToTensor()
        # y_pred = []
        # target = []
        args = self.args

        threshold = args["threshold"]
        seg_threshold = args.get("seg_threshold")

        seg_output = None

        with torch.no_grad():
            self.model.eval()
            self.center_model.eval()

            scores = []
            points = []
            vectors = []
            angles = []

            if self.input_crop:
                height = input_image.shape[0]
                width = input_image.shape[1]

                top_crop = int(self.input_crop['top'] if 'top' in self.input_crop else 0)
                left_crop = int(self.input_crop['left'] if 'left' in self.input_crop else 0)
                bottom_crop = int(height - (self.input_crop['bottom'] if 'bottom' in self.input_crop else 0))
                right_crop = int(width - (self.input_crop['right'] if 'right' in self.input_crop else 0))

                input_image = input_image[top_crop:bottom_crop, left_crop:right_crop]
                if depth is not None:
                    depth = depth[top_crop:bottom_crop, left_crop:right_crop]

            if 'size' not in args:
                args['size'] = None

            if type(args["size"]) not in [list, tuple]:
                args["size"] = [args["size"]]

            # process for each size
            for process_size in args["size"]:

                model_input = to_tensor(input_image)

                # resize image and depth if needed
                if process_size is not None:
                    if type(process_size) not in [list, tuple]:
                        process_size = (process_size, process_size)

                    # resize op
                    # Preserve tensor interpolation used to evaluate these weights.
                    # torchvision 0.17 changed the default antialias behavior.
                    resize = transforms.Resize(process_size, antialias=False)

                    model_input = resize(model_input)

                if self.use_depth:
                    assert input_depth is not None

                    depth = input_depth.astype(np.float32)

                    invalid_mask = np.isinf(depth) | np.isnan(depth) | (depth > 1e4) | (depth<0)
                    depth[invalid_mask] = depth[~invalid_mask].mean()

                    if self.norm_depth:
                        depth /= np.max(depth)

                    if self.normalize_depth_by_plane:
                        assert type(normalize_by_depth_plane_points) in [list,tuple], "ERROR: When normalizing depth by plane you need to pass P1 and P2 points as 'normalize_by_depth_plane_points' to call"
                        assert len(normalize_by_depth_plane_points) >= 2, "ERROR: When normalizing depth by plane you need to pass P1 and P2 points as 'normalize_by_depth_plane_points' to call"

                        depth = self.perform_normalize_depth_by_plane(depth, P1=normalize_by_depth_plane_points[0], P2=normalize_by_depth_plane_points[1])

                    depth = resize(to_tensor(depth))

                    model_input = torch.cat((model_input, depth))

                model_input = model_input.unsqueeze(0).to(self.device)

                # run main model
                output_batch_ = self.model(model_input)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                # run center detection model
                center_output = self.center_model(output_batch_)

                center_pred = center_output["center_pred"]
                pred_angle = center_output['pred_attributes']['angle']

                # make sure data is copied
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                # sort predictions and angles by score
                predictions_tensor = center_pred[0][center_pred[0,:,0] == 1]
                if predictions_tensor.numel() == 0:
                    continue
                predictions = predictions_tensor[:,1:].cpu().numpy()

                CENTER_INDEX = 3

                # use center score by default (should be 4th)
                sort_by_index = CENTER_INDEX

                # use confidence score if requested (should be last)
                if self.use_only_confidence_prediction_score:
                    sort_by_index = -1

                idx = np.argsort(predictions[:, sort_by_index])
                idx = idx[::-1]

                predictions = predictions[idx, :]
                angles_rot = pred_angle[0].cpu().numpy()[idx, :]

                # remove ones with low score
                idx = predictions[:, CENTER_INDEX] > threshold
                predictions = predictions[idx, :]
                angles_rot = angles_rot[idx, :]

                if args.get('use_segmnetation'):
                    output = center_output['output']
                    seg = torch.sigmoid(output[:, -1])

                    # remove ones from outside segmentation
                    seg_scores = np.array([seg[0,int(j),int(i)].item() for i,j in predictions[:,[0,1]]])

                    idx = seg_scores > seg_threshold

                    predictions = predictions[idx, :]
                    angles_rot = angles_rot[idx, :]

                    import torch.nn.functional as F
                    resized_seg = F.interpolate(seg.unsqueeze(0), size=input_image.shape[0:2])[0,0]

                    if seg_output is None:
                        seg_output = resized_seg


                # transform predictions to original image size
                scale_y = input_image.shape[0] / output_batch_.shape[2]
                scale_x = input_image.shape[1] / output_batch_.shape[3]

                predictions[:,0] = (predictions[:,0] * scale_x).astype(np.int32)
                predictions[:,1] = (predictions[:,1] * scale_y).astype(np.int32)

                length = 50

                # transform angles to vector points
                if self.enable_6dof_from_eulers:
                    assert input_depth is not None

                    plot_axis = []
                    for p,eulers in zip(predictions, angles_rot):
                        # we need to convert to sin/cos and back since center_estimator_with_orientation has sin/cos switched (should be eqvivalent to 90 - eulers in degrees)
                        eulers = np.arctan2(np.cos(np.deg2rad(eulers)),np.sin(np.deg2rad(eulers)))

                        i, j = p[[0,1]]

                        pt = self.C_K_inv @ np.array([i, j, 1])
                        pt = (pt / pt[-1]) * (input_depth[int(j),int(i)].item() + self.depth_offset)

                        if self.enable_6dof == False:
                            # assume we regressed Z-axis euler in world coord with zero Y-axis and X-axis set to -pi/2
                            eulers = np.array([-np.pi/2, 0, eulers[0]])

                        from scipy.spatial.transform import Rotation
                        if self.enable_orinetation_in_world_coord:
                            X_W_P = np.identity(4)
                            X_W_P[:3,:3] = Rotation.from_euler('xyz',eulers).as_matrix()

                            frame_pose_in_camera = np.linalg.inv(self.X_W_C) @ X_W_P

                            eulers = Rotation.from_matrix(frame_pose_in_camera[:3, :3]).as_euler("xyz")

                        assert len(eulers) == 3


                        R = Rotation.from_euler('xyz',eulers).as_matrix()

                        pt_3d = np.float32([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1], [0, 0, 0]]).reshape(-1, 3)
                        axisPoints, _ = cv2.projectPoints(pt_3d, R, pt, self.C_K, (0, 0, 0, 0))

                        plot_axis.append([axisPoints[axis].ravel() for axis in [0,1,2]])

                    angles_xy = np.stack(plot_axis,axis=1) if len(plot_axis) > 0 else []
                else:
                    angles_rot = (angles_rot - 180) * np.pi / 180

                    angles_xy = [np.stack((predictions[:,0] - length * np.cos(angles_rot[:,rot_axis]),
                                        predictions[:,1] - length * np.sin(angles_rot[:,rot_axis])), axis=1) for rot_axis in range(angles_rot.shape[1])]

                angles_xy = np.stack(angles_xy,axis=1) if len(plot_axis) > 0 else []

                # append detected points and vectors
                scores.append(predictions[:,sort_by_index])
                points.append(predictions[:,[0,1]])
                vectors.append(angles_xy)
                angles.append(angles_rot)


            if len(scores) == 0:
                return (
                    np.empty((0, 2), dtype=np.int32),
                    np.empty((0, 3, 2), dtype=np.int32),
                    [],
                    np.empty((0,), dtype=np.float32),
                    seg_output,
                )

            scores = np.concatenate(scores,axis=0)
            points = np.concatenate(points,axis=0).astype(np.int32)
            vectors = np.concatenate(vectors,axis=0).astype(np.int32)
            angles = np.concatenate(angles,axis=0).astype(np.float32)

            # perform non-maximum suppression on points (convert points to bounding box with R=40px)
            R = args.get("nms_radius", 40)
            nms_points = np.stack((points[:,0] - R/2, points[:,1] - R/2, R*np.ones(len(points)), R*np.ones(len(points))), axis=1)

            ids = cv2.dnn.NMSBoxes(nms_points.tolist(), scores.tolist(),
                                    score_threshold=threshold, nms_threshold=0.1)
            ids = np.array(ids).reshape(-1) if len(ids) > 0 else np.array([], dtype=np.int64)
            if len(ids) == 0:
                return (
                    np.empty((0, 2), dtype=np.int32),
                    np.empty((0, 3, 2), dtype=np.int32),
                    [],
                    np.empty((0,), dtype=np.float32),
                    seg_output,
                )
            points = points[ids]
            vectors = vectors[ids]
            angles = angles[ids]
            scores = scores[ids]

            idx_sort = np.argsort(scores)[::-1]

            points = points[idx_sort]
            vectors = vectors[idx_sort]
            angles = angles[idx_sort]
            scores = scores[idx_sort]

            if self.top_k_predictions:
                top_k = min(len(points),self.top_k_predictions)

                points = points[:top_k]
                vectors = vectors[:top_k]
                angles = angles[:top_k]
                scores = scores[:top_k]


        pose_in_3d = []
        if input_depth is not None and self.C_K_inv is not None and self.X_W_C is not None:
            # convert point to 3D world coordinates
            pose_in_3d = [self._project_pose_to_3d(p, r, input_depth[int(p[1]),int(p[0])]) for p, r in zip(points, angles)]

        return points, vectors, pose_in_3d, scores, seg_output

    def _project_pose_to_3d(self, point, rot, depth):
        # convert point to 3d
        x = np.array([point[0], point[1], 1])

        p = (self.C_K_inv @ x)
        p = (p / p[-1]) * depth

        # convert vectors to rotation
        from scipy.spatial.transform import Rotation

        if len(rot) > 1:
            if self.enable_6dof_from_eulers:
                rot = np.arctan2(np.cos(np.deg2rad(rot)),np.sin(np.deg2rad(rot)))
                R = Rotation.from_euler('xyz',rot).as_matrix()
            else:
                R = Rotation.from_rotvec(np.array(rot)).as_matrix()
        else:
            if self.enable_orinetation_in_world_coord and self.enable_6dof_from_eulers and self.enable_6dof == False:
                rot = np.arctan2(np.cos(np.deg2rad(rot)),np.sin(np.deg2rad(rot)))

                # assume we regressed Z-axis euler in world coord with zero Y-axis and X-axis set to -pi/2
                rot = np.array([-np.pi/2, 0, rot[0]])

                X_W_P = np.identity(4)
                X_W_P[:3,:3] = Rotation.from_euler('xyz',rot).as_matrix()

                frame_pose_in_camera = np.linalg.inv(self.X_W_C) @ X_W_P

                R = frame_pose_in_camera[:3, :3]
            else:

                R = Rotation.from_rotvec(np.array([0, 0, rot ])).as_matrix()

        X = np.identity(4)
        X[:3,3] = p
        X[:3,:3] = R

        p_in_3d = self.X_W_C @ X

        return p_in_3d


    def detect_and_plot_grasp_points(self, input_image):

        points, vectors, pose_in_3d, scores, seg = self.detect_grasp_points(input_image)

        draw_image = input_image
        for prediction, vector, score in zip(points, vectors, scores):
            x1, y1 = int(prediction[0]), int(prediction[1])

            if x1 < 0 or draw_image.shape[1] < x1:
                continue

            if y1 < 0 or draw_image.shape[0] < y1:
                continue

            for (x2,y2),c in zip(vector,AXIS_COLOR):
                draw_image = cv2.line(draw_image, (x1, y1), (x2, y2), color=c, thickness=draw_image.shape[1] // 480)
                draw_image = cv2.circle(draw_image, (x1, y1), radius=draw_image.shape[1] // 240, color=(0,0,255), thickness=-1)

        if seg is not None:
            seg = (seg*255).cpu().numpy().astype(np.uint8)
            heatmap = cv2.applyColorMap(seg, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

            draw_image = cv2.addWeighted(draw_image, 0.8, heatmap, 0.2, 0)

        return cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB)

    def detect_return_in_3d_and_plot_grasp_points(self, input_image, depth_image, cam_K=None, cam_K_inv=None, cam_in_3d=None, normalize_by_depth_plane_points=None):

        org_C_K_and_XWC = self.C_K, self.C_K_inv, self.X_W_C

        if cam_K is not None:
            self.C_K = cam_K
        if cam_K_inv is not None:
            self.C_K_inv = cam_K_inv
        if cam_in_3d is not None:
            self.X_W_C = cam_in_3d

        assert self.C_K is not None and self.C_K_inv is not None, "ERROR: Camera calibration is missing"
        assert self.X_W_C is not None, "ERROR: Camera position in 3D world is missing"

        points, vectors, pose_in_3d, scores, seg = self.detect_grasp_points(input_image, depth_image, normalize_by_depth_plane_points)

        # restore original C_K in X_W_C
        self.C_K, self.C_K_inv, self.X_W_C = org_C_K_and_XWC

        draw_image = input_image
        rank = 0
        for prediction, vector, score in zip(points, vectors, scores):
            x1, y1 = int(prediction[0]), int(prediction[1])

            if x1 < 0 or draw_image.shape[1] < x1:
                continue

            if y1 < 0 or draw_image.shape[0] < y1:
                continue

            for (x2,y2),c in zip(vector,AXIS_COLOR):
                draw_image = cv2.line(draw_image, (x1, y1), (x2, y2), color=c, thickness=5)
                draw_image = cv2.circle(draw_image, (x1, y1), radius=10, color=(0,0,255), thickness=-1)
                draw_image = cv2.putText(draw_image, f"{rank}: {score*100:.1f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color = (0, 0, 0) , thickness=2)

            rank += 1

        if seg is not None:
            seg = (seg*255).cpu().numpy().astype(np.uint8)
            heatmap = cv2.applyColorMap(seg, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

            draw_image = cv2.addWeighted(draw_image, 0.8, heatmap, 0.2, 0)

        return cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB), pose_in_3d, scores




    def perform_normalize_depth_by_plane(self, D, P1, P2):

        # We convert all pixel values to 3D then use as depth the distances in 3D world points to the plane
        # running between robot bases streaced in Z direction.
        # Actullay this values it he same as X coordinate in 3D world, but we compute it relative to robot bases
        # to be more independent of the coordinate system
        out_type = D.dtype

        # eliminate inf/nan values just in case
        invalid_mask = np.isinf(D) | np.isnan(D) | (D > 1e4) | (D<0)
        D[invalid_mask]=D[~invalid_mask].mean()

        X, Y = np.meshgrid(np.arange(D.shape[1]), np.arange(D.shape[0]))  # Fixed meshgrid order for width/height

        # Flatten and prepare homogeneous 2D points
        D_flat = D.flatten()
        X_flat = X.flatten()
        Y_flat = Y.flatten()

        ones = np.ones_like(X_flat)
        points_2d_homogeneous = np.vstack((X_flat, Y_flat, ones))

        # Apply the inverse of the intrinsic matrix to get normalized camera coordinates
        points_3d_normalized = self.C_K_inv @ points_2d_homogeneous  # (3, N)

        # Scale by the depth to get the 3D points in camera space
        points_3d_camera = points_3d_normalized * D_flat  # (3, N)

        # Convert points to homogeneous 3D coordinates for transformation (4xN)
        points_3d_camera_homogeneous = np.vstack((points_3d_camera, ones))

        # Apply the transformation from camera to world coordinates using the extrinsic matrix
        points_3d_world_homogeneous = self.X_W_C @ points_3d_camera_homogeneous  # (4, N)

        # Extract the 3D world coordinates from homogeneous coordinates
        points_3d_world = points_3d_world_homogeneous[:3, :]  # (3, N)

        # Reshape back into the original image shape if needed
        points_3d_world = points_3d_world.T.reshape(D.shape[0], D.shape[1], 3)

        # now calculate distance to plane defined by robot bases and Z axis:

        # Points defining the plane are P1 and P2

        # Step 1: Compute the normal vector of the plane using cross product
        v1 = P2 - P1  # Vector from P1 to P2
        v2 = np.array([0, 0, 1]) - P1  # Vector from z-axis direction to P1

        normal = np.cross(v1, v2)  # Normal vector (A, B, C)

        # Step 2: Compute D in the plane equation
        # Plane equation: A*x + B*y + C*z + D = 0
        A, B, C = normal
        D = -(A * P1[0] + B * P1[1] + C * P1[2])  # Using point P1 to calculate D

        # Step 3: Compute the distance from each point to the plane
        # points_3d_world is the array of shape (H, W, 3) containing 3D points in world coordinates
        X_world = points_3d_world[..., 0]  # X coordinates of all points
        Y_world = points_3d_world[..., 1]  # Y coordinates of all points
        Z_world = points_3d_world[..., 2]  # Z coordinates of all points

        # Distance formula: |A*x + B*y + C*z + D| / sqrt(A^2 + B^2 + C^2)
        numerator = A * X_world + B * Y_world + C * Z_world + D
        denominator = np.sqrt(A**2 + B**2 + C**2)
        distances = numerator / denominator  # Distance of each point to the plane

        # distances will be a (H, W) array representing the distance of each point to the plane
        return distances.astype(out_type)

class FolderServerProcessing:
    def __init__(self, inference_models, sample_folder, load_all_models=False, preemptive_model_load=False,
                 img_subfolder="observation_start", img_name="image_left.png",
                 depth_subfolder="observation_start", depth_name="depth_map.tiff",
                 out_subfolder="grasp_predicted", out_name="frontal_image_grasp.jpg",
                 calibration_subfolder="observation_start", camera_intrinsics_name="camera_intrinsics.json", camera_in_world_name="camera_pose_in_world.json",
                 enable_crop_by_robot_position=False, robot_position_subfolder="observation_start",
                 robot_left_base_name='arm_left_pose_in_world.json', robot_left_tcp_name='arm_left_tcp_pose_in_world.json',
                 robot_right_base_name='arm_right_pose_in_world.json', robot_right_tcp_name='arm_right_tcp_pose_in_world.json',
                 requested_model_filename='requested_model.json',
                 ):

        self.inference_models = inference_models
        self.sample_folder = sample_folder
        self.img_subfolder = img_subfolder
        self.img_name = img_name
        self.depth_subfolder = depth_subfolder
        self.depth_name = depth_name

        self.calibration_subfolder = calibration_subfolder
        self.camera_intrinsics_name = camera_intrinsics_name
        self.camera_in_world_name = camera_in_world_name

        self.enable_crop_by_robot_position = enable_crop_by_robot_position
        self.robot_position_subfolder = robot_position_subfolder
        self.robot_left_base_name = robot_left_base_name
        self.robot_left_tcp_name = robot_left_tcp_name
        self.robot_right_base_name = robot_right_base_name
        self.robot_right_tcp_name = robot_right_tcp_name

        self.requested_model_filename = requested_model_filename

        self.crop_by_robot_positions_in_center_height = False

        self.out_subfolder = out_subfolder
        self.out_name = out_name

        os.makedirs(self.sample_folder,exist_ok=True)

        self.load_all_models = load_all_models
        self.preemptive_model_load = preemptive_model_load

        # prepare list of keys and indexes
        self.model_names = list(self.inference_models.keys())

    def run(self):
        import glob, time

        already_processed = ['.','..']

        try:
            while True:

                folder_listing = glob.iglob(os.path.join(self.sample_folder, "*"))
                folder_listing = [f for f in folder_listing if os.path.isdir(f)]

                for_processing = set(folder_listing) - set(already_processed)

                for sample in sorted(for_processing):
                    sample_out_folder = os.path.join(sample,self.out_subfolder)
                    out_filename = os.path.join(sample_out_folder, f"grasp_pose.json")

                    if os.path.exists(out_filename):
                        continue

                    self.process_sample(sample)

                time.sleep(1)
        except KeyboardInterrupt:
            pass

    def process_sample(self, sample, requested_model_override=None, raise_errors=False, print_elapsed=True):
        import json
        import time
        from functools import partial
        from scipy.spatial.transform import Rotation

        sample_out_folder = os.path.join(sample, self.out_subfolder)
        out_filename = os.path.join(sample_out_folder, f"grasp_pose.json")
        requested_model = None
        infer = None
        time_start = time.time()

        print("Processing", sample)

        os.makedirs(sample_out_folder, exist_ok=True)

        frame = cv2.imread(os.path.join(sample, self.img_subfolder, self.img_name))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.array(frame)

        depth = None
        if self.depth_subfolder is not None:
            depth_filename = os.path.join(sample, self.depth_subfolder, self.depth_name)

            if os.path.exists(depth_filename):
                from PIL import Image
                depth = np.array(Image.open(depth_filename))

        try:
            # find out which model was requested for processing
            requested_model = requested_model_override
            if requested_model is None:
                requested_model_file = os.path.join(sample, self.img_subfolder, self.requested_model_filename)

                if os.path.exists(requested_model_file):
                    with open(requested_model_file, 'r') as file:
                        data = json.load(file)

                    assert 'requested_model' in data
                    requested_model = data['requested_model']
                else:
                    requested_model = ''

            assert requested_model in self.inference_models, f"ERROR: No '{requested_model}' model found in available list of loaded models"

            infer = self.inference_models[requested_model]

            # load model if not present
            if not infer.is_model_loaded():
                # unload all other models unless all models should be loaded
                if not self.load_all_models:
                    for k,i in self.inference_models.items():
                        i.unload_model()

                infer.load_model()

            call_fn = infer

            # attach camera calibration values if present in observation folder
            if self.calibration_subfolder is not None:
                camera_calibration = os.path.join(sample, self.calibration_subfolder, self.camera_intrinsics_name)
                camera_in_world = os.path.join(sample, self.calibration_subfolder, self.camera_in_world_name)

                if os.path.exists(camera_calibration):
                    cam_K, cam_K_inv = load_camera_intrinsics(camera_calibration)

                    # attach camera K matrix to main call
                    call_fn = partial(call_fn, cam_K=cam_K, cam_K_inv=cam_K_inv)

                if os.path.exists(camera_in_world):
                    cam_in_3d = load_camera_world(camera_in_world)

                    # attach camera extrinsic matrix to main call
                    call_fn = partial(call_fn, cam_in_3d=cam_in_3d)

                draw_image = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
                draw_image = self.draw_robot_axis(sample, draw_image, cam_K, cam_in_3d)

                cv2.imwrite(os.path.join(sample, self.out_subfolder, "input_space.jpg"), draw_image)

            if infer.normalize_depth_by_plane:
                from utils.utils_3d import read_competition_pose

                arm_left_base_path = os.path.join(sample,self.robot_position_subfolder,self.robot_left_base_name)
                arm_right_base_path = os.path.join(sample,self.robot_position_subfolder,self.robot_right_base_name)

                assert os.path.exists(arm_left_base_path), f"Missing {arm_left_base_path}"
                assert os.path.exists(arm_right_base_path), f"Missing {arm_right_base_path}"

                arm_left_base, _ = read_competition_pose(arm_left_base_path)
                arm_right_base, _ = read_competition_pose(arm_right_base_path)

                P1 = arm_right_base # First point in world coordinates
                P2 = arm_left_base # Second point in world coordinates

                call_fn = partial(call_fn, normalize_by_depth_plane_points=(P1,P2))

            if infer.get_enabled_crop_by_robot_position():
                assert 'cam_K' in locals()
                assert 'cam_in_3d' in locals()

                frame, depth, cam_K = self.crop_data_by_robot_position(frame, depth, sample, cam_K, cam_in_3d, infer.get_enabled_crop_by_robot_position())
                cam_K_inv = np.linalg.inv(cam_K)

                # override cam_k with new values
                call_fn = partial(call_fn, cam_K=cam_K, cam_K_inv=cam_K_inv)

            frame,pose_in_3d_list,scores = call_fn(frame,depth)

            # filter any unreachable points and add pregrasp depth
            pose_in_3d_list,scores = self.filter_unreachable_and_add_pregrasp_depth(pose_in_3d_list, scores, sample, **infer.get_postprocessing_kwargs())

            pose_in_3d_json = []

            for pose_in_3d,score in zip(pose_in_3d_list,scores):
                euler = Rotation.from_matrix(pose_in_3d[:3, :3]).as_euler("xyz")

                # check if euler angles are valid (they can be NaN due to Gimbal lock) and skip if not
                if np.any(np.isnan(euler)):
                    continue

                pose_in_3d_json.append(dict(position_in_meters=dict(x=pose_in_3d[0,3],
                                                                    y=pose_in_3d[1,3],
                                                                    z=pose_in_3d[2,3]),
                                            rotation_euler_xyz_in_radians=dict(roll=euler[0],
                                                                            pitch=euler[1],
                                                                            yaw=euler[2]),
                                            score=float(score)))
            with open(out_filename, 'w') as f:
                json.dump(pose_in_3d_json, f, indent=4)

            elapsed_time = time.time() - time_start
            if print_elapsed:
                print(f"Elapsed time: {elapsed_time:.6f} seconds")

        except Exception as e:
            import traceback
            error_message = traceback.format_exc()

            print("exception:", error_message)
            cv2.putText(frame, f"Failed to detect image", (500, 500), cv2.FONT_HERSHEY_TRIPLEX, 5, [0, 0, 255], 10)
            if raise_errors:
                raise RuntimeError(error_message) from e

            elapsed_time = None

        if self.out_subfolder != None:
            cv2.imwrite(os.path.join(sample, self.out_subfolder, self.out_name), frame)
        else:
            import pylab as plt
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.show(block=True)

        # pre-load next one if requested
        if self.preemptive_model_load and requested_model in self.model_names:
            # unload existing model
            if infer is not None:
                infer.unload_model()

            # find which model is next
            next_model = self.model_names[(self.model_names.index(requested_model)+1) % len(self.model_names)]

            # load next model
            print(f"Pre-emptivly loading next model '{next_model}'")
            self.inference_models[next_model].load_model()

        return dict(sample=sample, requested_model=requested_model, elapsed_time=elapsed_time, out_filename=out_filename)

    def filter_unreachable_and_add_pregrasp_depth(self, pose_in_3d_list, scores, sample, robot_distance_thr=None, pregrasp_aproach_depth=None):

        if robot_distance_thr is not None:
            MIN_DISTANCE_FROM_ROBOT, MAX_DISTANCE_FROM_ROBOT = min(robot_distance_thr), max(robot_distance_thr)
            from utils.utils_3d import read_competition_pose

            arm_right_base_path = os.path.join(sample,self.robot_position_subfolder,self.robot_right_base_name)
            assert os.path.exists(arm_right_base_path), f"Missing {arm_right_base_path}"

            # check distance to RIGHT robot for reachability
            arm_right_base, _ = read_competition_pose(arm_right_base_path)

            # calc distances to RIGHT robot
            distance_pose_to_base = [np.linalg.norm(p[:3,3] - arm_right_base) for p in pose_in_3d_list]

            # retain only ones with distance close enough for grasping
            filtered = [(p, s) for p,d,s in zip(pose_in_3d_list,distance_pose_to_base,scores)
                            if d > MIN_DISTANCE_FROM_ROBOT and d < MAX_DISTANCE_FROM_ROBOT]

            reachable_poses, reachable_scores = zip(*filtered) if len(filtered) > 0 else ([],[])
        else:
            reachable_poses, reachable_scores = pose_in_3d_list, scores

        if pregrasp_aproach_depth is not None:
            # add extra approach depth to Z axis
            translation_matrix = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, pregrasp_aproach_depth],
                    [0, 0, 0, 1]
                ])

            reachable_poses = [np.dot(pose, translation_matrix) for pose in reachable_poses]

        return reachable_poses, reachable_scores


    def crop_data_by_robot_position(self, image, depth, sample, cam_K, cam_in_3d, enable_crop_by_robot_position):
        from utils.utils_3d import read_competition_pose, project_grasp_pose_to_image_plane

        arm_left_base_path = os.path.join(sample,self.robot_position_subfolder,self.robot_left_base_name)
        arm_left_tcp_path = os.path.join(sample,self.robot_position_subfolder,self.robot_left_tcp_name)
        arm_right_base_path = os.path.join(sample,self.robot_position_subfolder,self.robot_right_base_name)
        arm_right_tcp_path = os.path.join(sample,self.robot_position_subfolder,self.robot_right_tcp_name)

        assert os.path.exists(arm_left_base_path), f"Missing {arm_left_base_path}"
        assert os.path.exists(arm_left_tcp_path), f"Missing {arm_left_tcp_path}"
        assert os.path.exists(arm_right_base_path), f"Missing {arm_right_base_path}"
        assert os.path.exists(arm_right_tcp_path), f"Missing {arm_right_tcp_path}"

        arm_left_base, arm_left_base_pose = read_competition_pose(arm_left_base_path)
        arm_left_tcp, arm_left_tcp_pose = read_competition_pose(arm_left_tcp_path)
        arm_right_base, arm_right_base_pose = read_competition_pose(arm_right_base_path)
        arm_right_tcp, arm_right_tcp_pose = read_competition_pose(arm_right_tcp_path)

        # project points to image

        pt_arm_left_base = project_grasp_pose_to_image_plane(arm_left_base, arm_left_base_pose, cam_K, cam_in_3d)[:2]
        pt_arm_left_tcp = project_grasp_pose_to_image_plane(arm_left_tcp+np.array([0,0,0.05]), arm_left_tcp_pose, cam_K, cam_in_3d)[:2]

        pt_arm_right_base = project_grasp_pose_to_image_plane(arm_right_base, arm_right_base_pose, cam_K, cam_in_3d)[:2]
        pt_arm_right_tcp = project_grasp_pose_to_image_plane(arm_right_tcp, arm_right_tcp_pose, cam_K, cam_in_3d)[:2]

        # get bounds of an image by:
        # left = left base
        # right = right base
        # top = left TCP
        # bottom = lowest point of (left_base, right_base)
        l,r,t,b = pt_arm_left_base[0],pt_arm_right_base[0],pt_arm_left_tcp[1],max(pt_arm_left_base[1],pt_arm_right_base[1])


        if self.crop_by_robot_positions_in_center_height:
            center_between_robot = (arm_right_base + arm_left_base) / 2
            center_between_robot[2] = self.crop_by_robot_positions_in_center_height

            pt_center_between_robot = project_grasp_pose_to_image_plane(center_between_robot, [0,0,0], cam_K, cam_in_3d)[:2]

            # if height specificaly set, use that instead of TCP position
            t = pt_center_between_robot[1]

        if type(enable_crop_by_robot_position) in [tuple, list]:
            crop_center = (l+r)/2, (b+t)/2
            w,h = enable_crop_by_robot_position
            # if requested width or height are smaller than actual distance then use actual distance
            if w < r-l:
                w = int(r-l)
            if h < b-t:
                h = int(b-t)
            x,y = int(crop_center[1] - h/2), int(crop_center[0] - w/2)
        else:
            x,y,w,h = t,l, r-l, b-t

        # ensure width and height crop are factors of 32
        if h % 32 != 0:
            cy = int(y + h//2)
            h = np.ceil(h/32)*32
            y = int(cy - h/2)

        if w % 32 != 0:
            cx = int(x + w//2)
            w = np.ceil(w/32)*32
            x = int(cx - w/2)

        x,y,w,h = map(int, [x,y,w,h])

        from torchvision.transforms import functional as F
        data = dict(image=F.to_tensor(image),depth=F.to_tensor(depth), K=cam_K)
        keys_img = ['image']

        if depth is not None:
            keys_img.append('depth')

        from utils.transforms import FixedCrop
        crop_op = FixedCrop(keys=keys_img, keys_camK=('K') if cam_K is not None else None, loc=(x,y), size=(h, w), pad_if_needed=False)
        crop_op(data)

        return (255*data['image']).permute([1, 2, 0]).contiguous().cpu().numpy().astype(np.uint8), \
                data['depth'].cpu().numpy().squeeze() if depth is not None else None, \
                data['K']


    def draw_robot_axis(self, sample, image, cam_K, cam_in_3d):
        from utils.utils_3d import read_competition_pose, project_grasp_pose_to_image_plane

        arm_left_base_path = os.path.join(sample,self.robot_position_subfolder,self.robot_left_base_name)
        arm_left_tcp_path = os.path.join(sample,self.robot_position_subfolder,self.robot_left_tcp_name)
        arm_right_base_path = os.path.join(sample,self.robot_position_subfolder,self.robot_right_base_name)
        arm_right_tcp_path = os.path.join(sample,self.robot_position_subfolder,self.robot_right_tcp_name)

        assert os.path.exists(arm_left_base_path), f"Missing {arm_left_base_path}"
        assert os.path.exists(arm_left_tcp_path), f"Missing {arm_left_tcp_path}"
        assert os.path.exists(arm_right_base_path), f"Missing {arm_right_base_path}"
        assert os.path.exists(arm_right_tcp_path), f"Missing {arm_right_tcp_path}"

        arm_left_base, arm_left_base_pose = read_competition_pose(arm_left_base_path)
        arm_left_tcp, arm_left_tcp_pose = read_competition_pose(arm_left_tcp_path)
        arm_right_base, arm_right_base_pose = read_competition_pose(arm_right_base_path)
        arm_right_tcp, arm_right_tcp_pose = read_competition_pose(arm_right_tcp_path)

        # project points to image

        pt_arm_left_base = project_grasp_pose_to_image_plane(arm_left_base, arm_left_base_pose, cam_K, cam_in_3d)[:2]
        pt_arm_left_tcp = project_grasp_pose_to_image_plane(arm_left_tcp+np.array([0,0,0.05]), arm_left_tcp_pose, cam_K, cam_in_3d)[:2]

        pt_arm_right_base = project_grasp_pose_to_image_plane(arm_right_base, arm_right_base_pose, cam_K, cam_in_3d)[:2]
        pt_arm_right_tcp = project_grasp_pose_to_image_plane(arm_right_tcp, arm_right_tcp_pose, cam_K, cam_in_3d)[:2]

        cam_K_inv = np.linalg.inv(cam_K)

        image = draw_pose(image, arm_left_base, arm_left_base_pose, cam_K, cam_in_3d)
        image = draw_pose(image, arm_left_tcp, arm_left_tcp_pose, cam_K, cam_in_3d)
        image = draw_pose(image, arm_right_base, arm_right_base_pose, cam_K, cam_in_3d)
        image = draw_pose(image, arm_right_tcp, arm_right_tcp_pose, cam_K, cam_in_3d)

        image = cv2.circle(image, pt_arm_left_base.astype(np.int32), radius=8, color=(0,0,255), thickness=-1)
        image = cv2.circle(image, pt_arm_left_tcp.astype(np.int32), radius=8, color=(0,0,255), thickness=-1)
        image = cv2.circle(image, pt_arm_right_base.astype(np.int32), radius=8, color=(0,0,255), thickness=-1)
        #image = cv2.circle(image, pt_arm_right_tcp.astype(np.int32), radius=8, color=(0,0,255), thickness=-1)

        return image


def draw_pose(img, p, euler , C_K, camera_pose):
    from scipy.spatial.transform import Rotation

    X_W_C = get_homogeneous_matrix(camera_pose["pos"], camera_pose["rot"]) if type(camera_pose) == dict else camera_pose
    X_W_P = get_homogeneous_matrix(p, euler)
    X_C_P = np.linalg.inv(X_W_C) @ X_W_P
    frame_pose_in_camera = X_C_P
    pt = frame_pose_in_camera[:3,3]

    R = Rotation.from_matrix(frame_pose_in_camera[:3,:3]).as_rotvec()

    return cv2.drawFrameAxes(img, C_K, None, R, pt, 0.1)


def build_inference(cfg, model, camera_calibration, camera_in_world, load_model=True):
    import importlib

    module = importlib.import_module(cfg)

    args = module.get_args()
    args['checkpoint_path'] = model
    args['center_checkpoint_path'] = model

    infer = Inference(args, camera_calibration=camera_calibration, camera_in_world=camera_in_world)
    if load_model:
        infer.load_model()

    return infer

def main(cmd_args):

    assert os.path.exists(cmd_args.model_definitions), f"model_definitions: File not found {cmd_args.model_definitions}"

    with open(cmd_args.model_definitions, 'r') as file:
        model_definitions = json.load(file)

    from collections import OrderedDict

    inference_models = OrderedDict()
    for name, model_def in model_definitions.items():
        cfg = "config." + model_def['cfg']
        model = model_def['model']
        inference_models[name] = build_inference(cfg, model, cmd_args.camera_calibration, cmd_args.camera_in_world,
                                                 load_model=cmd_args.load_all_models)

    assert cmd_args.serve_folder is not None

    p = FolderServerProcessing(inference_models, cmd_args.serve_folder,
                               load_all_models=cmd_args.load_all_models,
                               preemptive_model_load=cmd_args.preemptive_model_load)
    p.run()


def parseArgs():
    import argparse
    parser = argparse.ArgumentParser(description='Cloth grasp point detection')
    parser.add_argument(
        '--model-definitions',
        dest='model_definitions',
        help='model model file',
        default="/opt/model_definitions.json",
        type=str
    )
    parser.add_argument(
        '--serve-folder',
        dest='serve_folder',
        help='folder to samples with images for processing (default: None)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--camera-calibration',
        dest='camera_calibration',
        help='path to json file for camera calibration information (default: None)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--camera-in-world',
        dest='camera_in_world',
        help='path to json file for camera-in-world information (default: None)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--out-folder',
        dest='out_folder',
        help='folder to store output (default: None)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--load-all-models',
        dest='load_all_models',
        help='Load all models in advance (default: False)',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--preemptive-model-load',
        dest='preemptive_model_load',
        help='Load next models preemptively when finished with last one (default: False)',
        action='store_true',
        default=False
    )

    return parser.parse_args()

if __name__ == "__main__":
    cmd_args = parseArgs()

    main(cmd_args)
