import glob
import os, cv2

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import json, sys, os

import torch
from torch.utils.data import Dataset

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import transforms as my_transforms

from utils.utils_3d import (
    project_grasp_pose_to_direction_in_image_plane,
    project_grasp_pose_to_image_plane,
	project_from_image_to_world,
	get_homogeneous_matrix,
	get_trans_and_rot_from_matrix,
	get_trans_and_euler_from_matrix,
	grasp_pose_to_eulers_in_camera_frame,
	grasp_pose_to_eulers_in_world_frame,
	read_competition_pose
)

from utils.utils_depth import (
	get_surface_normal_by_depth_robust,
	get_hha_encoding
)


class ClothCompetitionICRA2024Dataset(Dataset):
	DATASET_NAME = 'cloth_competition_icra2024'

	IGNORE_LABEL = "ignore-region"

	IGNORE_FLAG = 1
	IGNORE_TRUNCATED_FLAG = 2
	IGNORE_OVERLAP_BORDER_FLAG = 4
	IGNORE_DIFFICULT_FLAG = 8

	def __init__(self, root_dir='./', subfolders=None, fixed_bbox_size=15, resize_factor=None, use_depth=False, integrate_depth_with_image=True, segment_cloth=False, segment_cloth_optional=False, segement_bg_by_depth_thr=False, MAX_NUM_CENTERS=1024,
			  	transform=None, valid_sample_names=None, num_cpu_threads=1, use_mean_for_depth_nan=False, use_only_depth=False, normalize_depth=True, normalize_depth_arm_bases_plane=False,
				use_6dof_pose=False, orientation_in_world_frame=False, orientation_as_euler_angles=False, remove_pregrasp_depth=False, crop_by_robot_positions=False, crop_by_robot_positions_in_center_height=None,
				compatibility_with_infer_docker=False,
				allow_grasp_predicted_as_backup=False,
				check_consistency=True, transform_only_valid_centers=False, transform_per_sample_rng=False,
				depth_encoding_camera_height=1.5, depth_encoding_gaussian_blur_k=5, save_original_image=False, save_robot_arm_data=False, **kwargs):
		print('ClothCompetition ICRA2024 Dataset created')

		assert orientation_as_euler_angles, "ClothCompetitionICRA2024Dataset requires orientation_as_euler_angles=True"

		if num_cpu_threads:
			torch.set_num_threads(num_cpu_threads)

		# use_depth options:
		#   False - depth not used (default)
		#   True - depth used with encoding based on normalize_depth flag (backward compatible)
		#   'raw' - raw depth values in meters (no normalization)
		#   'normalized' - depth normalized to [0,1] by dividing by max
		#   'normals' - surface normals encoding (3 channels), requires raw depth in meters
		#   'hha' - HHA encoding (Horizontal disparity, Height above ground, Angle with gravity - 3 channels), requires raw depth in meters
		#   'arm_bases_plane' - distance to plane between robot arm bases
		#
		# Backward compatibility:
		#   - use_depth=True with normalize_depth=True (default) -> 'normalized'
		#   - use_depth=True with normalize_depth=False -> 'raw'
		#   - use_depth=True with normalize_depth_arm_bases_plane=True -> 'arm_bases_plane'

		VALID_DEPTH_ENCODINGS = ['raw', 'normalized', 'normals', 'hha', 'arm_bases_plane']

		if isinstance(use_depth, str):
			# New way: use_depth is the encoding type directly
			assert use_depth in VALID_DEPTH_ENCODINGS, f"Invalid use_depth '{use_depth}', must be False, True, or one of {VALID_DEPTH_ENCODINGS}"
			depth_encoding = use_depth
			use_depth_flag = True
		elif use_depth:
			# Old way: use_depth=True, determine encoding from other flags
			if normalize_depth_arm_bases_plane:
				depth_encoding = 'arm_bases_plane'
			elif normalize_depth:
				depth_encoding = 'normalized'
			else:
				depth_encoding = 'raw'
			use_depth_flag = True
		else:
			# use_depth=False
			depth_encoding = None
			use_depth_flag = False

		self.use_depth = use_depth_flag
		self.depth_encoding = depth_encoding
		self.depth_encoding_camera_height = depth_encoding_camera_height
		self.depth_encoding_gaussian_blur_k = depth_encoding_gaussian_blur_k

		self.integrate_depth_with_image = integrate_depth_with_image
		self.use_mean_for_depth_nan = use_mean_for_depth_nan
		self.use_only_depth = use_only_depth
		# Keep these for backward compatibility, but they are now derived from use_depth
		self.normalize_depth = normalize_depth
		self.normalize_depth_arm_bases_plane = normalize_depth_arm_bases_plane

		self.segment_cloth = segment_cloth
		self.segment_cloth_optional = segment_cloth_optional
		self.segement_bg_by_depth_thr = segement_bg_by_depth_thr

		self.crop_by_robot_positions = crop_by_robot_positions
		self.crop_by_robot_positions_in_center_height = crop_by_robot_positions_in_center_height
		self.compatibility_with_infer_docker = compatibility_with_infer_docker

		self.use_6dof_pose = use_6dof_pose
		self.orientation_in_world_frame = orientation_in_world_frame
		self.orientation_as_euler_angles = orientation_as_euler_angles
		self.remove_pregrasp_depth = remove_pregrasp_depth

		self.allow_grasp_predicted_as_backup = allow_grasp_predicted_as_backup

		self.fixed_bbox_size = fixed_bbox_size
		self.resize_factor = resize_factor

		self.MAX_NUM_CENTERS = MAX_NUM_CENTERS

		self.transform = my_transforms.get_transform(transform) if type(transform) == list else transform
		self.rng = np.random.default_rng(1337)
		self.transform_only_valid_centers = transform_only_valid_centers
		self.transform_per_sample_rng = transform_per_sample_rng

		self.return_image = True
		self.check_consistency = check_consistency
		self.save_original_image = save_original_image
		self.save_robot_arm_data = save_robot_arm_data

		sample_list = []

		if subfolders is None:
			subfolders = ["*/*"]
		if type(subfolders) not in [list, tuple]:
			subfolders = [subfolders]


		for sub in subfolders:
			sample_list += sorted(glob.glob(os.path.join(root_dir,sub,'sample_*')))

		if valid_sample_names is not None:
			def filter_by_name(x):
				return any([v in x for v in valid_sample_names])

			sample_list = list(filter(filter_by_name, sample_list))


		self.sample_list = sample_list

		self.size = len(self.sample_list)
		print(f'ClothCompetition ICRA2024 Dataset of size {len(sample_list)}')

		self.remove_out_of_bounds_centers = True
		self.BORDER_MARGIN_FOR_CENTER = 10

	def __len__(self):
		return self.size

	def __getitem__(self, index):
		sample_fn = self.sample_list[index]

		im_fn = os.path.join(sample_fn, 'observation_start', 'image_left.png')

		image = Image.open(im_fn)
		im_size = image.size
		org_im_size = np.array(image.size)

		if self.resize_factor is not None:
			im_size = int(image.size[0] * self.resize_factor), int(image.size[1] * self.resize_factor)

		# avoid loading full buffer data if image not requested
		if self.return_image:
			if self.resize_factor is not None and self.resize_factor != 1.0:
				image = image.resize(im_size, Image.BILINEAR)
		else:
			image = None

		sample = dict(image=image,
					  im_name=im_fn,
					  org_im_size=org_im_size,
					  im_size=im_size,
					  index=index)

		if self.use_depth or self.segement_bg_by_depth_thr:
			depth_fn = os.path.join(sample_fn, 'observation_start', 'depth_map.tiff')

			depth = Image.open(depth_fn)

			if self.resize_factor is not None and self.resize_factor != 1.0:
				depth = depth.resize(im_size, Image.NEAREST)

			depth = np.array(depth)
			depth_org = np.array(depth).copy()

			invalid_mask = np.isinf(depth) | np.isnan(depth) | (depth > 1e4) | (depth<0)
			depth[invalid_mask]=depth[~invalid_mask].mean() if self.use_mean_for_depth_nan else 1e-6

			if self.segement_bg_by_depth_thr:
				sample['bg_mask_by_depth'] = depth < self.segement_bg_by_depth_thr

			if self.use_depth:
				sample['depth'] = depth

		if self.segment_cloth:
			segmentation_fn = os.path.join(sample_fn, 'observation_start', 'cloth_segmentation.png')

			if self.segment_cloth_optional and not os.path.exists(segmentation_fn):
				segmentation = torch.zeros((im_size[1], im_size[0]), dtype=torch.uint8)
			else:
				assert os.path.exists(segmentation_fn), f"Missing segmentation file {segmentation_fn}"

				segmentation = Image.open(segmentation_fn)

				if self.resize_factor is not None and self.resize_factor != 1.0:
					segmentation = segmentation.resize(im_size, Image.BILINEAR)

			sample['segmentation_mask'] = segmentation


		label = torch.zeros((im_size[1], im_size[0]), dtype=torch.uint8)
		# NOTE: orientation also holds confidence score at the end !!
		orientation = torch.zeros((4 if self.use_6dof_pose else 2, im_size[1], im_size[0]), dtype=torch.float32)
		instances = torch.zeros((im_size[1], im_size[0]), dtype=torch.int16)
		centers = []

		# read single grasp point
		grasp_path = os.path.join(sample_fn,'grasp','grasp_annotation.json')
		grasp_pose_path = os.path.join(sample_fn,'grasp','grasp_pose.json')

		# use 'grasp_predicted/grasp_pose.json' as backup if 'grasp/grasp_pose.json' does not exist and is allowed by user (self.allow_grasp_predicted_as_backup)
		grasp_pose_predicted_path = os.path.join(sample_fn,'grasp_predicted','grasp_pose.json')
		grasp_pose_attempted_path = os.path.join(sample_fn,'observation_result','attempted_grasp_pose_list.json')

		cam_k_path = os.path.join(sample_fn,'observation_start','camera_intrinsics.json')
		cam_in_3d_path = os.path.join(sample_fn,'observation_start','camera_pose_in_world.json')

		cam_K = None
		cam_in_3d = None

		if os.path.exists(cam_k_path):
			with open(cam_k_path, 'r') as f:
				data = json.load(f)

			# Create camera matrix
			cam_K = np.array([[data['focal_lengths_in_pixels']['fx'], 0, data['principal_point_in_pixels']['cx']],
					 			[0, data['focal_lengths_in_pixels']['fy'], data['principal_point_in_pixels']['cy']],
								[0, 0, 1]])

		if os.path.exists(cam_in_3d_path):
			with open(cam_in_3d_path, 'r') as f:
				data = json.load(f)

			cam_in_3d = dict(pos=np.array([data['position_in_meters']['x'], data['position_in_meters']['y'], data['position_in_meters']['z']]),
							 rot=np.array([data['rotation_euler_xyz_in_radians']['roll'],
			  				               data['rotation_euler_xyz_in_radians']['pitch'],
							               data['rotation_euler_xyz_in_radians']['yaw']]))

		if os.path.exists(grasp_pose_path) or \
			(self.allow_grasp_predicted_as_backup and os.path.exists(grasp_pose_predicted_path and os.path.exists(grasp_pose_attempted_path))):
			M = self.fixed_bbox_size

			pregrasp_depth = 0
			if os.path.exists(grasp_path):
				# Open the JSON file
				with open(grasp_path, 'r') as f:
					data = json.load(f)

				# Read the clicked_point_frontal value into pt1
				pt_manual = np.array(data['clicked_point_frontal'])
				pregrasp_depth = data['grasp_depth']

			if os.path.exists(grasp_pose_path):
				grasp_point, grasp_pose = read_competition_pose(grasp_pose_path)
			elif self.allow_grasp_predicted_as_backup and os.path.exists(grasp_pose_predicted_path) and os.path.exists(grasp_pose_attempted_path):

				# read prediction poses and take the one that was attempted for grasp
				with open(grasp_pose_predicted_path, 'r') as f:
					predicted_grasps = json.load(f)

				with open(grasp_pose_attempted_path, 'r') as f:
					attempted_grasp_pose = json.load(f)['attempted_grasp_pose']

				attempted_grasp_pose = np.array(attempted_grasp_pose)

				grasp_data = [np.sqrt(np.sum(np.array([d['position_in_meters']['x'], d['position_in_meters']['y'], d['position_in_meters']['z']]) - attempted_grasp_pose[0:3, 3])**2)
				  					for d in predicted_grasps]
				grasp_data = np.array(grasp_data)
				assert np.min(grasp_data) < 0.001, f"Error finding closest predicted grasp pose to the one attempted: distance more than 0.001 ({np.min(grasp_data)})"
				grasp_data = predicted_grasps[np.argmin(grasp_data)]

				grasp_point = np.array([grasp_data['position_in_meters']['x'], grasp_data['position_in_meters']['y'], grasp_data['position_in_meters']['z']])
				grasp_pose = np.array([grasp_data['rotation_euler_xyz_in_radians']['roll'],
							  			grasp_data['rotation_euler_xyz_in_radians']['pitch'],
										grasp_data['rotation_euler_xyz_in_radians']['yaw']])


			sample['grasp_point'] = grasp_point
			sample['grasp_pose'] = grasp_pose

			assert cam_K is not None, "Missing camera intrinsic matric"
			assert cam_in_3d is not None, "Missing camera-in-3D-world position"

			if self.remove_pregrasp_depth and pregrasp_depth > 0:
				X_W_P = get_homogeneous_matrix(grasp_point, grasp_pose)

				translation_matrix = np.array([
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, -pregrasp_depth],
                        [0, 0, 0, 1]
                    ])

				X_W_P = np.dot(X_W_P, translation_matrix)

				grasp_point, grasp_pose = get_trans_and_euler_from_matrix(X_W_P)

			pt1 = project_grasp_pose_to_image_plane(grasp_point, grasp_pose, cam_K, cam_in_3d)[:2]

			# use Euler angles as default representation returned by this dataset
			grasp_pose_for_regression = grasp_pose_to_eulers_in_world_frame if self.orientation_in_world_frame else grasp_pose_to_eulers_in_camera_frame

			if self.use_6dof_pose:
				direction = [grasp_pose_for_regression(grasp_point, grasp_pose, cam_K, cam_in_3d, axis='x'),
				 			 grasp_pose_for_regression(grasp_point, grasp_pose, cam_K, cam_in_3d, axis='y'),
							 grasp_pose_for_regression(grasp_point, grasp_pose, cam_K, cam_in_3d, axis='z')]
			else:
				direction = grasp_pose_for_regression(grasp_point, grasp_pose, cam_K, cam_in_3d)
			if self.resize_factor is not None and self.resize_factor != 1.0:
				pt1 = pt1 * self.resize_factor

			orientation[:3, int(pt1[1] - M):int(pt1[1] + M), int(pt1[0] - M):int(pt1[0] + M)] = torch.from_numpy(np.array(direction).reshape(-1,1,1))

			label[int(pt1[1] - M):int(pt1[1] + M), int(pt1[0] - M):int(pt1[0] + M)] = 1
			instances[int(pt1[1] - M):int(pt1[1] + M), int(pt1[0] - M):int(pt1[0] + M)] = 1 # only one instance

			centers.append(pt1)

			centers = np.array(centers)

			# read resulting grasping object score if exists
			grasp_result_filename = os.path.join(sample_fn,'observation_result','result.json')

			if os.path.exists(grasp_result_filename):

				with open(grasp_result_filename, 'r') as f:
					grasp_result = json.load(f)

				grasp_score = float(grasp_result['seg_percent_of_reference'])

				orientation[-1, int(pt1[1] - M):int(pt1[1] + M), int(pt1[0] - M):int(pt1[0] + M)] = grasp_score

		if self.crop_by_robot_positions:
			arm_left_base_path = os.path.join(sample_fn,'observation_start','arm_left_pose_in_world.json')
			arm_left_tcp_path = os.path.join(sample_fn,'observation_start','arm_left_tcp_pose_in_world.json')
			arm_right_base_path = os.path.join(sample_fn,'observation_start','arm_right_pose_in_world.json')
			arm_right_tcp_path = os.path.join(sample_fn,'observation_start','arm_right_tcp_pose_in_world.json')

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


			if self.resize_factor != 1.0:
				l, t, r, b = map(lambda x: x*self.resize_factor, [l, t, r, b])

			# we need to first apply all transformation and only then do cropping
			# (so that any resizing or rotation will be preserved)
			# we will hijack centers for to let all transform function convert our crop, and then remove them from centers
			crop_points = np.array([[l,t],[r,b]])
			centers = np.concatenate((centers, crop_points),axis=0) if len(centers) > 0 else crop_points

		if self.use_depth and self.depth_encoding == 'arm_bases_plane':
			assert cam_K is not None and cam_in_3d is not None, "Missing camera calibratio info in sample"

			arm_left_base_path = os.path.join(sample_fn,'observation_start','arm_left_pose_in_world.json')
			arm_left_tcp_path = os.path.join(sample_fn,'observation_start','arm_left_tcp_pose_in_world.json')
			arm_right_base_path = os.path.join(sample_fn,'observation_start','arm_right_pose_in_world.json')

			assert os.path.exists(arm_left_base_path), f"Missing {arm_left_base_path}"
			assert os.path.exists(arm_left_tcp_path), f"Missing {arm_left_tcp_path}"
			assert os.path.exists(arm_right_base_path), f"Missing {arm_right_base_path}"

			arm_left_base, arm_left_base_pose = read_competition_pose(arm_left_base_path)
			arm_left_tcp, arm_left_tcp_pose = read_competition_pose(arm_left_tcp_path)
			arm_right_base, arm_right_base_pose = read_competition_pose(arm_right_base_path)

			# project points to image
			pt_arm_left_base = project_grasp_pose_to_image_plane(arm_left_base, arm_left_base_pose, cam_K, cam_in_3d)[:2]
			pt_arm_left_tcp = project_grasp_pose_to_image_plane(arm_left_tcp, arm_left_tcp_pose, cam_K, cam_in_3d)[:2]

			pt_arm_right_base = project_grasp_pose_to_image_plane(arm_right_base, arm_right_base_pose, cam_K, cam_in_3d)[:2]


			# We convert all pixel values to 3D then use as depth the distances in 3D world points to the plane
			# running between robot bases streaced in Z direction.
			# Actullay this values it he same as X coordinate in 3D world, but we compute it relative to robot bases
			# to be more independent of the coordinate system

			# get inverse camera calibration matrix and camera in 3D woorld matrix
			cam_K_inv = np.linalg.inv(cam_K)
			X_W_C = get_homogeneous_matrix(cam_in_3d["pos"], cam_in_3d["rot"])

			D = sample['depth']
			X, Y = np.meshgrid(np.arange(D.shape[1]), np.arange(D.shape[0]))  # Fixed meshgrid order for width/height

			# Flatten and prepare homogeneous 2D points
			D_flat = D.flatten()
			X_flat = X.flatten()
			Y_flat = Y.flatten()

			ones = np.ones_like(X_flat)
			points_2d_homogeneous = np.vstack((X_flat, Y_flat, ones))

			# Apply the inverse of the intrinsic matrix to get normalized camera coordinates
			points_3d_normalized = cam_K_inv @ points_2d_homogeneous  # (3, N)

			# Scale by the depth to get the 3D points in camera space
			points_3d_camera = points_3d_normalized * D_flat  # (3, N)

			# Convert points to homogeneous 3D coordinates for transformation (4xN)
			points_3d_camera_homogeneous = np.vstack((points_3d_camera, ones))

			# Apply the transformation from camera to world coordinates using the extrinsic matrix
			points_3d_world_homogeneous = X_W_C @ points_3d_camera_homogeneous  # (4, N)

			# Extract the 3D world coordinates from homogeneous coordinates
			points_3d_world = points_3d_world_homogeneous[:3, :]  # (3, N)

			# Reshape back into the original image shape if needed
			points_3d_world = points_3d_world.T.reshape(D.shape[0], D.shape[1], 3)

			# now calculate distance to plane defined by robot bases and Z axis:

			# Points defining the plane
			P1 = arm_right_base # First point in world coordinates
			P2 = arm_left_base # Second point in world coordinates

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
			if self.integrate_depth_with_image:
				sample['depth'] = np.stack((sample['depth'], distances), axis=2)
			else:
				sample['depth'] = distances

		# Apply depth encoding (normals or HHA)
		elif self.use_depth and self.depth_encoding is not None:
			assert cam_K is not None, "Missing camera intrinsic matrix for depth encoding"

			if self.depth_encoding == 'normals':
				# Surface normals encoding: 3 channels (nx, ny, nz)
				normals = get_surface_normal_by_depth_robust(sample['depth'], cam_K)
				# Normalize normals to [0, 1] range: (n + 1) / 2
				depth_encoded = (normals + 1.0) / 2.0  # (H, W, 3)

			elif self.depth_encoding == 'hha':
				# HHA encoding: 3 channels (Horizontal disparity, Height above ground, Angle with gravity)
				depth_encoded = get_hha_encoding(
					sample['depth'],
					K=cam_K,
					camera_height=self.depth_encoding_camera_height,
					normalize=True,
					gaussian_blur_k=self.depth_encoding_gaussian_blur_k
				)  # (H, W, 3)
			elif self.depth_encoding == 'normalized':
				# Apply normalization based on depth_encoding
				depth_encoded = depth[..., np.newaxis] / np.max(depth)
			else:
				depth_encoded = depth[..., np.newaxis]

			# Stack raw depth with encoded depth: (H, W, 1+3) = (H, W, 4)
			if self.integrate_depth_with_image and depth_encoded is not None:
				sample['depth'] = np.concatenate((sample['depth'][..., np.newaxis], depth_encoded), axis=2)
			else:
				sample['depth'] = depth_encoded


		sample['center'] = np.zeros((self.MAX_NUM_CENTERS, 2))
		if len(centers) > 0:
			sample['center'][:centers.shape[0], :] = centers
		sample['label'] = label.unsqueeze(0)
		sample['mask'] = (label > 0).unsqueeze(0)
		sample['ignore'] = torch.zeros((1, im_size[1], im_size[0]), dtype=torch.uint8)
		sample['instance'] = instances.unsqueeze(0)
		sample['orientation'] = orientation
		sample['name'] = im_fn

		if cam_K is not None:
			sample['K'] = np.array(cam_K)

			if self.resize_factor != 1.0:
				sample['K'][0,:] *= self.resize_factor
				sample['K'][1,:] *= self.resize_factor

			sample['cam_in_3d'] = cam_in_3d

		if self.save_robot_arm_data:
			# Read and store robot arm data for IK validation in evaluation
			arm_data_dir = os.path.join(sample_fn, 'observation_start')
			arm_data = {}

			# Left arm
			arm_left_base_f = os.path.join(arm_data_dir, 'arm_left_pose_in_world.json')
			arm_left_tcp_f = os.path.join(arm_data_dir, 'arm_left_tcp_pose_in_world.json')
			arm_left_joints_f = os.path.join(arm_data_dir, 'arm_left_joints.json')

			if os.path.exists(arm_left_base_f):
				pos, rot = read_competition_pose(arm_left_base_f)
				arm_data['left_base_pos'] = pos
				arm_data['left_base_rot'] = rot
			if os.path.exists(arm_left_tcp_f):
				pos, rot = read_competition_pose(arm_left_tcp_f)
				arm_data['left_tcp_pos'] = pos
				arm_data['left_tcp_rot'] = rot
			if os.path.exists(arm_left_joints_f):
				with open(arm_left_joints_f, 'r') as f:
					arm_data['left_joints'] = np.array(json.load(f)['values'])

			# Right arm
			arm_right_base_f = os.path.join(arm_data_dir, 'arm_right_pose_in_world.json')
			arm_right_tcp_f = os.path.join(arm_data_dir, 'arm_right_tcp_pose_in_world.json')
			arm_right_joints_f = os.path.join(arm_data_dir, 'arm_right_joints.json')

			if os.path.exists(arm_right_base_f):
				pos, rot = read_competition_pose(arm_right_base_f)
				arm_data['right_base_pos'] = pos
				arm_data['right_base_rot'] = rot
			if os.path.exists(arm_right_tcp_f):
				pos, rot = read_competition_pose(arm_right_tcp_f)
				arm_data['right_tcp_pos'] = pos
				arm_data['right_tcp_rot'] = rot
			if os.path.exists(arm_right_joints_f):
				with open(arm_right_joints_f, 'r') as f:
					arm_data['right_joints'] = np.array(json.load(f)['values'])

			if arm_data:
				sample['arm_data'] = arm_data

		if self.save_original_image:
			sample['org_image'] = sample['image']

			if 'depth' in sample:
				sample['org_depth'] = sample['depth']
			if 'cam_in_3d' in sample:
				sample['org_cam_in_3d'] = sample['cam_in_3d']
			if 'K' in sample:
				sample['org_K'] = sample['K']

			to_tensor_op = my_transforms.ToTensor(keys=['org_image'] + (['org_depth'] if 'depth' in sample else []), type=torch.FloatTensor)

			to_tensor_op(sample)

		if self.transform is not None:
			import copy
			do_transform = True

			ii = 0
			while do_transform:
				if ii > 10 and ii % 10 == 0:
					print(f"WARNING: unable to generate valid transform for {ii} iterations")
				new_sample = self.transform(copy.deepcopy(sample), self.rng if not self.transform_per_sample_rng else np.random.default_rng(1337))

				out_of_bounds_ids = [id for id, c in enumerate(new_sample['center']) if c[0] < 0 or c[1] < 0 or c[0] >= new_sample['image'].shape[-1] or c[1] >= new_sample['image'].shape[-2]]

				# stop if sufficent centers still visible
				if not self.transform_only_valid_centers or self.transform_only_valid_centers <= 0:
					do_transform = False
					sample = new_sample
				else:
					if type(self.transform_only_valid_centers) == bool:
						if len(out_of_bounds_ids) < len(centers): # at least one must be present
							do_transform = False
							sample = new_sample
					elif type(self.transform_only_valid_centers) == int:
						if len(centers) - len(out_of_bounds_ids) >= self.transform_only_valid_centers:
							do_transform = False
							sample = new_sample
					elif type(self.transform_only_valid_centers) == float:
						min_visible = int(self.transform_only_valid_centers * len(centers))
						if len(centers) - len(out_of_bounds_ids) >= min_visible:
							do_transform = False
							sample = new_sample
					else:
						raise Exception("Invalid type of transform_only_valid_centers, allowed types: bool, int, float")
				ii += 1

		if self.crop_by_robot_positions:
			# retieve last two centers that represent our crop points (and remove them from center list)
			idx = np.where((sample['center'][:, 0] != 0) * (sample['center'][:, 1] != 0))[0]

			(l, t), (r, b) = sample['center'][idx[-2:],:]

			sample['center'][idx[-2:]] = 0


			keys=['image', 'instance', 'label', 'ignore', 'orientation', 'mask']
			if self.segment_cloth:
				keys.append('segmentation_mask')
			if self.use_depth:
				keys.append('depth')
			if self.segement_bg_by_depth_thr:
				keys.append('bg_mask_by_depth')

			if type(self.crop_by_robot_positions) in [tuple, list]:
				if self.compatibility_with_infer_docker:
					# Keep float bounds until final crop origin cast to match infer_docker behavior.
					crop_center = (l+r)/2, (b+t)/2
				else:
					l, t, r, b = map(int, [l, t, r, b])
					crop_center = (l+r)/2, (b+t)/2
				w,h = self.crop_by_robot_positions
				x,y= int(crop_center[1] - h/2), int(crop_center[0] - w/2)
			else:
				l, t, r, b = map(int, [l, t, r, b])
				x,y,w,h = t,l, r-l, b-t

			from utils.transforms import FixedCrop
			crop_op = FixedCrop(keys=keys, keys_bbox=('center',), keys_camK=('K') if cam_K is not None else None,
					   			loc=(x,y), size=(h, w), pad_if_needed=False)

			crop_op(sample)

		if self.use_depth and self.integrate_depth_with_image:
			# sample['depth'] is either:
			#   - (C, H, W) with C>=2: first channel is raw depth, rest are encoded (arm_bases_plane or depth_encoding)
			#   - (1, H, W) or (H, W): just raw depth (no encoding)
			if len(sample['depth'].shape) > 2 and sample['depth'].shape[0] >= 2:
				# Has encoded channels: split raw depth from encoded depth
				depth_to_integrate = sample['depth'][1:]
				sample['depth'] = sample['depth'][0:1]
			else:
				# No encoding, just use raw depth
				depth_to_integrate = sample['depth']

			if self.use_only_depth:
				sample['image'] = torch.cat((sample['image']*0, depth_to_integrate))
			else:
				sample['image'] = torch.cat((sample['image'], depth_to_integrate))

		if self.remove_out_of_bounds_centers:
			# if instance has out-of-bounds center then ignore it if requested so
			out_of_bounds_ids = [id for id, c in enumerate(sample['center'])
							 # if center closer to border then this margin than mark it as truncated
							 if id >= 0 and (c[0] < 0 or c[1] < 0 or
											c[0] >= sample['image'].shape[-1] or
											c[1] >= sample['image'].shape[-2])]
			for id in out_of_bounds_ids:
				sample['instance'][sample['instance'] == id+1] = 0
				sample['orientation'][sample['instance'] == id+1] = 0
				sample['center'][id,:] = -1

		if self.transform is not None and 'instance' in sample:
			# recheck for ignore regions due to changes from augmentation:
			#  - mark with ignore any centers/instances that are now outside of image size
			valid_ids = np.unique(sample['instance'])
			truncated_ids = [id for id, c in enumerate(sample['center'])
							 # if center closer to border then this margin than mark it as truncated
							 if id >= 0 and id in valid_ids and (c[0] < self.BORDER_MARGIN_FOR_CENTER or
																c[1] < self.BORDER_MARGIN_FOR_CENTER or
																c[0] >= sample['image'].shape[-1]-self.BORDER_MARGIN_FOR_CENTER or
																c[1] >= sample['image'].shape[-2]-self.BORDER_MARGIN_FOR_CENTER)]
			for id in truncated_ids:
				sample['ignore'][sample['instance'] == id+1] |= self.IGNORE_TRUNCATED_FLAG


		if self.check_consistency:
			if np.any(np.unique(instances) != torch.unique(sample['instance']).numpy()):
				print(f"error in augmentation - missing one or more sample in instance mask after augmentation (before was {np.unique(instances)} now is {torch.unique(sample['instance'])})")

		return sample
