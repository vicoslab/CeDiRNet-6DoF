import numpy as np, cv2, glob, os
from matplotlib import pyplot as plt
from numpy.linalg import norm, inv

def get_surface_normal_by_depth(depth, K=None):
	"""
	depth: (h, w) of float, the unit of depth is meter
	K: (3, 3) of float, the depth camere's intrinsic
	"""
	K = [[1, 0], [0, 1]] if K is None else K
	fx, fy = K[0][0], K[1][1]

	dz_dv, dz_du = np.gradient(depth)  # u, v mean the pixel coordinate in the image
	du_dx = fx / depth  # x is xyz of camera coordinate
	dv_dy = fy / depth

	dz_dx = dz_du * du_dx
	dz_dy = dz_dv * dv_dy
	normal_cross = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth)))
	normal_unit = normal_cross / np.linalg.norm(normal_cross, axis=2, keepdims=True)
	normal_unit[~np.isfinite(normal_unit).all(2)] = [0, 0, 1]
	return normal_unit

def get_surface_normal_by_depth_robust(depth, K):
    # depth: (h,w) with actual metric depth
    # K: (3,3) camera intrinsics [fx, 0, cx; 0, fy, cy; 0,0,1]

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # Back project to 3D
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    # Compute spatial derivatives of 3D points
    dX_dv, dX_du = np.gradient(X)
    dY_dv, dY_du = np.gradient(Y)
    dZ_dv, dZ_du = np.gradient(Z)

    # Two tangent vectors
    t1 = np.stack((dX_du, dY_du, dZ_du), axis=2)
    t2 = np.stack((dX_dv, dY_dv, dZ_dv), axis=2)

    normal = np.cross(t1, t2)
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal_unit = normal / (norm + 1e-8)

    # Handle invalid: find pixels where any of the 3 components is not finite
    invalid_mask = ~np.isfinite(normal_unit).all(axis=2)
    normal_unit[invalid_mask] = [0, 0, 1]
    return normal_unit


def get_angle_from_depth(depth, household=False, k=15):
	# https://math.stackexchange.com/questions/3433645/how-can-i-find-the-angle-of-the-surface-3d-plane

	depth = cv2.GaussianBlur(depth,(k,k),0)

	# get normals from depth
	if household: # for kinect images
		fx = 1081.3720703125
		cx = 959.5
		cy = 539.5
	else:
		fx = 256
		cx = 256.0
		cy = 256.0

	K = np.array([[fx,0,cx],[0,fx,cy],[0,0,1]])
	normals = get_surface_normal_by_depth(depth, K=K)

	return normals

def eul2rot(theta) :

	R = np.array(
		[
			[np.cos(theta[1])*np.cos(theta[2]), np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
			[np.sin(theta[2])*np.cos(theta[1]), np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
			[-np.sin(theta[1]),np.sin(theta[0])*np.cos(theta[1]),np.cos(theta[0])*np.cos(theta[1])]
		]
  	)

	return R

def get_normals(depth, normals_mode=1, household=False, k=15):

	normals = get_angle_from_depth(depth, household=household, k=k)

	# 1 is normal vector as 3 channels
	# 2 is angle between normal vector and each of the axes
	# 3 is Sobel operator x and y

	if normals_mode==1:
		normals = get_angle_from_depth(depth, household=household, k=k)
	elif normals_mode==2:
		dx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=k)
		dy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=k)

		dx[np.isnan(dx)]=0
		dy[np.isnan(dy)]=0
		dx[np.isinf(dx)]=0
		dy[np.isinf(dy)]=0

		dx/=np.max(dx)
		dy/=np.max(dy)
		normals = np.dstack((dx,dy))

	return normals

	normals = get_angle_from_depth(depth, household=household, k=k)

	# 1 is normal vector as 3 channels
	# 2 is dot product of normal vector with reference normal vector in 1 channel
	# 3 is angle between normal plane and reference vector in 1 channel
	# 4 is same as 3, except expressed as sin and cos

	n = np.zeros_like(normals)
	n[...,0]=reference_normal[0]
	n[...,1]=reference_normal[1]
	n[...,2]=reference_normal[2]

	if normals_mode==1:
		depth = normals.copy()
	elif normals_mode==2:
		depth = np.sum(normals*n, axis=-1)
	elif normals_mode==3:
		depth = np.sum(normals*n, axis=-1)

		# print("dot 0", np.unique((normals*n)[...,0]))
		# print("dot 1", np.unique((normals*n)[...,1]))
		# print("dot 2", np.unique((normals*n)[...,2]))

		depth = np.arccos(depth)
		# TODO normalize
	elif normals_mode==4:
		# print("dot 0", np.unique((normals*n)[...,0]))
		# print("dot 1", np.unique((normals*n)[...,1]))
		# print("dot 2", np.unique((normals*n)[...,2]))
		depth = np.sum(normals*n, axis=-1)
		angle = np.arccos(depth)
		depth = np.dstack((np.sin(angle), np.cos(angle)))
		# print("depth", depth.shape)
	elif normals_mode==5:
		# depth = normals.copy()

		# x1 = 220
		# x2 = 1051
		# y1 = 400
		# y2 = 1530

		# mask = np.zeros_like(depth)
		# mask[313:1534, 286:1051]=1
		# mask[x1:x2, y1:y2]=1
		# mask = mask.astype(bool)
		# depth[~mask]=0


		dx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=k)
		dy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=k)

		dx[np.isnan(dx)]=0
		dy[np.isnan(dy)]=0
		dx[np.isinf(dx)]=0
		dy[np.isinf(dy)]=0

		dx/=np.max(dx)
		dy/=np.max(dy)

		# print("dx", dx.shape, dx.dtype)
		# print("dy", dy.shape, dy.dtype)
		# print("min max dx", np.min(dx), np.max(dx))
		# print("min max dy", np.min(dy), np.max(dy))

		# dx[x1-k:x1+k, :] = 0
		# dx[x2-k:x2+k, :] = 0
		# dx[:, y1-k:y1+k] = 0
		# dx[:, y2-k:y2+k] = 0

		# dy[x1-k:x1+k, :] = 0
		# dy[x2-k:x2+k, :] = 0
		# dy[:, y1-k:y1+k] = 0
		# dy[:, y2-k:y2+k] = 0


		depth = np.dstack((dx,dy))

	return depth

def rotate_depth(depth, R, K):

	h, w = depth.shape

	xx, yy = np.meshgrid(range(w), range(h))
	xx = np.ravel(xx)
	yy = np.ravel(yy)
	dd = np.ravel(depth)

	pc = np.vstack((xx,yy,dd))
	pc = inv(K)@pc
	pc = K@R@pc

	d_rotated = pc[-1,:].reshape(h,w)

	return d_rotated


def depth_to_point_cloud(depth, K):
	"""
	Convert depth map to 3D point cloud in camera coordinates.
	
	Args:
		depth: (h, w) depth map in meters
		K: (3, 3) camera intrinsic matrix
		
	Returns:
		points: (h, w, 3) point cloud with X, Y, Z coordinates
	"""
	h, w = depth.shape
	fx, fy = K[0, 0], K[1, 1]
	cx, cy = K[0, 2], K[1, 2]

	# Create pixel coordinate grids
	u, v = np.meshgrid(np.arange(w), np.arange(h))

	# Convert to camera coordinates
	Z = depth
	X = (u - cx) * Z / fx
	Y = (v - cy) * Z / fy

	points = np.dstack((X, Y, Z))
	return points


def compute_horizontal_disparity(depth, K, baseline=1.0):
	"""
	Compute horizontal disparity from depth.
	
	Disparity is inversely proportional to depth: d = f * B / Z
	where f is focal length, B is baseline, Z is depth.
	
	Args:
		depth: (h, w) depth map in meters
		K: (3, 3) camera intrinsic matrix
		baseline: baseline distance (for stereo cameras), default 1.0
		
	Returns:
		disparity: (h, w) horizontal disparity map
	"""
	fx = K[0, 0]

	# Avoid division by zero
	valid_mask = depth > 0
	disparity = np.zeros_like(depth)
	disparity[valid_mask] = (fx * baseline) / depth[valid_mask]

	return disparity


def compute_height_above_ground(depth, K, camera_height=1.0, gravity_direction=None):
	"""
	Compute height above ground plane for each pixel.
	
	Args:
		depth: (h, w) depth map in meters
		K: (3, 3) camera intrinsic matrix
		camera_height: height of camera above ground plane in meters
		gravity_direction: (3,) unit vector pointing in gravity direction in camera frame
						   Default is [0, 1, 0] (Y-down convention)
						   
	Returns:
		height: (h, w) height above ground for each pixel
	"""
	if gravity_direction is None:
		gravity_direction = np.array([0, 1, 0])  # Y-axis points down
	gravity_direction = gravity_direction / np.linalg.norm(gravity_direction)

	# Get 3D point cloud
	points = depth_to_point_cloud(depth, K)

	# Height is the projection onto gravity direction, offset by camera height
	# For Y-down convention: height = camera_height - Y
	height = camera_height - np.sum(points * gravity_direction, axis=-1)

	# Set invalid depth regions to 0
	height[depth <= 0] = 0

	return height


def compute_angle_with_gravity(depth, K, gravity_direction=None, gaussian_blur_k=5):
	"""
	Compute the angle between local surface normals and gravity direction.
	
	Args:
		depth: (h, w) depth map in meters
		K: (3, 3) camera intrinsic matrix
		gravity_direction: (3,) unit vector pointing in gravity direction in camera frame
						   Default is [0, 1, 0] (Y-down convention)
		gaussian_blur_k: kernel size for Gaussian blur before computing normals
		
	Returns:
		angle: (h, w) angle with gravity in radians [0, pi]
	"""
	if gravity_direction is None:
		gravity_direction = np.array([0, 1, 0])
	gravity_direction = gravity_direction / np.linalg.norm(gravity_direction)

	# Smooth depth to reduce noise in normal estimation
	if gaussian_blur_k > 0:
		depth_smooth = cv2.GaussianBlur(depth, (gaussian_blur_k, gaussian_blur_k), 0)
	else:
		depth_smooth = depth

	# Compute surface normals
	normals = get_surface_normal_by_depth(depth_smooth, K)

	# Compute dot product with gravity direction
	# angle = arccos(dot(normal, gravity))
	dot_product = np.sum(normals * gravity_direction, axis=-1)

	# Clamp to valid range for arccos
	dot_product = np.clip(dot_product, -1.0, 1.0)
	angle = np.arccos(dot_product)

	# Set invalid regions to 0
	angle[depth <= 0] = 0

	return angle


def get_hha_encoding(depth, K=None, camera_height=1.0, gravity_direction=None,
					 baseline=1.0, normalize=True, gaussian_blur_k=5):
	"""
	Compute HHA (Horizontal disparity, Height above ground, Angle with gravity) encoding.
	
	This converts a depth map into a 3-channel image where:
	- Channel 0: Horizontal disparity (inversely proportional to depth)
	- Channel 1: Height above ground plane
	- Channel 2: Angle between local surface normal and gravity direction
	
	Reference: Gupta et al., "Learning Rich Features from RGB-D Images for Object 
	Detection and Segmentation", ECCV 2014
	
	Args:
		depth: (h, w) depth map in meters
		K: (3, 3) camera intrinsic matrix. If None, uses default values.
		camera_height: height of camera above ground plane in meters
		gravity_direction: (3,) unit vector pointing in gravity direction in camera frame
						   Default is [0, 1, 0] (Y-down convention)
		baseline: baseline distance for disparity computation
		normalize: if True, normalizes each channel to [0, 1] range
		gaussian_blur_k: kernel size for Gaussian blur before computing normals
		
	Returns:
		hha: (h, w, 3) HHA encoded image
	"""
	h, w = depth.shape

	# Default camera intrinsics if not provided
	if K is None:
		fx = fy = min(h, w)
		cx, cy = w / 2.0, h / 2.0
		K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

	if gravity_direction is None:
		gravity_direction = np.array([0, 1, 0])

	# Compute the three channels
	horizontal_disparity = compute_horizontal_disparity(depth, K, baseline)
	height_above_ground = compute_height_above_ground(depth, K, camera_height, gravity_direction)
	angle_with_gravity = compute_angle_with_gravity(depth, K, gravity_direction, gaussian_blur_k)

	# Stack into HHA image
	hha = np.dstack((horizontal_disparity, height_above_ground, angle_with_gravity))

	if normalize:
		hha = normalize_hha(hha, depth)

	return hha


def normalize_hha(hha, depth=None):
	"""
	Normalize HHA encoding to [0, 1] range for each channel.
	
	Args:
		hha: (h, w, 3) HHA encoded image
		depth: (h, w) original depth map (optional, used for masking invalid regions)
		
	Returns:
		hha_normalized: (h, w, 3) normalized HHA image
	"""
	hha_normalized = np.zeros_like(hha)

	# Create valid mask
	if depth is not None:
		valid_mask = depth > 0
	else:
		valid_mask = np.ones(hha.shape[:2], dtype=bool)

	for i in range(3):
		channel = hha[:, :, i]
		valid_values = channel[valid_mask]

		if len(valid_values) > 0:
			min_val = np.percentile(valid_values, 1)  # Use percentiles for robustness
			max_val = np.percentile(valid_values, 99)

			if max_val > min_val:
				normalized = (channel - min_val) / (max_val - min_val)
				normalized = np.clip(normalized, 0, 1)
			else:
				normalized = np.zeros_like(channel)

			hha_normalized[:, :, i] = normalized

	# Zero out invalid regions
	if depth is not None:
		hha_normalized[~valid_mask] = 0

	return hha_normalized


def hha_to_uint8(hha):
	"""
	Convert normalized HHA encoding to uint8 image for visualization/saving.
	
	Args:
		hha: (h, w, 3) HHA encoded image (should be normalized to [0, 1])
		
	Returns:
		hha_uint8: (h, w, 3) uint8 image
	"""
	hha_uint8 = (np.clip(hha, 0, 1) * 255).astype(np.uint8)
	return hha_uint8


def get_hha_encoding_from_kinect(depth, camera_height=1.0, normalize=True):
	"""
	Compute HHA encoding using default Kinect camera parameters.
	
	Args:
		depth: (h, w) depth map in meters (or millimeters, will be auto-detected)
		camera_height: height of camera above ground plane in meters
		normalize: if True, normalizes each channel to [0, 1] range
		
	Returns:
		hha: (h, w, 3) HHA encoded image
	"""
	# Kinect v2 intrinsics (typical values)
	K = np.array([
		[1081.3720703125, 0, 959.5],
		[0, 1081.3720703125, 539.5],
		[0, 0, 1]
	], dtype=np.float64)

	# Auto-detect if depth is in millimeters and convert to meters
	depth_float = depth.astype(np.float64)
	if np.nanmax(depth_float) > 100:  # Likely in millimeters
		depth_float = depth_float / 1000.0

	return get_hha_encoding(depth_float, K, camera_height, normalize=normalize)


def visualize_hha(hha, save_path=None):
	"""
	Visualize HHA encoding with separate plots for each channel.
	
	Args:
		hha: (h, w, 3) HHA encoded image
		save_path: if provided, saves the figure to this path
	"""
	fig, axes = plt.subplots(1, 4, figsize=(16, 4))

	# Channel names
	channel_names = ['Horizontal Disparity', 'Height Above Ground', 'Angle with Gravity']
	cmaps = ['viridis', 'RdYlBu', 'twilight']

	for i, (name, cmap) in enumerate(zip(channel_names, cmaps)):
		im = axes[i].imshow(hha[:, :, i], cmap=cmap)
		axes[i].set_title(name)
		axes[i].axis('off')
		plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

	# Combined RGB view
	axes[3].imshow(hha)
	axes[3].set_title('HHA Combined')
	axes[3].axis('off')

	plt.tight_layout()

	if save_path:
		plt.savefig(save_path, dpi=150, bbox_inches='tight')

	plt.show()
