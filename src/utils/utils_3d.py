import json
import cv2

import numpy as np
from scipy.spatial.transform import Rotation

def get_trans_and_rot_from_matrix(X):
	tran = X[:3, 3]
	rot =  Rotation.from_matrix(X[:3, :3]).as_rotvec()

	return tran, rot

def get_trans_and_euler_from_matrix(X, seq='xyz'):
	tran = X[:3, 3]
	eulers =  Rotation.from_matrix(X[:3, :3]).as_euler(seq)

	return tran, eulers

def get_homogeneous_matrix(pos,euler):
	# Create a homogeneous matrix
	rot = Rotation.from_euler("xyz", euler, degrees=False).as_matrix()

	homogeneous_matrix = np.eye(4)
	homogeneous_matrix[:3, :3] = rot
	homogeneous_matrix[:3, 3] = pos

	return homogeneous_matrix

def project_grasp_pose_to_image_plane(grasp_point, grasp_pose, cam_K, cam_in_3d):

    X_W_C = get_homogeneous_matrix(cam_in_3d["pos"], cam_in_3d["rot"]) if type(cam_in_3d) == dict else cam_in_3d
    X_W_P = get_homogeneous_matrix(grasp_point, grasp_pose)

    frame_pose_in_camera = np.linalg.inv(X_W_C) @ X_W_P

    grasp_trans, grasp_rot = get_trans_and_rot_from_matrix(frame_pose_in_camera)

    # same as: cv2.projectPoints(np.float32([0, 0, 0]), grasp_rot, grasp_trans, cam_K, distCoeffs=None)[0].squeeze()
    X = cam_K @ grasp_trans
    return X / X[-1]

def project_grasp_pose_to_direction_in_image_plane(grasp_point, grasp_pose, cam_K, cam_in_3d, axis='z'):

	grasp_axis = dict(x=1,y=2,z=3)

	X_W_C = get_homogeneous_matrix(cam_in_3d["pos"], cam_in_3d["rot"])
	X_W_P = get_homogeneous_matrix(grasp_point, grasp_pose)

	frame_pose_in_camera = np.linalg.inv(X_W_C) @ X_W_P

	grasp_trans, grasp_rot = get_trans_and_rot_from_matrix(frame_pose_in_camera)

	axis_points = np.float32([[0, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]).reshape(-1, 3)
	grasp_2d_axis = cv2.projectPoints(axis_points, grasp_rot, grasp_trans, cam_K, distCoeffs=None)[0]

	origin = tuple(grasp_2d_axis[0].ravel())
	grasp_axis = tuple(grasp_2d_axis[grasp_axis[axis]].ravel())

	angle_in_image = np.arctan2(origin[0] - grasp_axis[0],
								origin[1] - grasp_axis[1])

	return float(angle_in_image)

def print_grasp_pose_to_direction_in_image_plane(img, grasp_point, grasp_pose, cam_K, cam_in_3d):

	X_W_C = get_homogeneous_matrix(cam_in_3d["pos"], cam_in_3d["rot"])
	X_W_P = get_homogeneous_matrix(grasp_point, grasp_pose)

	frame_pose_in_camera = np.linalg.inv(X_W_C) @ X_W_P

	grasp_trans, grasp_rot = get_trans_and_rot_from_matrix(frame_pose_in_camera)

	return cv2.drawFrameAxes(img, cam_K, None, grasp_rot, grasp_trans, 0.1)

def rotation_to_image_point(point, rot):
    R = 100
    return point - np.array([np.sin(rot) * R,
                             np.cos(rot) * R, ])



def project_from_image_to_world(point, depth, rot_by_axis, C_K_inv, X_W_C):

    x = np.array([point[0], point[1], 1])

    p = (C_K_inv @ x)
    p = (p / p[-1]) * depth

    p = np.array(list(p) + [1])

    p_in_3d = X_W_C @ p

    if len(rot_by_axis) > 1:
        x_axis = np.array(list(rotation_to_image_point(point, rot_by_axis[0])) + [1])
        y_axis = np.array(list(rotation_to_image_point(point, rot_by_axis[1])) + [1])
        z_axis = np.array(list(rotation_to_image_point(point, rot_by_axis[2])) + [1])

        p_x_axis = (C_K_inv @ x_axis)
        p_y_axis = (C_K_inv @ z_axis)
        p_z_axis = (C_K_inv @ y_axis)

        # Define the function to solve
        def func(tuv):
            t, u, v = tuv
            X_axis = t * p_x_axis
            Y_axis = u * p_y_axis
            Z_axis = v * p_z_axis
            XP = X_axis - p[:-1]
            YP = Y_axis - p[:-1]
            ZP = Z_axis - p[:-1]
            return [np.dot(XP, YP), np.dot(YP, ZP), np.dot(ZP, XP)]

        tuv0 = [0, 0, 0]

        # Solve the system of equations
        from scipy.optimize import fsolve
        t, u, v = fsolve(func, tuv0)

        X_axis = (X_W_C @ np.array(list(t * p_x_axis - p[:3])+[1]))[:3]
        Y_axis = (X_W_C @ np.array(list(u * p_y_axis - p[:3])+[1]))[:3]
        Z_axis = (X_W_C @ np.array(list(v * p_z_axis - p[:3])+[1]))[:3]

        def get_rot_from_axis(V, axis):
            V = V / np.linalg.norm(V)
            vx = np.cross(V, axis)
            cx = np.dot(V, axis)
            return Rotation.from_rotvec(np.arccos(cx) * vx / np.linalg.norm(vx))

        R_in_3D = np.column_stack((X_axis / np.linalg.norm(X_axis),
                             Y_axis / np.linalg.norm(Y_axis),
                             Z_axis / np.linalg.norm(Z_axis)))

        rot_by_x = get_rot_from_axis(X_axis, np.array([1, 0, 0]))
        rot_by_y = get_rot_from_axis(Y_axis, np.array([0, 1, 0]))
        rot_by_z = get_rot_from_axis(Z_axis, np.array([0, 0, 1]))

        R_in_3D_by_x = rot_by_x.as_matrix()
        R_in_3D_by_y = rot_by_y.as_matrix()
        R_in_3D_by_z = rot_by_z.as_matrix()

        euler_in_3D_by_x = rot_by_x.as_euler('xyz')
        euler_in_3D_by_y = rot_by_y.as_euler('xyz')
        euler_in_3D_by_z = rot_by_z.as_euler('xyz')

        #R_in_3D = R_in_3D_by_x

    else:
        R_in_3D = Rotation.from_rotvec(np.array([0, 0, rot_by_axis ])).as_matrix()

    X_in_3d = np.identity(4)
    X_in_3d[:3,3] = p_in_3d[:3]
    X_in_3d[:3,:3] = R_in_3D

    return X_in_3d


def grasp_pose_to_eulers_in_camera_frame(grasp_point, grasp_pose, cam_K, cam_in_3d, axis='z'):

    X_W_C = get_homogeneous_matrix(cam_in_3d["pos"], cam_in_3d["rot"])
    X_W_P = get_homogeneous_matrix(grasp_point, grasp_pose)

    frame_pose_in_camera = np.linalg.inv(X_W_C) @ X_W_P

    grasp_axis = dict(x=0,y=1,z=2)
    rot = Rotation.from_matrix(frame_pose_in_camera[:3, :3]).as_euler("xyz")

    return rot[grasp_axis[axis]]

def grasp_pose_to_eulers_in_world_frame(grasp_point, grasp_pose, cam_K, cam_in_3d, axis='z'):

    X_W_P = get_homogeneous_matrix(grasp_point, grasp_pose)

    grasp_axis = dict(x=0,y=1,z=2)
    rot = Rotation.from_matrix(X_W_P[:3, :3]).as_euler("xyz")

    return rot[grasp_axis[axis]]

def euler_to_quaternion_matrix_(euler_angles):
    """
    Convert a 3D tensor of Euler angles to a 4D tensor of quaternions using scipy.
    
    Parameters:
    euler_angles : torch.Tensor : A 3D array of shape (3, W, H) where
                                the first dimension corresponds to [roll, pitch, yaw].
    
    Returns:
    torch.Tensor : A 4D array of shape (4, W, H) representing quaternions [w, x, y, z].
    """
    import torch
    # Reshape euler_angles for scipy's Rotation
    N_AXIS = len(euler_angles.shape)
    if N_AXIS == 3:
        euler_angles = euler_angles.permute(1, 2, 0)  # from (3, W, H) to (W, H, 3)
    elif N_AXIS == 2:
        euler_angles = euler_angles.permute(1, 0)  # from (3, N) to (N, 3)
    else:
        raise Exception("Can only handle 2d or 3d data")

    # Create Rotation objects from Euler angles
    rotation = Rotation.from_euler('xyz', euler_angles.cpu().numpy(), degrees=False)

    # Get the quaternion representation
    quaternions = rotation.as_quat()  # Shape (W, H, 4)

    quaternions = torch.from_numpy(quaternions).to(euler_angles.device)

    # Transpose to get shape (4, W, H)
    if N_AXIS == 3:
        quaternions = quaternions.permute(2, 0, 1)  # Shape (4, W, H)
    elif N_AXIS == 2:
        quaternions = quaternions.permute(1, 0)  # Shape (4, N)
    else:
        raise Exception("Can only handle 2d or 3d data")

    return quaternions

def quaternion_to_euler_matrix_(quaternions):
    """
    Convert a 4D tensor of quaternions to a 3D tensor of Euler angles.
    
    Parameters:
    quaternions : torch.Tensor : A 4D array of shape (4, W, H) 
                                representing quaternions [w, x, y, z].
    
    Returns:
    torch.Tensor : A 3D array of shape (3, W, H) where 
                 the first dimension corresponds to [roll, pitch, yaw].
    """
    import torch
    N_AXIS = len(quaternions.shape)
    # Reshape euler_angles for scipy's Rotation
    if N_AXIS == 3:
        quaternions = quaternions.permute(1, 2, 0)  # from (4, W, H) to (W, H, 4)
    elif N_AXIS == 2:
        quaternions = quaternions.permute(1, 0)  # from (4, N) to (N, 4)

    # Create Rotation objects from quaternion representation
    rotation = Rotation.from_quat(quaternions.cpu().numpy())

    # Get the Euler angles
    euler_angles = rotation.as_euler("xyz", degrees=False)  # Shape (W, H, 3)

    euler_angles = torch.from_numpy(euler_angles).to(quaternions.device)

    # Transpose to get shape (3, W, H)
    if N_AXIS == 3:
        euler_angles = euler_angles.permute(2, 0, 1)  # Shape (3, W, H)
    elif N_AXIS == 2:
        euler_angles = euler_angles.permute(1, 0)  # Shape (3, N)
    else:
        raise Exception("Can only handle 2d or 3d data")

    return euler_angles


import torch


def euler_to_quaternion_matrix(euler_angles):
    """
    Convert Euler angles (XYZ convention) to quaternions.
    
    Parameters:
    euler_angles : torch.Tensor : A tensor of shape (..., 3) where the last dimension 
                                   represents the Euler angles [roll, pitch, yaw].
    
    Returns:
    torch.Tensor : A tensor of quaternions of shape (..., 4) representing 
                   [w, x, y, z].
    """
    # Ensure the input tensor is of the correct shape
    assert euler_angles.dim() >= 2 and euler_angles.size(0) == 3, "Input must have shape (..., 3)"

    # Extract Euler angles
    roll, pitch, yaw  = euler_angles.unbind(dim=0)

    # Calculate the quaternion components
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    w = cy * cp * cr + sy * sp * sr

    # Stack the quaternion components into a single tensor
    quaternions = torch.stack((x, y, z, w), dim=0)

    return quaternions


def quaternion_to_euler_matrix(quat):
    """
    Convert a tensor of quaternions to Euler angles (XYZ convention).
    
    Parameters:
    quat : torch.Tensor : A tensor of shape (..., 4) where the last dimension 
                          represents the quaternion [w, x, y, z].
    
    Returns:
    torch.Tensor : A tensor of Euler angles of shape (..., 3) representing 
                   [yaw, pitch, roll].
    """
    # Ensure the input tensor is of the correct shape
    assert quat.dim() >= 2 and quat.size(0) == 4, "Input must have shape (..., 4)"

    # Extract quaternion components
    x, y, z, w = quat.unbind(dim=0)  # Unbind the last dimension into separate tensors

    # Calculate Euler angles (XYZ convention)
    roll = torch.atan2(2 * (x * w + y * z), 1 - 2 * (y**2 + x**2))
    pitch = torch.asin(torch.clamp(2 * (x * y - w * z), -1.0, 1.0))
    yaw = torch.atan2(2 * (y * w + x * z), 1 - 2 * (z**2 + y**2))

    # Stack  roll, pitch, yaw into a single tensor
    euler_angles = torch.stack((roll, pitch, yaw), dim=0)

    return euler_angles



def read_competition_pose(pose_path, as_homogeneous_matrix=False):
    with open(pose_path, 'r') as f:
        data = json.load(f)

    point = np.array([data['position_in_meters']['x'], data['position_in_meters']['y'], data['position_in_meters']['z']])
    pose = np.array([data['rotation_euler_xyz_in_radians']['roll'],
                    data['rotation_euler_xyz_in_radians']['pitch'],
                    data['rotation_euler_xyz_in_radians']['yaw']])
    if as_homogeneous_matrix:
        return get_homogeneous_matrix(point, pose)
    else:
        return point, pose

def read_competition_joints(joints_path):
    with open(joints_path, 'r') as f:
        data = json.load(f)

    return np.array(data['values'])
