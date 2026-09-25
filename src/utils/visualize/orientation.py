import os
from functools import partial
from dataclasses import dataclass

import numpy as np
import torch

import cv2

from utils.transforms import inverse_transform
from utils.visualize.vis import Visualizer


@dataclass
class PoseAxisPlot:
    lines: np.ndarray
    axis_order: np.ndarray


def _extract_pose_lines_and_order(pose_axis):
    if isinstance(pose_axis, PoseAxisPlot):
        lines = pose_axis.lines
        order = pose_axis.axis_order
    else:
        lines = pose_axis
        order = np.arange(len(lines), dtype=np.int32)

    lines = np.asarray(lines)
    order = np.asarray(order, dtype=np.int32)
    return lines, order

class RotationMatrixParser():

    def __init__(self, sample, length = 0.1):
        assert 'depth' in sample, 'Missing "depth" in sample needed for 3d information'
        assert 'K' in sample, 'Missing camera calibration matrix "K" in sample needed for 3d information'

        self.sample = sample

        gt_depth = sample['depth']
        gt_cam_K = sample['K']

        assert gt_depth is not None, "Depth information required if ploting from axis_from_eulers"
        assert gt_cam_K is not None, "Camera calibration (cam_K) information required if ploting from axis_from_eulers"

        self.depth = gt_depth
        self.cam_K = gt_cam_K

        self.length = length

    def plot_pose_axis(self, p):
        raise Exception("NOT IMPLEMENTED")

    def plot_from_rotation_matrix(self, p, rot):
        length = self.length
        cam_K = self.cam_K.cpu().numpy().squeeze()

        j, i = p

        pt = np.linalg.inv(cam_K) @ np.array([j, i, 1])
        pt = (pt / pt[-1]) * self.depth[0,0,int(i),int(j)].item()

        points = np.float32([[length, 0, 0], [0, length, 0], [0, 0, length], [0, 0, 0]]).reshape(-1, 3)
        axisPoints, _ = cv2.projectPoints(points, rot, pt, cam_K, (0, 0, 0, 0))

        points_cam = (rot @ points.T).T + pt.reshape(1, 3)
        axis_tip_depth = points_cam[:3, 2]
        axis_order = np.argsort(axis_tip_depth)[::-1]

        plot_axis = []
        for axis in [0,1,2]:
            d = axisPoints[axis].ravel()
            o = axisPoints[-1].ravel()
            plot_axis.append([o,d])

        return PoseAxisPlot(lines=np.array(plot_axis), axis_order=axis_order)

class EulerAnglesParser(RotationMatrixParser):

    def __init__(self, sin_eulers, cos_eulers, sample, length = 0.1):
        super().__init__(sample, length)

        self.sin_eulers = sin_eulers
        self.cos_eulers = cos_eulers

    def plot_pose_axis(self, p):
        j, i = p

        sin_eulers = self.sin_eulers[:,int(i),int(j)]
        cos_eulers = self.cos_eulers[:,int(i),int(j)]

        # convert sin/cos to eulers
        eulers = torch.atan2(sin_eulers, cos_eulers)  # convert to angle to fix scaling
        eulers = eulers.cpu().numpy()

        from scipy.spatial.transform import Rotation
        rot = Rotation.from_euler('xyz',eulers).as_matrix()

        return self.plot_from_rotation_matrix(p, rot)

    @staticmethod
    def create_pose_plotter(gt, pred, sample, **kwargs):
        assert 'sin' in gt and 'cos' in gt, "Missing 'sin' and/or 'cos' in provided gt fields for create_pose_plotter() call. Ensure you pass values returned by EulerAnglesParser.parse_gt_and_prediction_fields()"
        assert 'sin' in pred and 'cos' in pred, "Missing 'sin' and/or 'cos' in provided pred fields for create_pose_plotter() call. Ensure you pass values returned by EulerAnglesParser.parse_gt_and_prediction_fields()"

        # return classes for ploting euler pose for gt and predictions
        return dict(plot_gt_orientation_fn=EulerAnglesParser(sin_eulers=gt['sin'], cos_eulers=gt['cos'], sample=sample, length=0.1).plot_pose_axis,
                     plot_orientation_fn=EulerAnglesParser(sin_eulers=pred['sin'], cos_eulers=pred['cos'], sample=sample, length=0.1).plot_pose_axis)

    @staticmethod
    def parse_gt_and_prediction_fields(output, centerdir_gt, dims, OFFSET_IN_OUTPUT=3):
        assert dims == 3, 'Requires 6-DoF regression when ploting pose axis from eulers'

        # use the same fields as axis-alinged angles
        return AxisAlignedParser.parse_gt_and_prediction_fields(output, centerdir_gt, dims, OFFSET_IN_OUTPUT)

class EulerAnglesWorld3DoFParser(EulerAnglesParser):
    """Special parser for 3-DoF world-frame euler setup.

    Reconstructs full world eulers from regressed z-angle and converts them to
    camera-frame eulers before projection (same convention as infer_docker).
    """

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    def _cam_world_transform(self):
        cam_in_3d = self.sample.get('cam_in_3d', None)
        if cam_in_3d is None:
            return None

        if isinstance(cam_in_3d, dict):
            pos = self._to_numpy(cam_in_3d['pos']).squeeze()
            rot = self._to_numpy(cam_in_3d['rot']).squeeze()

            from utils.utils_3d import get_homogeneous_matrix
            return get_homogeneous_matrix(pos, rot)

        cam_in_3d = self._to_numpy(cam_in_3d).squeeze()
        if cam_in_3d.shape == (4, 4):
            return cam_in_3d

        return None

    def plot_pose_axis(self, p):
        j, i = p

        sin_eulers = self.sin_eulers[:, int(i), int(j)]
        cos_eulers = self.cos_eulers[:, int(i), int(j)]

        eulers = torch.atan2(sin_eulers, cos_eulers).detach().cpu().numpy()
        eulers = np.array([-np.pi / 2, 0, eulers[0]], dtype=np.float64)

        from scipy.spatial.transform import Rotation

        X_W_C = self._cam_world_transform()
        if X_W_C is not None:
            X_W_P = np.identity(4)
            X_W_P[:3, :3] = Rotation.from_euler('xyz', eulers).as_matrix()
            frame_pose_in_camera = np.linalg.inv(X_W_C) @ X_W_P
            eulers = Rotation.from_matrix(frame_pose_in_camera[:3, :3]).as_euler('xyz')

        rot = Rotation.from_euler('xyz', eulers).as_matrix()
        return self.plot_from_rotation_matrix(p, rot)

    @staticmethod
    def create_pose_plotter(gt, pred, sample, **kwargs):
        assert 'sin' in gt and 'cos' in gt, "Missing 'sin' and/or 'cos' in provided gt fields for create_pose_plotter() call. Ensure you pass values returned by EulerAnglesWorld3DoFParser.parse_gt_and_prediction_fields()"
        assert 'sin' in pred and 'cos' in pred, "Missing 'sin' and/or 'cos' in provided pred fields for create_pose_plotter() call. Ensure you pass values returned by EulerAnglesWorld3DoFParser.parse_gt_and_prediction_fields()"

        return dict(
            plot_gt_orientation_fn=EulerAnglesWorld3DoFParser(sin_eulers=gt['sin'], cos_eulers=gt['cos'], sample=sample, length=0.1).plot_pose_axis,
            plot_orientation_fn=EulerAnglesWorld3DoFParser(sin_eulers=pred['sin'], cos_eulers=pred['cos'], sample=sample, length=0.1).plot_pose_axis,
        )

    @staticmethod
    def parse_gt_and_prediction_fields(output, centerdir_gt, dims, OFFSET_IN_OUTPUT=3):
        assert dims == 1, 'Requires single-axis (3-DoF world-euler) regression for world parser'
        return AxisAlignedParser.parse_gt_and_prediction_fields(output, centerdir_gt, dims, OFFSET_IN_OUTPUT)

class QuaternionsParser(RotationMatrixParser):

    def __init__(self, quaternions, sample, length = 0.1):
        super().__init__(sample, length)

        self.quaternions = quaternions

    def plot_pose_axis(self, p):
        j, i = p

        quaternions = self.quaternions[:,int(i),int(j)]
        quaternions = quaternions.cpu().numpy()

        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat(quaternions).as_matrix()

        return self.plot_from_rotation_matrix(p, rot)

    @staticmethod
    def create_pose_plotter(gt, pred, sample, **kwargs):
        assert 'quat' in gt, "Missing 'quat' in provided gt fields for create_pose_plotter() call. Ensure you pass values returned by EulerAnglesParser.parse_gt_and_prediction_fields()"
        assert 'quat' in pred, "Missing 'quat' in provided pred fields for create_pose_plotter() call. Ensure you pass values returned by EulerAnglesParser.parse_gt_and_prediction_fields()"

        # return classes for ploting quaternions pose for gt and predictions
        return dict(plot_gt_orientation_fn=QuaternionsParser(quaternions=gt['quat'], sample=sample, length=0.1).plot_pose_axis,
                    plot_orientation_fn=QuaternionsParser(quaternions=pred['quat'], sample=sample, length=0.1).plot_pose_axis)

    @staticmethod
    def parse_gt_and_prediction_fields(output, centerdir_gt, dims, OFFSET_IN_OUTPUT=3):
        assert dims == 3, 'Requires 6-DoF regression when ploting pose axis from quaternions'

        QUATERNIONS_DIMS = 4 # quaternion representation always have 4 dimensions

        # get groundtruth for quaternions
        from models.center_groundtruth import CenterDirGroundtruth
        gt_orientation_quaternion = CenterDirGroundtruth.parse_groundtruth_map(centerdir_gt, keys=['gt_orientation_quaternion'])
        gt_orientation_quaternion = gt_orientation_quaternion[:QUATERNIONS_DIMS,0]

        # get predicted quaterntions
        orientation_quat_pred = output[0, OFFSET_IN_OUTPUT:OFFSET_IN_OUTPUT+QUATERNIONS_DIMS, ...]

        # return groundtruth fields and prediction fields
        return dict(quat=gt_orientation_quaternion), dict(quat=orientation_quat_pred)

class AxisAlignedParser():

    def __init__(self, sin_angle, cos_angle, R=30):

        self.sin_angle = sin_angle
        self.cos_angle = cos_angle

        self.R = R

    def plot_pose_axis(self, p):
        R = self.R
        j, i = p

        sin_angle = sin_angle[:,int(i),int(j)]
        cos_angle = cos_angle[:,int(i),int(j)]

        pred_angle = torch.atan2(sin_angle, cos_angle)+np.pi  # convert to angle to fix scaling

        sin_angle = torch.sin(pred_angle)
        cos_angle = torch.cos(pred_angle)

        plot_axis = []
        for s,c in zip(sin_angle, cos_angle):
            plot_axis.append((p, (p[0] + c * R, p[1] + s * R)))

        return np.array(plot_axis)

    @staticmethod
    def create_pose_plotter(gt, pred, **kwargs):
        assert 'sin' in gt and 'cos' in gt, "Missing 'sin' and/or 'cos' in provided gt fields for create_pose_plotter() call. Ensure you pass values returned by EulerAnglesParser.parse_gt_and_prediction_fields()"
        assert 'sin' in pred and 'cos' in pred, "Missing 'sin' and/or 'cos' in provided pred fields for create_pose_plotter() call. Ensure you pass values returned by EulerAnglesParser.parse_gt_and_prediction_fields()"

        # return classes for ploting axis-alinged pose for gt and predictions
        return dict(plot_gt_orientation_fn=AxisAlignedParser(sin_angle=gt['sin'], cos_angle=gt['cos'], R=50).plot_pose_axis,
                    plot_orientation_fn=AxisAlignedParser(sin_angle=pred['sin'], cos_angle=pred['cos'], R=50).plot_pose_axis)

    @staticmethod
    def parse_gt_and_prediction_fields(output, centerdir_gt, dims, OFFSET_IN_OUTPUT=3):
        # get groundtruth for default axis-aligned angles
        from models.center_groundtruth import CenterDirGroundtruth
        gt_orientation_sin, gt_orientation_cos = CenterDirGroundtruth.parse_groundtruth_map(centerdir_gt, keys=['gt_orientation_sin', 'gt_orientation_cos'])

        gt_orientation_sin = gt_orientation_sin[:dims,0]
        gt_orientation_cos = gt_orientation_cos[:dims,0]

        # get predictions for axis-alinged angles
        orientation_sin_pred = output[0, OFFSET_IN_OUTPUT:OFFSET_IN_OUTPUT+dims, ...]
        orientation_cos_pred = output[0, OFFSET_IN_OUTPUT+dims:OFFSET_IN_OUTPUT+dims+dims, ...]

        # return groundtruth fields and prediction fields
        return dict(sin=gt_orientation_sin, cos=gt_orientation_cos), \
               dict(sin=orientation_sin_pred, cos=orientation_cos_pred)

class InversePoseWrapper:
    """Wraps a pose plotter function to inverse-transform the plotted axis points.
    
    The original parsers (RotationMatrixParser, EulerAnglesParser, etc.) work in the
    transformed image space (e.g. cropped/resized/rotated). They compute pose axes
    endpoints in that space using feature maps that correspond to that space.
    
    This wrapper:
    1. If lookup points provided: Maps input point (Original Space) -> Transformed Space.
    1. Calls the original plotter with the point in TRANSFORMED space (so it can read feature maps correctly).
    2. Receives axis endpoints in TRANSFORMED space.
    3. Uses inverse_transform to map these endpoints back to ORIGINAL image space.
    """
    def __init__(self, plotter_fn, sample, lookup_points_orig=None, lookup_points_trans=None, is_gt=False):
        self.plotter_fn = plotter_fn
        self.sample = sample
        self.lookup_points_orig = lookup_points_orig
        self.lookup_points_trans = lookup_points_trans
        self.is_gt = is_gt

    def __call__(self, p):
        """
        Args:
           p: point (x,y).
        
        Returns:
           np.array of axes lines in ORIGINAL image space.
        """
        p_transformed = p

        if self.lookup_points_orig is not None:
             # Find closest point in original set
            dists = np.sum((self.lookup_points_orig - p)**2, axis=1)
            idx = np.argmin(dists)

            # Get corresponding transformed point
            p_transformed = self.lookup_points_trans[idx]

            # If GT, p_trans is (y,x), inner parser needs (x,y)
            if self.is_gt:
                p_transformed = p_transformed[::-1]

        # 1. Get axes in transformed space
        # plot_axis shape is typically (3, 2, 2) i.e. [axis_idx, start/end, x/y]
        # or (N, 2, 2)
        axes_transformed = self.plotter_fn(p_transformed)
        axes_lines, axis_order = _extract_pose_lines_and_order(axes_transformed)

        if len(axes_lines) == 0:
            return axes_lines

        # axes_transformed is (N_axes, 2, 2) where last dim is (x, y)
        n_axes = axes_lines.shape[0]

        # Flatten to list of points for batch inverse transform
        # shape (N_axes * 2, 2)
        points_flat = axes_lines.reshape(-1, 2)

        # 2. Inverse transform these points
        inv_res = inverse_transform(self.sample, points=[points_flat])
        points_orig = inv_res['points'][0]

        # 3. Reshape back to axes format
        axes_orig = points_orig.reshape(n_axes, 2, 2)

        return PoseAxisPlot(lines=axes_orig, axis_order=axis_order)

class OrientationVisualizeTest(Visualizer):
    KEYS = ['image', 'centers', 'centers-est', 'pred', 'sigma', 'seed', 'gt-diff', 'centerdir_gt']

    def __init__(self, keys=(), show_rot_axis=(True,True,True), orientation_type='axis', axis_from_eulers=False, show_regressed_maps=True,
                 show_gt_maps=True, plot_rank_number=False, bgr_axis=False, plot_score=False,  show_confidence_score=False,
                 plot_gt_pose=True, plot_pose_in_regression=True, pose_line_thickness=4, use_light_pose_colors=False,
                 show_original_image=False, save_jpeg=False, z_axis_thickness_boost=2, **kwargs):
        super(OrientationVisualizeTest, self).__init__(keys=self.KEYS + list(keys), **kwargs)

        self.show_original_image = show_original_image
        self.save_jpeg = save_jpeg

        self.show_rot_axis = show_rot_axis
        self.axis_from_eulers = axis_from_eulers
        self.orientation_type = orientation_type

        if self.axis_from_eulers:
            if self.orientation_type.lower() != 'euler':
                print("WARNING: using legacy axis_from_eulers=True ... setting orientation_type to 'euler' in OrientationVisualizeTest!!")
            self.orientation_type = 'euler'

        if self.orientation_type.lower() == 'axis':
            self.orientation_cls = AxisAlignedParser
        elif self.orientation_type.lower() == 'euler':
            self.orientation_cls = EulerAnglesParser
        elif self.orientation_type.lower() == '3dof-euler-world':
            self.orientation_cls = EulerAnglesWorld3DoFParser
        elif self.orientation_type.lower() == 'quaternions':
            self.orientation_cls = QuaternionsParser
        else:
            raise Exception("Unknown orinetation type in OrientationVisualizeTest")


        self.show_regressed_maps = show_regressed_maps
        self.show_gt_maps = show_gt_maps
        self.show_confidence_score = show_confidence_score

        self.plot_rank_number = plot_rank_number
        self.plot_score = plot_score

        self.plot_gt_pose = plot_gt_pose
        self.plot_pose_in_regression = plot_pose_in_regression
        self.pose_line_thickness = pose_line_thickness
        self.z_axis_thickness_boost = z_axis_thickness_boost

        if bgr_axis:
            self.color_rot_axis = [(0, 0, 128), (0, 128, 0),(128, 0, 0)]
            self.color_rot_axis_gt = [(128, 128, 255),(128, 255, 128),(255, 128, 128)]
        else:
            self.color_rot_axis = [(180, 0, 0), (0, 128, 0),(0, 0, 128)]
            self.color_rot_axis_gt=[(255, 128, 128), (128, 255, 128),(128, 128, 255)]

        if use_light_pose_colors:
            tmp = self.color_rot_axis
            self.color_rot_axis = self.color_rot_axis_gt
            self.color_rot_axis_gt = tmp

    def parse_results(self, sample, result, difficult):
        im = sample['image'][0]
        output = result['output']
        output_cls, resnet_features, CAM = result.get('output'), result.get('resnet_features'), result.get('CAM')
        center_est_imshow = result['pred_heatmap']
        centerdir_gt = sample.get('centerdir_groundtruth')
        gt_centers_dict = sample['center_dict']

        gt_list = np.array([gt_centers_dict[k] for k in sorted(gt_centers_dict.keys())])

        is_difficult_gt = np.array([difficult[np.clip(int(c[0]), 0, difficult.shape[0] - 1),
                                              np.clip(int(c[1]), 0, difficult.shape[1] - 1),].item() != 0 for c in gt_list])

        return im, output, output_cls, resnet_features, CAM, center_est_imshow, centerdir_gt, gt_list, is_difficult_gt

    @staticmethod
    def plot_orientations_cv(img, pred_list, plot_orientation_fn, plot_gt_orientation_fn, pred_poly_mask, pred_match, gt_list, is_difficult_gt, gt=False,
                             bbox=False, predictions_args=dict(), bbox_args=dict(), show_rot_axis=(True,True,True),
                             color_rot_axis=[(128, 0, 0), (0, 128, 0),(0, 0, 128)], color_rot_axis_gt=[(255, 128, 128), (128, 255, 128),(128, 128, 255)],
                             plot_rank_number=False, confidence_heatmap=None, pose_line_thickness=4, plot_score=False, pred_score=None,
                             z_axis_thickness_boost=2):
        predictions_args_ = dict(markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        predictions_args_.update(predictions_args)
        bbox_args_ = dict(thickness=1)
        bbox_args_.update(bbox_args)

        if show_rot_axis is None:
            show_rot_axis = (True,True,True)

        if gt:
            for i, p in enumerate(gt_list):
                cv2.circle(img, (int(p[1]), int(p[0])), radius=4, color=(0, 255, 0) if is_difficult_gt[i] == 0 else (0, 255, 255), thickness=-1)
                cv2.circle(img, (int(p[1]), int(p[0])), radius=4, color=(0, 0, 0), thickness=1)

                plot_axis = plot_gt_orientation_fn(p[::-1])
                plot_axis, axis_order = _extract_pose_lines_and_order(plot_axis)

                assert len(show_rot_axis) == len(plot_axis)

                for axis_idx in axis_order:
                    pt = plot_axis[axis_idx]
                    color = color_rot_axis_gt[axis_idx]
                    show = show_rot_axis[axis_idx]
                    if show:
                        line_thickness = 4 + (z_axis_thickness_boost if axis_idx == 2 else 0)
                        cv2.line(img, (int(pt[0][0]), int(pt[0][1])), (int(pt[1][0]), int(pt[1][1])), color=(255, 255, 255), thickness=line_thickness + 4)
                        cv2.line(img, (int(pt[0][0]), int(pt[0][1])), (int(pt[1][0]), int(pt[1][1])), color=(0, 0, 0), thickness=line_thickness + 2)
                        cv2.line(img, (int(pt[0][0]), int(pt[0][1])), (int(pt[1][0]), int(pt[1][1])), color=color, thickness=line_thickness)

        if len(pred_list) > 0:
            pred_list_true = pred_list[pred_match[:, 0] > 0, :]
            pred_list_false = pred_list[pred_match[:, 0] <= 0, :]

            pred_rank_true = np.argwhere(pred_match[:, 0] > 0)
            pred_rank_false = np.argwhere(pred_match[:, 0] <= 0)

            for p in pred_list_true: cv2.drawMarker(img, (int(p[0]), int(p[1])), color=(0, 255, 0), **predictions_args_)
            for p in pred_list_false: cv2.drawMarker(img, (int(p[0]), int(p[1])), color=(0, 0, 255), **predictions_args_)

            for plot_list, pred_rank, color in [(pred_list_true, pred_rank_true, (0, 255, 0)),
                                                (pred_list_false, pred_rank_false, (0, 0, 255))]:
                for p,rank in zip(plot_list,pred_rank):
                    plot_axis = plot_orientation_fn(p)
                    plot_axis, axis_order = _extract_pose_lines_and_order(plot_axis)

                    assert len(show_rot_axis) == len(plot_axis)

                    for axis_idx in axis_order:
                        pt = plot_axis[axis_idx]
                        color = color_rot_axis[axis_idx]
                        show = show_rot_axis[axis_idx]
                        if show:
                            line_thickness = pose_line_thickness + (z_axis_thickness_boost if axis_idx == 2 else 0)
                            cv2.line(img, (int(pt[0][0]), int(pt[0][1])), (int(pt[1][0]), int(pt[1][1])), color=(255, 255, 255), thickness=line_thickness + 4)
                            cv2.line(img, (int(pt[0][0]), int(pt[0][1])), (int(pt[1][0]), int(pt[1][1])), color=(0, 0, 0), thickness=line_thickness + 2)
                            cv2.line(img, (int(pt[0][0]), int(pt[0][1])), (int(pt[1][0]), int(pt[1][1])),
                                     color=color, thickness=line_thickness)

                    if plot_rank_number:
                        text = f"{rank[0]}"
                        if plot_score:
                            text= f"{text}: {pred_score[rank[0]]*100:.1f}"
                        if confidence_heatmap is not None:
                            text= f"{text}: {confidence_heatmap[0,int(p[1]), int(p[0])]*100:.1f}"
                        cv2.putText(img, text, (int(pt[0][0]), int(pt[0][1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color = (255, 0, 0) , thickness=2)


        if bbox:
            poly_true = [np.round(poly[:, [1, 0]]).astype(np.int) for poly, is_correct_ap50, iou in pred_poly_mask.values()
                         if is_correct_ap50]
            poly_false = [np.round(poly[:, [1, 0]]).astype(np.int) for poly, is_correct_ap50, iou in pred_poly_mask.values()
                          if not is_correct_ap50]
            cv2.polylines(img, poly_true, isClosed=True, color=(0, 255, 0), **bbox_args_)
            cv2.polylines(img, poly_false, isClosed=True, color=(0, 0, 255), **bbox_args_)

        return img

    def visualize_pylab(self, sample, result, pred_list, pred_score, pred_poly_mask, pred_gt_match, difficult, base, save_dir=None, plot_bbox_all=False):
        raise Exception("NOT IMPLEMENTED") # not implemented any more

    def visualize_opencv(self, sample, result, pred_data, pred_gt_match, difficult, base, save_dir, plot_bbox_all=False):

        pred_poly_mask = pred_data.get('polygon')
        if len(pred_data['center']) > 0:
            pred_list, pred_angle = map(np.array, zip(*[(pred_data['center'][id],
                                                        pred_data['angle'][id]) for id in sorted(pred_data['center'].keys())]))

            pred_score = np.array([pred_data['score'][id] for id in sorted(pred_data['center'].keys())]) if 'score' in pred_data else []
        else:
            pred_list, pred_angle, pred_score = [], [], []


        im, output, output_cls, resnet_features, CAM, \
        center_est_imshow, centerdir_gt, gt_list, is_difficult_gt = self.parse_results(sample, result, difficult)

        orientation_dims = result['pred_attributes']['angle'].shape[-1]

        sin_pred = output[0, 0, ...]
        cos_pred = output[0, 1, ...]

        assert centerdir_gt is not None

        from models.center_groundtruth import CenterDirGroundtruth

        ORIENT_OFFSET_IN_OUTPUT=3
        gt_sin_th, gt_cos_th = CenterDirGroundtruth.parse_groundtruth_map(centerdir_gt, keys=['gt_sin_th', 'gt_cos_th'])

        # parse groundtruth and prediction fields based on type of orientation (i.e. using AxisAlignedParser, EulerAnglesParser or QuaternionsParser)
        orient_gt_fields, orient_pred_fields = self.orientation_cls.parse_gt_and_prediction_fields(output, centerdir_gt, dims=orientation_dims, OFFSET_IN_OUTPUT=ORIENT_OFFSET_IN_OUTPUT)

        # get function pointers for ploting rotation pose (prediced and gt) based on the defined orientation representation (i.e. using AxisAlignedParser, EulerAnglesParser or QuaternionsParser)
        plot_orientation_args = self.orientation_cls.create_pose_plotter(orient_gt_fields, orient_pred_fields, sample)

        # function pointer for ploting detection and rotation pose
        plot_predictions = partial(self.plot_orientations_cv,
                                pred_list=pred_list, pred_poly_mask=pred_poly_mask, pred_match=pred_gt_match, gt_list=gt_list,
                                is_difficult_gt=is_difficult_gt, show_rot_axis=self.show_rot_axis, plot_rank_number=self.plot_rank_number,
                                color_rot_axis=self.color_rot_axis, color_rot_axis_gt=self.color_rot_axis_gt,
                                pose_line_thickness=self.pose_line_thickness, z_axis_thickness_boost=self.z_axis_thickness_boost,
                                **plot_orientation_args)

        if self.show_confidence_score:
            confidence_heatmap = output[0, ORIENT_OFFSET_IN_OUTPUT+2*orientation_dims:ORIENT_OFFSET_IN_OUTPUT+2*orientation_dims+1, ...]
            plot_predictions = partial(plot_predictions, confidence_heatmap=confidence_heatmap.detach().cpu())


        if self.plot_score:
            plot_predictions = partial(plot_predictions, plot_score=self.plot_score, pred_score=pred_score)

        im = im.cpu()
        if im.shape[0]>3:
            im = im[:3,...]

        fig_img = self.display_opencv(im, 'image', plot_fn=partial(plot_predictions, gt=self.plot_gt_pose, bbox=True))

        if self.show_original_image and 'transforms' in sample:

            # --- Dedicated Inverse Transform Visualization Block ---
            # 1. Prepare points for inverse transform
            inv_points = []
            inv_point_labels = []

            # pred_list is (N, 2) as (x, y) - can use directly
            if len(pred_list) > 0:
                inv_points.append(pred_list.copy())
                inv_point_labels.append('pred')

            # gt_list is (N, 2) as (y, x) - convert to (x, y) for inverse_transform
            if len(gt_list) > 0:
                gt_as_xy = gt_list[:, ::-1].copy()  # (y,x) -> (x,y)
                inv_points.append(gt_as_xy)
                inv_point_labels.append('gt')

            # 2. Perform Inverse Transform
            # Instead of inverse transforming the already transformed image 'im',
            # we use the stored original image from the sample if available.

            if 'org_image' in sample:
                im_orig = sample['org_image'][0].cpu()
                # Inverse transform only the points
                inv_result = inverse_transform(sample,
                                            images=[im],
                                            points=inv_points if inv_points else None)
            else:
                # Fallback: Inverse transform the transformed image
                inv_result = inverse_transform(sample,
                                            images=[im],
                                            points=inv_points if inv_points else None)
                im_orig = inv_result['images'][0]

            # Extract transformed points in ORIGINAL space
            pred_list_orig = []
            gt_list_orig = []

            idx = 0
            if 'pred' in inv_point_labels:
                pred_list_orig = inv_result['points'][idx]
                idx += 1
            else:
                pred_list_orig = np.array([])

            if 'gt' in inv_point_labels:
                gt_as_xy_inv = inv_result['points'][idx]
                gt_list_orig = gt_as_xy_inv[:, ::-1].copy()  # (x,y) -> (y,x)
                idx += 1
            else:
                gt_list_orig = np.array([])

            # 3. Create Wrapped Plotters for Inverse Visualization
            # They need to map: Point(Original) -> Point(Transformed) -> plot -> Inverse Axes

            plot_args_orig = plot_orientation_args.copy()

            # Wrap prediction plotter
            if 'plot_orientation_fn' in plot_args_orig and len(pred_list) > 0:
                plot_args_orig['plot_orientation_fn'] = InversePoseWrapper(
                    plot_args_orig['plot_orientation_fn'],
                    sample,
                    lookup_points_orig=pred_list_orig, # Original space
                    lookup_points_trans=pred_list,      # Transformed space (current scope vars)
                    is_gt=False
                )

            # Wrap GT plotter
            if 'plot_gt_orientation_fn' in plot_args_orig and len(gt_list) > 0:
                plot_args_orig['plot_gt_orientation_fn'] = InversePoseWrapper(
                    plot_args_orig['plot_gt_orientation_fn'],
                    sample,
                    lookup_points_orig=gt_list_orig,   # Original space
                    lookup_points_trans=gt_list,        # Transformed space (current scope vars)
                    is_gt=True
                )

            # 4. Create Partial Plotter for Original Image
            plot_predictions_orig = partial(self.plot_orientations_cv,
                                          pred_list=pred_list_orig,
                                          pred_poly_mask=pred_poly_mask, # poly masks not transformed here, ignoring for now or TODO
                                          pred_match=pred_gt_match,
                                          gt_list=gt_list_orig,
                                          is_difficult_gt=is_difficult_gt,
                                          show_rot_axis=self.show_rot_axis,
                                          plot_rank_number=self.plot_rank_number,
                                          color_rot_axis=self.color_rot_axis,
                                          color_rot_axis_gt=self.color_rot_axis_gt,
                                          pose_line_thickness=self.pose_line_thickness,
                                          z_axis_thickness_boost=self.z_axis_thickness_boost,
                                          **plot_args_orig)

            if self.show_confidence_score:
                 # Confidence heatmap is in transformed space, so plotting text lookup might be wrong
                 # unless we also warp the heatmap back?
                 # For now, let's skip confidence heatmap text on original image or accept mismatch
                 pass

            if self.plot_score:
                plot_predictions_orig = partial(plot_predictions_orig, plot_score=self.plot_score, pred_score=pred_score)

            # 5. Display Original Image
            fig_original_image = self.display_opencv(im_orig, 'original_image',
                                plot_fn=partial(plot_predictions_orig, gt=self.plot_gt_pose, bbox=False)) # bbox disabled as poly mask not inv transformed yet

        if self.show_regressed_maps:
            # add sin/cos and tan2 of cedirnet
            regressed_maps = [sin_pred.detach().cpu(), cos_pred.detach().cpu(), torch.atan2(sin_pred, cos_pred).detach().cpu()]
            regressed_maps_colormap = [cv2.COLORMAP_PARULA, cv2.COLORMAP_PARULA, cv2.COLORMAP_HSV]

            # add all prediction fields for orientation as defualt COLORMAP_PARULA
            regressed_maps += [ch.detach().cpu() for field in orient_pred_fields.values() for ch in field]
            regressed_maps_colormap += [cv2.COLORMAP_PARULA for field in orient_pred_fields.values() for ch in field]

            # if prediction fields for orientation contain sin and cos then also add atan2 of it
            if 'sin' in orient_pred_fields and 'cos' in orient_pred_fields:
                regressed_maps += list(torch.atan2(orient_pred_fields['sin'], orient_pred_fields['cos']).detach().cpu())
                regressed_maps_colormap += [cv2.COLORMAP_HSV] * len(orient_pred_fields['sin'])

            # add field for confidence score
            if self.show_confidence_score:
                regressed_maps += list(confidence_heatmap.detach().cpu())
                regressed_maps_colormap += [cv2.COLORMAP_PARULA]

            fig_centers = self.display_opencv(regressed_maps,'centers',plot_fn=partial(plot_predictions, bbox=plot_bbox_all) if self.plot_pose_in_regression else None, image_colormap=regressed_maps_colormap )

            fig_centers_conv = self.display_opencv(
                 [f.detach().cpu() if type(f) == torch.Tensor else f for f in center_est_imshow],
                 'centers-est',
                 plot_fn=partial(plot_predictions, bbox=plot_bbox_all, predictions_args=dict(thickness=1)) if self.plot_pose_in_regression else None)

            seed = output[0][3].cpu()
            fig_seed = self.display_opencv(seed, 'seed', plot_fn=partial(plot_predictions, bbox=plot_bbox_all) if self.plot_pose_in_regression else None)

        if output_cls is not None and resnet_features is not None:
            fig_cam = self.display_opencv((CAM[0] * (CAM[0] > 0.10 * CAM[0].max()).float()).cpu(), 'CAM', # , vmin=-1, vmax=1)
                                          plot_fn=partial(self.plot_predictions_cam_cv,
                                                          pred_list=pred_list,pred_match=pred_gt_match))

        if centerdir_gt is not None and self.show_gt_maps:
            # add diff between gt and pred for CeDiRNet fields
            gt_diff_maps = [torch.abs(sin_pred.detach().cpu() - gt_sin_th[0].cpu()),
                            torch.abs(cos_pred.detach().cpu() - gt_cos_th[0].cpu()),]

            # add diff between gt and pred for orientation fields
            gt_diff_maps += [torch.abs(ch_pred.detach().cpu() - ch_gt.cpu())
                                    for k in orient_gt_fields.keys() for ch_gt,ch_pred in zip(orient_gt_fields[k], orient_pred_fields[k])]
            # also just add gt for orientation fields
            gt_diff_maps += [ch for field in orient_gt_fields.values() for ch in field]

            fig_centerdir = self.display_opencv(gt_diff_maps, 'gt-diff',
                                                plot_fn=partial(plot_predictions, bbox=plot_bbox_all))
        if save_dir is not None:
            ext = 'jpg' if self.save_jpeg else 'png'
            cv2.imwrite(os.path.join(save_dir, f'{base}_0.img.{ext}' ), fig_img)
            if 'fig_centers' in locals():
                cv2.imwrite(os.path.join(save_dir, f'{base}_1.centers.{ext}'), fig_centers)
            if 'fig_centers_conv' in locals():
                cv2.imwrite(os.path.join(save_dir, f'{base}_2.centers_conv.{ext}'), fig_centers_conv)
            if 'fig_seed' in locals():
                cv2.imwrite(os.path.join(save_dir, f'{base}_3.seed.{ext}'), fig_seed)
            if 'fig_cam' in locals():
                cv2.imwrite(os.path.join(save_dir, f'{base}_5.cam.{ext}'), fig_cam)
            if 'fig_centerdir' in locals():
                cv2.imwrite(os.path.join(save_dir, f'{base}_1.gt-diff.{ext}'), fig_centerdir)
            if 'fig_original_image' in locals():
                cv2.imwrite(os.path.join(save_dir, f'{base}_original_image.{ext}'), fig_original_image)

class OrientationVisualizeTrain(Visualizer):
    # default keys for visualization windows
    KEYS = ['image', 'centers', 'pred', 'sigma', 'seed', 'centerdir_gt', 'conv_centers', 'fourier_mask']

    def __init__(self, keys=(), **kwargs):
        super(OrientationVisualizeTrain, self).__init__(keys=self.KEYS + list(keys), **kwargs)

    def visualize_pylab(self, im, output, pred_mask=None, center_conv_resp=None, centerdir_gt=None,
                        gt_centers_dict=None, gt_difficult=None, log_r_fn=None, plot_batch_i=0, device=None, denormalize_args=None):
        from models.center_groundtruth import CenterDirGroundtruth

        with torch.no_grad():
            gt_list = np.array([c for k, c in gt_centers_dict[plot_batch_i].items()]) if gt_centers_dict is not None else []
            is_difficult_gt = np.array([False] * len(gt_list))
            if gt_difficult is not None:
                gt_difficult = gt_difficult[plot_batch_i]
                is_difficult_gt = np.array([gt_difficult[np.clip(int(c[0]),0,gt_difficult.shape[0]-1),
                                                         np.clip(int(c[1]),0,gt_difficult.shape[1]-1)].item() != 0 for c in gt_list])

            # print("centerdir_gt.shape", centerdir_gt[0].shape)
            # print("output.shape", output.shape) # if shape[1]==10 -> 6dof, else shape[1]==6
            ENABLE_6DOF = output.shape[1]>=10

            r = 30

            gt_sin_th, gt_cos_th = CenterDirGroundtruth.parse_groundtruth_map(centerdir_gt,['gt_sin_th', 'gt_cos_th'])
            gt_sin_orientation, gt_cos_orientation = CenterDirGroundtruth.parse_groundtruth_map(centerdir_gt,['gt_orientation_sin','gt_orientation_cos'])

            if ENABLE_6DOF:

                orientation_sin_gt = gt_sin_orientation[plot_batch_i].clone().cpu().numpy()[:,0,...]
                orientation_cos_gt = gt_cos_orientation[plot_batch_i].clone().cpu().numpy()[:,0,...]

                # print("orientation_sin_gt", orientation_sin_gt.shape)
                # print("orientation_cos_gt", orientation_cos_gt.shape)

                _, ax = self.display(im[plot_batch_i].cpu()[:3,...], 'image', denormalize_args=denormalize_args)
                self.plot_gt(ax, gt_list, is_difficult_gt)


                # print(gt_list)
                for p in gt_list:
                    s = orientation_sin_gt[2, int(p[0]), int(p[1])]
                    c = orientation_cos_gt[2, int(p[0]), int(p[1])]
                    ax.plot([p[1], p[1]+c*r], [p[0], p[0]+s*r], 'lime')

                    gt_angle = np.degrees(np.arctan2(s,c))
                    # print("gt_angle", gt_angle)

                    # s = sin_pred[int(p[0]), int(p[1])]
                    # c = cos_pred[int(p[0]), int(p[1])]

                    # pred_angle = np.arctan2(s,c) # convert to angle to fix scaling
                    # s = np.sin(pred_angle)
                    # c = np.cos(pred_angle)



                    # ax.plot([p[1], p[1]+c*r], [p[0], p[0]+s*r], 'r')

                out_viz = [(output[plot_batch_i, 0]).detach().cpu(),
                            (output[plot_batch_i, 1]).detach().cpu()]
                out_viz += list(output[plot_batch_i, 3:9].detach().cpu())
                if centerdir_gt is not None and len(centerdir_gt) > 0:

                    # print("gt_sin_orientation", gt_sin_orientation.shape)
                    # print("gt_cos_orientation", gt_cos_orientation.shape)

                    out_viz += [torch.abs(output[plot_batch_i, 0] - gt_sin_th[plot_batch_i, 0].to(device)).detach().cpu(),
                                torch.abs(output[plot_batch_i, 1] - gt_cos_th[plot_batch_i, 0].to(device)).detach().cpu(),]
                    out_viz += list(torch.abs(output[plot_batch_i, 3:9] - torch.cat((gt_sin_orientation[plot_batch_i, :,0], gt_cos_orientation[plot_batch_i, :,0])).to(device)).detach().cpu())
                else:
                    out_viz += [(output[plot_batch_i, 3 + i]).detach().cpu() for i in
                                range(len(output[plot_batch_i, :]) - 4)]

                # print(len(out_viz))

                _, ax = self.display(out_viz, 'centers')
                # self.plot_gt(ax, gt_list, is_difficult_gt)

                # if center_conv_resp is not None:
                #     conv_centers = center_conv_resp[plot_batch_i].detach().cpu()
                #     if centerdir_gt is not None and len(centerdir_gt) > 0:
                #         gt_center_mask = centerdir_gt[0][plot_batch_i, 5]
                #         conv_centers = [conv_centers,
                #                         torch.abs(center_conv_resp[plot_batch_i] - gt_center_mask.to(device)).detach().cpu()]

                #     _, ax = self.display(conv_centers, 'conv_centers')
                #     self.plot_gt(ax, gt_list, is_difficult_gt)

                if centerdir_gt is not None and len(centerdir_gt) > 0:
                    gt_R, gt_theta = CenterDirGroundtruth.parse_groundtruth_map(centerdir_gt,['gt_R', 'gt_theta'])

                    out_viz = [gt_R[plot_batch_i, 0].cpu(),
                               gt_theta[plot_batch_i, 0].cpu(),
                               gt_sin_th[plot_batch_i, 0].cpu(),
                               gt_cos_th[plot_batch_i, 0].cpu()]
                    out_viz += list(np.concatenate((orientation_sin_gt, orientation_cos_gt)))

                    _, ax = self.display(out_viz, 'centerdir_gt')
                    self.plot_gt(ax, gt_list, is_difficult_gt)
            else:
                from matplotlib import pyplot as plt

                orientation_sin_gt = gt_sin_orientation[plot_batch_i].clone().cpu().numpy()[0:1,0,...]
                orientation_cos_gt = gt_cos_orientation[plot_batch_i].clone().cpu().numpy()[0:1,0,...]

                S = output[plot_batch_i, 3].detach().cpu().numpy()
                C = output[plot_batch_i, 4].detach().cpu().numpy()
                angle = np.arctan2(S,C)+np.pi
                skip = 10
                scale = 0.15
                scale_units = 'x'
                [X,Y] = np.meshgrid(np.arange(0,angle.shape[0], skip),np.arange(0,angle.shape[1], skip))
                S = np.sin(angle)
                C = np.cos(angle)
                s = S[X,Y]
                c = C[X,Y]

                # print("orientation_sin_gt", orientation_sin_gt.shape)
                im = im[plot_batch_i].cpu()[:3,...]
                # print("im", im.shape, im.dtype)
                _, ax = self.display(im, 'image', denormalize_args=denormalize_args)
                self.plot_gt(ax, gt_list, is_difficult_gt)

                ax = [ax]
                for a in ax:
                    a.quiver(Y, X, s, c, angles="xy", scale=scale, scale_units=scale_units)

                    for p in gt_list:
                        sx = orientation_sin_gt[0, int(p[0]), int(p[1])]
                        cx = orientation_cos_gt[0, int(p[0]), int(p[1])]
                        angle = np.arctan2(sx,cx)+np.pi
                        sx = np.sin(angle)
                        cx = np.cos(angle)
                        a.plot([p[1], p[1]+sx*r], [p[0], p[0]+cx*r], 'lime')

                out_viz = [(output[plot_batch_i, 0]).detach().cpu(),
                            (output[plot_batch_i, 1]).detach().cpu(),
                            (output[plot_batch_i, 2]).detach().cpu(),
                            (output[plot_batch_i, 3]).detach().cpu(),
                            (output[plot_batch_i, 4]).detach().cpu(),
                            (output[plot_batch_i, 5]).detach().cpu(),
                ]

                _, ax = self.display(out_viz, 'pred')
                gt_R, gt_theta = CenterDirGroundtruth.parse_groundtruth_map(centerdir_gt,['gt_R', 'gt_theta'])

                out_viz = [gt_R[plot_batch_i, 0].cpu(),
                            gt_theta[plot_batch_i, 0].cpu(),
                            gt_sin_th[plot_batch_i, 0].cpu(),
                            gt_cos_th[plot_batch_i, 0].cpu(),
                            orientation_sin_gt,  orientation_cos_gt]

                _, ax = self.display(out_viz, 'centerdir_gt')
                self.plot_gt(ax, gt_list, is_difficult_gt)
