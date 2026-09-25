try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import json
import random
import os

import numpy as np
from PIL import Image
import cv2
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from torchvision.transforms import InterpolationMode

import torch

def get_image_size(im):
    if isinstance(im,torch.Tensor) or isinstance(im, np.ndarray):
        return im.shape[-2:][::-1]
    elif isinstance(im, Image.Image):
        return im.size
    else:
        raise Exception("Cannot get size of NON-tensor/numpy/PIL image object")

class CropRandomObject:

    def __init__(self, keys=[],keys_bbox=[],object_key="instance", size=100):
        self.keys = keys
        self.keys_bbox = keys_bbox
        self.object_key = object_key
        self.size = size

    def __call__(self, sample, rng=None, *args):
        if rng is None:
            rng = np.random.default_rng()

        object_map = np.array(sample[self.object_key], copy=False)
        h, w = object_map.shape

        unique_objects = np.unique(object_map)
        unique_objects = unique_objects[unique_objects != 0]

        if unique_objects.size > 0:
            random_id = rng.choice(unique_objects, 1)

            y, x = np.where(object_map == random_id)
            ym, xm = np.mean(y), np.mean(x)

            i = int(np.clip(ym-self.size[1]/2, 0, h-self.size[1]))
            j = int(np.clip(xm-self.size[0]/2, 0, w-self.size[0]))

        else:
            i = rng.integers(max(1,h - self.size[1]))
            j = rng.integers(max(1,w - self.size[0]))

        # Capture original size before crop
        if len(self.keys) > 0:
            _orig_size = get_image_size(sample[self.keys[0]])  # (w, h)

        for k in self.keys:
            assert(k in sample)

            if type(sample[k]) == torch.Tensor and len(sample[k].shape) == 2:
                sample[k] = F.crop(sample[k].unsqueeze(0), i, j, self.size[1], self.size[0])[0]
            else:
                sample[k] = F.crop(sample[k], i, j, self.size[1], self.size[0])

        for k in self.keys_bbox:
            assert (k in sample)
            assert isinstance(sample[k], torch.Tensor) or isinstance(sample[k], np.ndarray)

            bi,bj = 0,1
            # ignore first element if odd number of elements (we assume it is ID)
            if sample[k].shape[1] % 2 == 1:
                bi, bj = bi+1,bj+1
                idx = sample[k][:, 0] > 0
            else:
                idx = (sample[k][:, bi] != 0) * (sample[k][:, bj] != 0)

            sample[k][idx, bi] -= j
            sample[k][idx, bj] -= i

        # Record transform for inverse
        _record_transform(sample, 'CropRandomObject', {
            'crop_i': i, 'crop_j': j,
            'crop_h': self.size[1], 'crop_w': self.size[0],
            'orig_w': _orig_size[0] if len(self.keys) > 0 else w,
            'orig_h': _orig_size[1] if len(self.keys) > 0 else h,
        })

        return sample

    @staticmethod
    def inverse(params, images=None, points=None, bboxes=None):
        """Inverse of crop: pad image back to original size, shift points/bboxes.
        
        Args:
            params: dict with crop parameters.
            images: optional list of images (Tensor/ndarray/PIL) to inverse-transform.
            points: optional list of (N, 2) arrays of (x, y) points.
            bboxes: optional list of (N, 4+) arrays of bboxes.
        
        Returns:
            dict with keys 'images', 'points', 'bboxes' for those that were provided.
        """
        result = {}
        i, j = params['crop_i'], params['crop_j']
        orig_h, orig_w = params['orig_h'], params['orig_w']
        crop_h, crop_w = params['crop_h'], params['crop_w']

        pad_left = j
        pad_top = i
        pad_right = orig_w - crop_w - j
        pad_bottom = orig_h - crop_h - i

        if images:
            out_images = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    if len(img.shape) == 2:
                        img = F.pad(img.unsqueeze(0), [pad_left, pad_top, pad_right, pad_bottom], fill=0)[0]
                    else:
                        img = F.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
                elif isinstance(img, np.ndarray):
                    if img.ndim == 2:
                        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
                    else:
                        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
                elif isinstance(img, Image.Image):
                    img = F.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
                out_images.append(img)
            result['images'] = out_images

        if points:
            out_points = []
            for pts in points:
                pts = np.array(pts, dtype=np.float64)
                pts[:, 0] += params['crop_j']  # x offset
                pts[:, 1] += params['crop_i']  # y offset
                out_points.append(pts)
            result['points'] = out_points

        if bboxes:
            out_bboxes = []
            for bbs in bboxes:
                bbs = np.array(bbs, dtype=np.float64)
                if bbs.size > 0:
                    i_col, j_col = 0, 1
                    if bbs.shape[1] % 2 == 1:
                        i_col, j_col = 1, 2
                    bbs[:, i_col] += params['crop_j']
                    bbs[:, j_col] += params['crop_i']
                out_bboxes.append(bbs)
            result['bboxes'] = out_bboxes

        return result

class RandomCrop(T.RandomCrop):

    def __init__(self, keys=[], keys_bbox=[], keys_camK=[], size=100, pad_if_needed=False, fixed_offset=None):
        self.disabled = False

        try:
            if type(size) in [list, tuple]:
                size = [int(s) if isinstance(s, str) else s for s in size]
            else:
                size = int(size) if isinstance(size, str) else size
        except ValueError:
            # disable it if invalid input size !!
            self.disabled = True
            size = 0

        super().__init__(size, pad_if_needed=pad_if_needed)
        self.keys = keys
        self.keys_bbox = keys_bbox
        self.keys_camK = keys_camK
        self.fixed_offset = fixed_offset

    def _pad_if_needed(self, img):

        pad_size = [0,0,0,0] # (top,left,right,bottom)
        if self.pad_if_needed:
            img_size = get_image_size(img)
            # pad the width if needed
            if img_size[0] < self.size[1]:
                pad_w = self.size[1] - img_size[0]
                # pad both sides equally
                pad_size[0] = pad_w // 2
                pad_size[2] = pad_w-pad_size[0]

            # pad the height if needed
            if img_size[1] < self.size[0]:
                pad_h = self.size[0] - img_size[1]

                # pad both sides equally
                pad_size[1] = pad_h // 2
                pad_size[3] = pad_h - pad_size[1]

            if type(img) == torch.Tensor and len(img.shape) == 2:
                img = F.pad(img.unsqueeze(0), pad_size, self.fill, self.padding_mode)[0]
            else:
                img = F.pad(img, pad_size, self.fill, self.padding_mode)

        return img, pad_size
    @staticmethod
    def get_params(img, output_size, fixed_offset=None, rng=None):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """

        if rng is None:
            rng = np.random.default_rng()

        w, h = get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        if fixed_offset is not None:
            i,j = fixed_offset

        else:
            i = rng.integers(max(h - th,1))
            j = rng.integers(max(w - tw,1))


        return i, j, th, tw

    def __call__(self, sample, rng=None, *args):
        if self.disabled:
            return sample

        params = None
        pad_size = None

        # Capture original image size before any padding/cropping
        if len(self.keys) > 0:
            _first_key = self.keys[0]
            _orig_size = get_image_size(sample[_first_key])  # (w, h)
            orig_w, orig_h = _orig_size
        else:
            orig_w, orig_h = 0, 0

        for k in self.keys:
            assert(k in sample)

            if self.pad_if_needed:
                sample[k], pad_size = self._pad_if_needed(sample[k])

            if params is None:
                params = self.get_params(sample[k], self.size, self.fixed_offset, rng)

            if type(sample[k]) == torch.Tensor and len(sample[k].shape) == 2:
                sample[k] = F.crop(sample[k].unsqueeze(0), *params)[0]
            else:
                sample[k] = F.crop(sample[k], *params)

        for k in self.keys_bbox:
            assert (k in sample)
            assert isinstance(sample[k], torch.Tensor) or isinstance(sample[k], np.ndarray)
            assert params

            dx,dy, _,_ = params

            if self.pad_if_needed:
                dx, dy = dx - pad_size[1], dy - pad_size[0]

            i,j = 0,1
            # ignore first element if odd number of elements (we assume it is ID)
            if sample[k].shape[1] % 2 == 1:
                i, j = i+1,j+1
                idx = sample[k][:, 0] > 0
            else:
                idx = (sample[k][:,i] > 0) | (sample[k][:,j] > 0)
            sample[k][idx, i] -= dy
            sample[k][idx, j] -= dx

        for k in self.keys_camK:
            assert (k in sample)
            assert isinstance(sample[k], torch.Tensor) or isinstance(sample[k], np.ndarray)
            assert params

            dx,dy, _,_ = params

            sample[k][0, 2] -= dy
            sample[k][1, 2] -= dx

        # Record transform for inverse
        _record_transform(sample, 'RandomCrop', {
            'crop_i': params[0], 'crop_j': params[1],
            'crop_h': params[2], 'crop_w': params[3],
            'orig_w': orig_w, 'orig_h': orig_h,
            'pad_size': pad_size if self.pad_if_needed else None,
        })

        return sample

    @staticmethod
    def inverse(params, images=None, points=None, bboxes=None):
        """Inverse of crop (and optional padding): pad back then un-pad.
        
        Args:
            params: dict with crop parameters.
            images: optional list of images to inverse-transform.
            points: optional list of (N, 2) arrays of (x, y) points.
            bboxes: optional list of (N, 4+) arrays of bboxes.
        
        Returns:
            dict with keys 'images', 'points', 'bboxes' for those that were provided.
        """
        result = {}
        i, j = params['crop_i'], params['crop_j']
        orig_h, orig_w = params['orig_h'], params['orig_w']
        crop_h, crop_w = params['crop_h'], params['crop_w']
        pad_size = params.get('pad_size')

        if pad_size is not None:
            padded_w = orig_w + pad_size[0] + pad_size[2]
            padded_h = orig_h + pad_size[1] + pad_size[3]
        else:
            padded_w = orig_w
            padded_h = orig_h

        pad_left = j
        pad_top = i
        pad_right = padded_w - crop_w - j
        pad_bottom = padded_h - crop_h - i

        if images:
            out_images = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    if len(img.shape) == 2:
                        img = F.pad(img.unsqueeze(0), [pad_left, pad_top, pad_right, pad_bottom], fill=0)[0]
                    else:
                        img = F.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
                elif isinstance(img, np.ndarray):
                    if img.ndim == 2:
                        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
                    else:
                        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
                elif isinstance(img, Image.Image):
                    img = F.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=0)

                # Then undo the padding if it was applied
                if pad_size is not None:
                    pl, pt, pr, pb = pad_size
                    if isinstance(img, torch.Tensor):
                        if len(img.shape) == 2:
                            img = img[pt:img.shape[0] - pb if pb > 0 else img.shape[0],
                                      pl:img.shape[1] - pr if pr > 0 else img.shape[1]]
                        else:
                            img = img[..., pt:img.shape[-2] - pb if pb > 0 else img.shape[-2],
                                           pl:img.shape[-1] - pr if pr > 0 else img.shape[-1]]
                    elif isinstance(img, np.ndarray):
                        h, w = img.shape[0], img.shape[1]
                        img = img[pt:h - pb if pb > 0 else h, pl:w - pr if pr > 0 else w]
                    elif isinstance(img, Image.Image):
                        w, h = img.size
                        img = img.crop((pl, pt, w - pr if pr > 0 else w, h - pb if pb > 0 else h))
                out_images.append(img)
            result['images'] = out_images

        if points:
            out_points = []
            for pts in points:
                pts = np.array(pts, dtype=np.float64)
                pts[:, 0] += params['crop_j']  # x
                pts[:, 1] += params['crop_i']  # y
                if pad_size is not None:
                    pts[:, 0] -= pad_size[0]  # left pad
                    pts[:, 1] -= pad_size[1]  # top pad
                out_points.append(pts)
            result['points'] = out_points

        if bboxes:
            out_bboxes = []
            for bbs in bboxes:
                bbs = np.array(bbs, dtype=np.float64)
                if bbs.size > 0:
                    i_col, j_col = 0, 1
                    if bbs.shape[1] % 2 == 1:
                        i_col, j_col = 1, 2
                    bbs[:, i_col] += params['crop_j']
                    bbs[:, j_col] += params['crop_i']
                    if pad_size is not None:
                        bbs[:, i_col] -= pad_size[0]
                        bbs[:, j_col] -= pad_size[1]
                out_bboxes.append(bbs)
            result['bboxes'] = out_bboxes

        return result

class FixedCrop(RandomCrop):
    def __init__(self, loc=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loc = loc

    def get_params(self, img, output_size, fixed_offset=None, rng=None):
        i,j = self.loc
        th, tw = output_size

        return i, j, th, tw

class CenterCrop(RandomCrop):
    @staticmethod
    def get_params(img, output_size, fixed_offset=None, rng=None):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """

        if rng is None:
            rng = np.random.default_rng()

        w, h = get_image_size(img)
        th, tw = output_size

        if w == tw and h == th:
            return 0, 0, h, w

        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        return i, j, th, tw

class RandomCustomRotation(object):

    def __init__(self, angles, rate=0.5, resample=False, expand=False, center=None, keys=[], keys_bbox=[], keys_custom_fn=dict()):

        self.keys = keys
        self.keys_bbox = keys_bbox
        self.keys_custom_fn = keys_custom_fn
        self.angles = angles
        self.rate = rate
        self.resample = InterpolationMode.BILINEAR if resample else InterpolationMode.NEAREST
        self.expand = expand
        self.center = center

        if isinstance(self.resample, Iterable):
            assert (len(keys) == len(self.resample))

    @staticmethod
    def get_params(angles, rate, rng=None):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        if rng is None:
            rng = np.random.default_rng()

        if rng.random() < rate:
            return rng.choice(angles, 1)[0]
        else:
            return 0

    def __call__(self, sample, rng=None, *args):

        angle = self.get_params(self.angles, self.rate, rng)

        if angle != 0:
            if len(self.keys) > 0:
                org_im_size = get_image_size(sample[self.keys[0]])

            for idx, k in enumerate(self.keys):

                assert(k in sample)

                resample = self.resample
                if isinstance(resample, Iterable):
                    resample = resample[idx]

                if type(sample[k]) == torch.Tensor and len(sample[k].shape) == 2:
                    sample[k] = F.rotate(sample[k].unsqueeze(0), float(angle), resample,
                                         self.expand, self.center)[0]
                else:
                    sample[k] = F.rotate(sample[k], float(angle), resample,
                                         self.expand, self.center)

            if len(self.keys_bbox) > 0:
                if len(self.keys) > 0:
                    new_im_size = get_image_size(sample[self.keys[0]])
                assert org_im_size
                assert new_im_size
            elif len(self.keys) > 0:
                new_im_size = get_image_size(sample[self.keys[0]])

                old_center = (org_im_size[0]/2,org_im_size[1]/2) if self.center is None else self.center
                new_center = (new_im_size[0]/2,new_im_size[1]/2) if self.expand else old_center

                for k in self.keys_bbox:
                    assert (k in sample)
                    assert isinstance(sample[k],torch.Tensor) or isinstance(sample[k],np.ndarray)

                    i, j = 0, 1
                    # ignore first element if odd number of elements (we assume it is ID)
                    if sample[k].shape[1] % 2 == 1:
                        i, j = i + 1, j + 1
                        idx = sample[k][:, 0] > 0
                    else:
                        idx = (sample[k][:, i] != 0) * (sample[k][:, j] != 0)
                    org_shape = sample[k][idx, i:].shape
                    points = sample[k][idx, i:].reshape(-1,2)

                    # position relative to the old center
                    points[:, 0] -= old_center[0]
                    points[:, 1] -= old_center[1]

                    # rotate each point
                    angle_rad = angle*np.pi/180
                    rot_mat = np.array([[np.cos(-angle_rad), -np.sin(-angle_rad)],
                                        [np.sin(-angle_rad), np.cos(-angle_rad)]], dtype=np.float32)

                    points = np.matmul(rot_mat, points.T).T

                    # position relative to the new center
                    points[:, 0] += new_center[0]
                    points[:, 1] += new_center[1]

                    sample[k][idx, i:] = points.reshape(org_shape)

            if self.keys_custom_fn is not None:
                for k,fn in self.keys_custom_fn.items():
                    sample[k] = fn(sample[k], angle)

        if angle != 0:
            _record_transform(sample, 'RandomCustomRotation', {
                'angle': float(angle),
                'expand': self.expand,
                'center': self.center,
                'org_im_size': org_im_size if len(self.keys) > 0 else None,
                'new_im_size': new_im_size if len(self.keys) > 0 else None,
            })

        return sample

    @staticmethod
    def inverse(params, images=None, points=None, bboxes=None):
        """Inverse of rotation: rotate by -angle, crop/pad to original size.
        
        Args:
            params: dict with rotation parameters.
            images: optional list of images to inverse-transform.
            points: optional list of (N, 2) arrays of (x, y) points.
            bboxes: optional list of (N, 4+) arrays of bboxes.
        
        Returns:
            dict with keys 'images', 'points', 'bboxes' for those that were provided.
        """
        result = {}
        angle = params['angle']
        expand = params['expand']
        center = params['center']
        org_im_size = params['org_im_size']
        new_im_size = params['new_im_size']

        if images:
            out_images = []
            for img in images:
                if isinstance(img, torch.Tensor) and len(img.shape) == 2:
                    img = F.rotate(img.unsqueeze(0), -angle, InterpolationMode.BILINEAR, expand, center)[0]
                else:
                    img = F.rotate(img, -angle, InterpolationMode.BILINEAR, expand, center)

                if expand and org_im_size is not None:
                    cur_w, cur_h = get_image_size(img)
                    target_w, target_h = org_im_size
                    left = (cur_w - target_w) // 2
                    top = (cur_h - target_h) // 2
                    if isinstance(img, torch.Tensor) and len(img.shape) == 2:
                        img = F.crop(img.unsqueeze(0), top, left, target_h, target_w)[0]
                    else:
                        img = F.crop(img, top, left, target_h, target_w)
                out_images.append(img)
            result['images'] = out_images

        if points:
            out_points = []
            if org_im_size is not None:
                new_center = (new_im_size[0] / 2, new_im_size[1] / 2) if expand else \
                             ((org_im_size[0] / 2, org_im_size[1] / 2) if center is None else center)
                old_center = (org_im_size[0] / 2, org_im_size[1] / 2) if center is None else center
                angle_rad = -angle * np.pi / 180
                rot_mat = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad), np.cos(angle_rad)]], dtype=np.float64)
            for pts in points:
                pts = np.array(pts, dtype=np.float64)
                if org_im_size is not None:
                    pts[:, 0] -= new_center[0]
                    pts[:, 1] -= new_center[1]
                    pts = np.matmul(rot_mat, pts.T).T
                    pts[:, 0] += old_center[0]
                    pts[:, 1] += old_center[1]
                out_points.append(pts)
            result['points'] = out_points

        if bboxes:
            out_bboxes = []
            if org_im_size is not None:
                new_center = (new_im_size[0] / 2, new_im_size[1] / 2) if expand else \
                             ((org_im_size[0] / 2, org_im_size[1] / 2) if center is None else center)
                old_center = (org_im_size[0] / 2, org_im_size[1] / 2) if center is None else center
                angle_rad = -angle * np.pi / 180
                rot_mat = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad), np.cos(angle_rad)]], dtype=np.float64)
            for bbs in bboxes:
                bbs = np.array(bbs, dtype=np.float64)
                if bbs.size > 0 and org_im_size is not None:
                    i_col, j_col = 0, 1
                    if bbs.shape[1] % 2 == 1:
                        i_col, j_col = 1, 2
                    coords = bbs[:, i_col:].reshape(-1, 2)
                    coords[:, 0] -= new_center[0]
                    coords[:, 1] -= new_center[1]
                    coords = np.matmul(rot_mat, coords.T).T
                    coords[:, 0] += old_center[0]
                    coords[:, 1] += old_center[1]
                    bbs[:, i_col:] = coords.reshape(bbs[:, i_col:].shape)
                out_bboxes.append(bbs)
            result['bboxes'] = out_bboxes

        return result

    def __repr__(self):
        format_string = self.__class__.__name__ + '(angles={0}'.format(self.angles)
        format_string += ', rate={0}'.format(self.rate)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string



class RandomRotation(T.RandomRotation):

    def __init__(self, keys=[], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.keys = keys

        if isinstance(self.resample, Iterable):
            assert(len(keys) == len(self.resample))

    @staticmethod
    def get_params(degrees, rng=None):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        if rng is None:
            rng = np.random.default_rng()

        angle = rng.uniform(degrees[0], degrees[1])

        return angle
    def __call__(self, sample, rng=None, *args):

        angle = self.get_params(self.degrees, rng)

        if len(self.keys) > 0:
            org_im_size = get_image_size(sample[self.keys[0]])

        for idx, k in enumerate(self.keys):

            assert(k in sample)

            resample = self.resample
            if isinstance(resample, Iterable):
                resample = resample[idx]

            if type(sample[k]) == torch.Tensor and len(sample[k].shape) == 2:
                sample[k] = F.rotate(sample[k].unsqueeze(0), angle, resample, self.expand, self.center)[0]
            else:
                sample[k] = F.rotate(sample[k], angle, resample, self.expand, self.center)

        if len(self.keys) > 0:
            new_im_size = get_image_size(sample[self.keys[0]])
        else:
            new_im_size = None
            org_im_size = None

        if angle != 0:
            _record_transform(sample, 'RandomRotation', {
                'angle': float(angle),
                'expand': self.expand,
                'center': self.center,
                'org_im_size': org_im_size,
                'new_im_size': new_im_size,
            })

        return sample

    @staticmethod
    def inverse(params, images=None, points=None, bboxes=None):
        """Inverse of rotation: delegates to RandomCustomRotation.inverse()."""
        return RandomCustomRotation.inverse(params, images=images, points=points, bboxes=bboxes)


class RandomHorizontalFlip(T.RandomHorizontalFlip):

    def __init__(self, keys=[], keys_bbox=[], keys_custom_fn=dict(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = keys
        self.keys_bbox = keys_bbox
        self.keys_custom_fn = keys_custom_fn

    def __call__(self, sample, rng=None, *args):
        if rng is None:
            rng = np.random.default_rng()

        if rng.random() < self.p:
            for k in self.keys:
                assert (k in sample)
                if type(sample[k]) == torch.Tensor and len(sample[k].shape) == 2:
                    sample[k] = F.hflip(sample[k].unsqueeze(0))[0]
                else:
                    sample[k] = F.hflip(sample[k])

            if len(self.keys_bbox) > 0:
                if len(self.keys) > 0:
                    im_size = get_image_size(sample[self.keys[0]])
                assert im_size
                for k in self.keys_bbox:
                    assert (k in sample)
                    assert isinstance(sample[k], torch.Tensor) or isinstance(sample[k], np.ndarray)

                    i, j = 0, 1
                    # ignore first element if odd number of elements (we assume it is ID)
                    if sample[k].shape[1] % 2 == 1:
                        i, j = i + 1, j + 1
                        idx = sample[k][:, 0] > 0
                    else:
                        idx = (sample[k][:, i] != 0) * (sample[k][:, j] != 0)
                    sample[k][idx, i] = im_size[0] - sample[k][idx, i]
                    if sample[k].shape[1] > i+2:
                        sample[k][idx, i] -= sample[k][idx, i+2]

            if self.keys_custom_fn is not None:
                for k,fn in self.keys_custom_fn.items():
                    sample[k] = fn(sample[k])

            # Record that flip was applied
            if len(self.keys) > 0:
                im_size = get_image_size(sample[self.keys[0]])
            _record_transform(sample, 'RandomHorizontalFlip', {
                'im_w': im_size[0], 'im_h': im_size[1],
            })

        return sample

    @staticmethod
    def inverse(params, images=None, points=None, bboxes=None):
        """Inverse of horizontal flip.
        
        Args:
            params: dict with flip parameters.
            images: optional list of images to inverse-transform.
            points: optional list of (N, 2) arrays of (x, y) points.
            bboxes: optional list of (N, 4+) arrays of bboxes.
        
        Returns:
            dict with keys 'images', 'points', 'bboxes' for those that were provided.
        """
        result = {}

        if images:
            out_images = []
            for img in images:
                if isinstance(img, torch.Tensor) and len(img.shape) == 2:
                    img = F.hflip(img.unsqueeze(0))[0]
                else:
                    img = F.hflip(img)
                out_images.append(img)
            result['images'] = out_images

        if points:
            out_points = []
            for pts in points:
                pts = np.array(pts, dtype=np.float64)
                pts[:, 0] = params['im_w'] - pts[:, 0]
                out_points.append(pts)
            result['points'] = out_points

        if bboxes:
            out_bboxes = []
            for bbs in bboxes:
                bbs = np.array(bbs, dtype=np.float64)
                if bbs.size > 0:
                    i_col = 0
                    if bbs.shape[1] % 2 == 1:
                        i_col = 1
                    bbs[:, i_col] = params['im_w'] - bbs[:, i_col]
                out_bboxes.append(bbs)
            result['bboxes'] = out_bboxes

        return result

class RandomVerticalFlip(T.RandomVerticalFlip):

    def __init__(self, keys=[], keys_bbox=[], keys_custom_fn=dict(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = keys
        self.keys_bbox = keys_bbox
        self.keys_custom_fn = keys_custom_fn

    def __call__(self, sample, rng=None, *args):
        if rng is None:
            rng = np.random.default_rng()

        if rng.random() < self.p:
            for k in self.keys:
                assert (k in sample)
                if type(sample[k]) == torch.Tensor and len(sample[k].shape) == 2:
                    sample[k] = F.vflip(sample[k].unsqueeze(0))[0]
                else:
                    sample[k] = F.vflip(sample[k])

            if len(self.keys_bbox) > 0:
                if len(self.keys) > 0:
                    im_size = get_image_size(sample[self.keys[0]])
                assert im_size
                for k in self.keys_bbox:
                    assert (k in sample)
                    assert isinstance(sample[k], torch.Tensor) or isinstance(sample[k], np.ndarray)

                    i, j = 0, 1
                    # ignore first element if odd number of elements (we assume it is ID)
                    if sample[k].shape[1] % 2 == 1:
                        i, j = i + 1, j + 1
                        idx = sample[k][:, 0] > 0
                    else:
                        idx = (sample[k][:, i] != 0) * (sample[k][:, j] != 0)
                    sample[k][idx, j] = im_size[1] - sample[k][idx, j]
                    if sample[k].shape[1] > j+2:
                        sample[k][idx, j] -= sample[k][idx, j+2]

            if self.keys_custom_fn is not None:
                for k,fn in self.keys_custom_fn.items():
                    sample[k] = fn(sample[k])

            # Record that flip was applied
            if len(self.keys) > 0:
                im_size = get_image_size(sample[self.keys[0]])
            _record_transform(sample, 'RandomVerticalFlip', {
                'im_w': im_size[0], 'im_h': im_size[1],
            })

        return sample

    @staticmethod
    def inverse(params, images=None, points=None, bboxes=None):
        """Inverse of vertical flip.
        
        Args:
            params: dict with flip parameters.
            images: optional list of images to inverse-transform.
            points: optional list of (N, 2) arrays of (x, y) points.
            bboxes: optional list of (N, 4+) arrays of bboxes.
        
        Returns:
            dict with keys 'images', 'points', 'bboxes' for those that were provided.
        """
        result = {}

        if images:
            out_images = []
            for img in images:
                if isinstance(img, torch.Tensor) and len(img.shape) == 2:
                    img = F.vflip(img.unsqueeze(0))[0]
                else:
                    img = F.vflip(img)
                out_images.append(img)
            result['images'] = out_images

        if points:
            out_points = []
            for pts in points:
                pts = np.array(pts, dtype=np.float64)
                pts[:, 1] = params['im_h'] - pts[:, 1]
                out_points.append(pts)
            result['points'] = out_points

        if bboxes:
            out_bboxes = []
            for bbs in bboxes:
                bbs = np.array(bbs, dtype=np.float64)
                if bbs.size > 0:
                    j_col = 1
                    if bbs.shape[1] % 2 == 1:
                        j_col = 2
                    bbs[:, j_col] = params['im_h'] - bbs[:, j_col]
                out_bboxes.append(bbs)
            result['bboxes'] = out_bboxes

        return result

class RandomResize(object):

    def __init__(self, scale_range, keys=[], keys_bbox=[], keys_camK=[], keys_custom_fn=dict(), interpolation=Image.BILINEAR, *args, **kwargs):
        self.scale_range = scale_range
        self.keys = keys
        self.keys_bbox = keys_bbox
        self.keys_camK = keys_camK
        self.keys_custom_fn = keys_custom_fn
        self.interpolation = interpolation

        if isinstance(self.interpolation, Iterable):
            assert(len(keys) == len(self.interpolation))

    @staticmethod
    def get_params(factor_range, rng=None):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        if rng is None:
            rng = np.random.default_rng()

        resize_factor = rng.uniform(factor_range[0], factor_range[1])

        return resize_factor

    def __call__(self, sample, rng=None, *args):

        resize_factor = self.get_params(self.scale_range, rng)

        if len(self.keys) > 0:
            org_im_size = get_image_size(sample[self.keys[0]])

        for idx, k in enumerate(self.keys):

            assert(k in sample)

            interpolation = self.interpolation
            if isinstance(interpolation, Iterable):
                interpolation = interpolation[idx]

            new_size = int(get_image_size(sample[k])[1]*resize_factor), int(get_image_size(sample[k])[0]*resize_factor)
            if type(sample[k]) == torch.Tensor and len(sample[k].shape) == 2:
                sample[k] = F.resize(sample[k].unsqueeze(0), new_size, interpolation, antialias=None)[0]
            else:
                sample[k] = F.resize(sample[k], new_size, interpolation, antialias=None)

        if len(self.keys_bbox) > 0:
            if len(self.keys) > 0:
                new_im_size = get_image_size(sample[self.keys[0]])
            assert org_im_size
            assert new_im_size
            real_resize_factor = new_im_size[0] / org_im_size[0], new_im_size[1] / org_im_size[1]

            for k in self.keys_bbox:
                assert (k in sample)
                assert isinstance(sample[k], torch.Tensor) or isinstance(sample[k], np.ndarray)

                i, j = 0, 1
                # ignore first element if odd number of elements (we assume it is ID)
                if sample[k].shape[1] % 2 == 1:
                    i, j = i + 1, j + 1
                    idx = sample[k][:, 0] > 0
                else:
                    idx = (sample[k][:, i] != 0) * (sample[k][:, j] != 0)
                sample[k][idx, i] *= real_resize_factor[0]
                sample[k][idx, j] *= real_resize_factor[1]
                if sample[k].shape[1] > j+2:
                    sample[k][idx, i+2] *= real_resize_factor[0]
                    sample[k][idx, j+2] *= real_resize_factor[1]

        if self.keys_custom_fn is not None:
            if len(self.keys) > 0:
                new_im_size = get_image_size(sample[self.keys[0]])
            assert org_im_size
            assert new_im_size
            for k,fn in self.keys_custom_fn.items():
                sample[k] = fn(sample[k], org_im_size, new_im_size)

        # adjustment of camera calibration matrix
        if len(self.keys_camK) > 0:
            if len(self.keys) > 0:
                new_im_size = get_image_size(sample[self.keys[0]])
            assert org_im_size
            assert new_im_size
            real_resize_factor = new_im_size[0] / org_im_size[0], new_im_size[1] / org_im_size[1]

            for k in self.keys_camK:
                if k not in sample:
                    continue

                sample[k][0,:] *= real_resize_factor[0]
                sample[k][1,:] *= real_resize_factor[1]

        # Record transform for inverse
        if len(self.keys) > 0:
            new_im_size = get_image_size(sample[self.keys[0]])
            _record_transform(sample, 'RandomResize', {
                'org_im_size': org_im_size,  # (w, h)
                'new_im_size': new_im_size,  # (w, h)
            })

        return sample

    @staticmethod
    def inverse(params, images=None, points=None, bboxes=None):
        """Inverse of resize: scale back to original dimensions.
        
        Args:
            params: dict with resize parameters.
            images: optional list of images to inverse-transform.
            points: optional list of (N, 2) arrays of (x, y) points.
            bboxes: optional list of (N, 4+) arrays of bboxes.
        
        Returns:
            dict with keys 'images', 'points', 'bboxes' for those that were provided.
        """
        result = {}
        org_w, org_h = params['org_im_size']
        new_w, new_h = params['new_im_size']
        target_size = (org_h, org_w)  # F.resize expects (h, w)
        scale_x = org_w / new_w
        scale_y = org_h / new_h

        if images:
            out_images = []
            for img in images:
                if isinstance(img, torch.Tensor) and len(img.shape) == 2:
                    img = F.resize(img.unsqueeze(0), target_size, InterpolationMode.BILINEAR, antialias=None)[0]
                else:
                    img = F.resize(img, target_size, InterpolationMode.BILINEAR, antialias=None)
                out_images.append(img)
            result['images'] = out_images

        if points:
            out_points = []
            for pts in points:
                pts = np.array(pts, dtype=np.float64)
                pts[:, 0] *= scale_x
                pts[:, 1] *= scale_y
                out_points.append(pts)
            result['points'] = out_points

        if bboxes:
            out_bboxes = []
            for bbs in bboxes:
                bbs = np.array(bbs, dtype=np.float64)
                if bbs.size > 0:
                    i_col, j_col = 0, 1
                    if bbs.shape[1] % 2 == 1:
                        i_col, j_col = 1, 2
                    bbs[:, i_col] *= scale_x
                    bbs[:, j_col] *= scale_y
                    if bbs.shape[1] > j_col + 2:
                        bbs[:, i_col + 2] *= scale_x
                        bbs[:, j_col + 2] *= scale_y
                out_bboxes.append(bbs)
            result['bboxes'] = out_bboxes

        return result


class Resize(T.Resize):

    def __init__(self, keys=[], keys_bbox=[], keys_camK=[], keys_custom_fn=dict(), *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.keys = keys
        self.keys_bbox = keys_bbox
        self.keys_camK = keys_camK
        self.keys_custom_fn = keys_custom_fn

        if isinstance(self.interpolation, Iterable):
            assert(len(keys) == len(self.interpolation))

    def __call__(self, sample, *args):

        if len(self.keys) > 0:
            org_im_size = get_image_size(sample[self.keys[0]])

        for idx, k in enumerate(self.keys):

            assert(k in sample)

            interpolation = self.interpolation
            if isinstance(interpolation, Iterable):
                interpolation = interpolation[idx]

            if type(sample[k]) == torch.Tensor and len(sample[k].shape) == 2:
                sample[k] = F.resize(sample[k].unsqueeze(0), self.size, interpolation, antialias=None)[0]
            else:
                sample[k] = F.resize(sample[k], self.size, interpolation, antialias=None)

        if self.keys_custom_fn is not None:
            for k,fn in self.keys_custom_fn.items():
                sample[k] = fn(sample[k], org_im_size, self.size)

        if len(self.keys_bbox) > 0:
            assert org_im_size
            resize_factor = self.size[1]/org_im_size[0], self.size[0]/org_im_size[1]

            for k in self.keys_bbox:
                assert (k in sample)
                assert isinstance(sample[k], torch.Tensor) or isinstance(sample[k], np.ndarray)

                i, j = 0, 1
                # ignore first element if odd number of elements (we assume it is ID)
                if sample[k].shape[1] % 2 == 1:
                    i, j = i + 1, j + 1
                    idx = sample[k][:, 0] > 0
                else:
                    idx = (sample[k][:, i] != 0) * (sample[k][:, j] != 0)
                sample[k][idx, i] *= resize_factor[0]
                sample[k][idx, j] *= resize_factor[1]
                if sample[k].shape[1] > j+2:
                    sample[k][idx, i+2] *= resize_factor[0]
                    sample[k][idx, j+2] *= resize_factor[1]

        # adjustment of camera calibration matrix
        if len(self.keys_camK) > 0:

            resize_factor = self.size[1]/org_im_size[0], self.size[0]/org_im_size[1]

            for k in self.keys_camK:
                if k not in sample:
                    continue

                sample[k][0,:] *= resize_factor[0]
                sample[k][1,:] *= resize_factor[1]

        # Record transform for inverse
        if len(self.keys) > 0:
            new_im_size = get_image_size(sample[self.keys[0]])
            _record_transform(sample, 'Resize', {
                'org_im_size': org_im_size,  # (w, h)
                'new_im_size': new_im_size,  # (w, h)
                'target_size': tuple(self.size),
            })

        return sample

    @staticmethod
    def inverse(params, images=None, points=None, bboxes=None):
        """Inverse of resize: scale back to original dimensions.
        
        Args:
            params: dict with resize parameters.
            images: optional list of images to inverse-transform.
            points: optional list of (N, 2) arrays of (x, y) points.
            bboxes: optional list of (N, 4+) arrays of bboxes.
        
        Returns:
            dict with keys 'images', 'points', 'bboxes' for those that were provided.
        """
        result = {}
        org_w, org_h = params['org_im_size']
        new_w, new_h = params['new_im_size']
        target_size = (org_h, org_w)
        scale_x = org_w / new_w
        scale_y = org_h / new_h

        if images:
            out_images = []
            for img in images:
                if isinstance(img, torch.Tensor) and len(img.shape) == 2:
                    img = F.resize(img.unsqueeze(0), target_size, InterpolationMode.BILINEAR, antialias=None)[0]
                else:
                    img = F.resize(img, target_size, InterpolationMode.BILINEAR, antialias=None)
                out_images.append(img)
            result['images'] = out_images

        if points:
            out_points = []
            for pts in points:
                pts = np.array(pts, dtype=np.float64)
                pts[:, 0] *= scale_x
                pts[:, 1] *= scale_y
                out_points.append(pts)
            result['points'] = out_points

        if bboxes:
            out_bboxes = []
            for bbs in bboxes:
                bbs = np.array(bbs, dtype=np.float64)
                if bbs.size > 0:
                    i_col, j_col = 0, 1
                    if bbs.shape[1] % 2 == 1:
                        i_col, j_col = 1, 2
                    bbs[:, i_col] *= scale_x
                    bbs[:, j_col] *= scale_y
                    if bbs.shape[1] > j_col + 2:
                        bbs[:, i_col + 2] *= scale_x
                        bbs[:, j_col + 2] *= scale_y
                out_bboxes.append(bbs)
            result['bboxes'] = out_bboxes

        return result


class ColorJitter(T.ColorJitter):

    def __init__(self, keys=[], p=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = keys
        self.p = p

    @staticmethod
    def get_params(brightness, contrast, saturation, hue, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        fn_idx = rng.permutation(4)

        b = None if brightness is None else float(rng.uniform(brightness[0], brightness[1], size=1))
        c = None if contrast is None else float(rng.uniform(contrast[0], contrast[1], size=1))
        s = None if saturation is None else float(rng.uniform(saturation[0], saturation[1], size=1))
        h = None if hue is None else float(rng.uniform(hue[0], hue[1], size=1))

        return fn_idx, b, c, s, h

    def __call__(self, sample, rng=None, *args):
        if rng is None:
            rng = np.random.default_rng()

        if rng.random() < self.p:
            for idx, k in enumerate(self.keys):

                assert(k in sample)

                fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
                    self.get_params(self.brightness, self.contrast, self.saturation, self.hue, rng)

                for fn_id in fn_idx:
                    if type(sample[k]) == torch.Tensor and len(sample[k].shape) == 2:
                        sample[k] = sample[k].unsqueeze(0)

                    if fn_id == 0 and brightness_factor is not None:
                        sample[k] = F.adjust_brightness(sample[k], brightness_factor)
                    elif fn_id == 1 and contrast_factor is not None:
                        sample[k] = F.adjust_contrast(sample[k], contrast_factor)
                    elif fn_id == 2 and saturation_factor is not None:
                        sample[k] = F.adjust_saturation(sample[k], saturation_factor)
                    elif fn_id == 3 and hue_factor is not None:
                        sample[k] = F.adjust_hue(sample[k], hue_factor)

                    if type(sample[k]) == torch.Tensor and len(sample[k].shape) == 2:
                        sample[k] = sample[k][0]
        return sample


class RandomGaussianNoise(object):

    def __init__(self, keys=[], p=0, mean=0., std=1., **kwargs):
        self.keys = keys
        self.p = p
        self.mean = mean
        self.std = std


    def __call__(self, sample, rng=None, *args):
        if rng is None:
            rng = np.random.default_rng()

        if rng.random() < self.p:
            noise = None
            for idx, k in enumerate(self.keys):
                assert(k in sample)

                if noise is  None:
                    mean = rng.uniform(self.mean[0], self.mean[1]) if type(self.mean) in [list,tuple] else self.mean
                    std = rng.uniform(self.std[0], self.std[1]) if type(self.std) in [list,tuple] else self.std

                    noise = np.random.normal(mean, std, sample[k].shape).astype(np.float32)

                sample[k] = sample[k] + noise

        return sample

class Padding(object):

    IGNORE_PADDING_EDGE = 64
    IGNORE_PADDING_SYMMETRIC = 128

    def __init__(self, keys=[], keys_bbox=[], borders=(0,0,0,0), pad_to_size_factor=0, fill=0, padding_mode="constant", mark_key_with_flag=None):

        self.borders = borders # (left,top,right,bottom)
        self.pad_to_size_factor = pad_to_size_factor
        self.fill = fill
        self.padding_mode = padding_mode
        self.keys = keys
        self.keys_bbox = keys_bbox
        self.mark_with_flag_key = mark_key_with_flag
        self.mark_with_flag = False
        if mark_key_with_flag:
            if padding_mode == "edge":
                self.mark_with_flag = self.IGNORE_PADDING_EDGE
            elif padding_mode == "symmetric":
                self.mark_with_flag = self.IGNORE_PADDING_SYMMETRIC

    def get_pad_size(self, img):

        pad_size = list(self.borders)

        if self.pad_to_size_factor > 0:
            img_size = get_image_size(img)
            # pad the width if needed
            pad_w = (img_size[0] + pad_size[0] + pad_size[2]) % self.pad_to_size_factor
            if pad_w > 0:
                pad_w = self.pad_to_size_factor - pad_w
                # pad both sides equally
                pad_size[0] += pad_w // 2
                pad_size[2] += pad_w - (pad_w // 2)

            # pad the height if needed
            pad_h = (img_size[1] + pad_size[1] + pad_size[3]) % self.pad_to_size_factor
            if pad_h > 0:
                pad_h = self.pad_to_size_factor - pad_h
                # pad both sides equally
                pad_size[1] += pad_h // 2
                pad_size[3] += pad_h - pad_h // 2

        return pad_size

    def __call__(self, sample, rng=None, *args):
        pad_size = None

        for k in self.keys:
            assert(k in sample)

            pad_size_ = self.get_pad_size(sample[k])

            if pad_size is None:
                pad_size = pad_size_
                assert pad_size_ == pad_size

            if type(sample[k]) == torch.Tensor and len(sample[k].shape) == 2:
                sample[k] = F.pad(sample[k].unsqueeze(0), pad_size, self.fill, self.padding_mode)[0]
            else:
                sample[k] = F.pad(sample[k], pad_size, self.fill, self.padding_mode)

            if self.mark_with_flag and k == self.mark_with_flag_key:
                l, t, r, b = pad_size
                # ignore = np.array(sample[k])
                # borders = np.zeros_like(ignore)
                # borders[t:borders.shape[0]-b,l:borders.shape[1]-r] = 1
                #
                # ignore |= (1 - borders) * self.mark_with_flag
                # sample[k] = Image.fromarray(ignore)
                # convert to Tensor first
                ignore = torch.tensor(np.array(sample[k])) if not isinstance(sample[k], torch.Tensor) else sample[k]
                borders = torch.zeros_like(sample[k] if isinstance(sample[k], torch.Tensor) else torch.from_numpy(np.array(sample[k])))
                borders[...,t:borders.shape[-2]-b,l:borders.shape[-1]-r] = 1

                ignore |= (1 - borders) * self.mark_with_flag

                # convert back to original type
                if isinstance(sample[k], np.ndarray):
                    ignore = ignore.numpy()
                elif isinstance(sample[k], Image.Image):
                    ignore = Image.fromarray(ignore.numpy())

                sample[k] = ignore

        for k in self.keys_bbox:
            assert (k in sample)
            assert isinstance(sample[k], torch.Tensor) or isinstance(sample[k], np.ndarray)
            assert pad_size

            i,j = 0,1
            # ignore first element if odd number of elements (we assume it is ID)
            if sample[k].shape[1] % 2 == 1:
                i, j = i+1,j+1
                idx = sample[k][:, 0] > 0
            else:
                idx = (sample[k][:,i] > 0) | (sample[k][:,j] > 0)
            sample[k][idx, i] += pad_size[0]
            sample[k][idx, j] += pad_size[1]

        if pad_size is not None:
            sample['Padding'] = np.array(pad_size).reshape(-1,4)

            _record_transform(sample, 'Padding', {
                'pad_left': pad_size[0],
                'pad_top': pad_size[1],
                'pad_right': pad_size[2],
                'pad_bottom': pad_size[3],
            })

        return sample

    @staticmethod
    def inverse(params, images=None, points=None, bboxes=None):
        """Inverse of padding: crop away the padded borders.
        
        Args:
            params: dict with padding parameters.
            images: optional list of images to inverse-transform.
            points: optional list of (N, 2) arrays of (x, y) points.
            bboxes: optional list of (N, 4+) arrays of bboxes.
        
        Returns:
            dict with keys 'images', 'points', 'bboxes' for those that were provided.
        """
        result = {}
        pl = params['pad_left']
        pt = params['pad_top']
        pr = params['pad_right']
        pb = params['pad_bottom']

        if images:
            out_images = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    if len(img.shape) == 2:
                        h, w = img.shape
                        img = img[pt:h - pb if pb > 0 else h, pl:w - pr if pr > 0 else w]
                    else:
                        h, w = img.shape[-2], img.shape[-1]
                        img = img[..., pt:h - pb if pb > 0 else h, pl:w - pr if pr > 0 else w]
                elif isinstance(img, np.ndarray):
                    h, w = img.shape[0], img.shape[1]
                    img = img[pt:h - pb if pb > 0 else h, pl:w - pr if pr > 0 else w]
                elif isinstance(img, Image.Image):
                    w, h = img.size
                    img = img.crop((pl, pt, w - pr if pr > 0 else w, h - pb if pb > 0 else h))
                out_images.append(img)
            result['images'] = out_images

        if points:
            out_points = []
            for pts in points:
                pts = np.array(pts, dtype=np.float64)
                pts[:, 0] -= pl
                pts[:, 1] -= pt
                out_points.append(pts)
            result['points'] = out_points

        if bboxes:
            out_bboxes = []
            for bbs in bboxes:
                bbs = np.array(bbs, dtype=np.float64)
                if bbs.size > 0:
                    i_col, j_col = 0, 1
                    if bbs.shape[1] % 2 == 1:
                        i_col, j_col = 1, 2
                    bbs[:, i_col] -= pl
                    bbs[:, j_col] -= pt
                out_bboxes.append(bbs)
            result['bboxes'] = out_bboxes

        return result

class ToTensor(object):

    def __init__(self, keys=[], type="float"):

        if isinstance(type, Iterable):
            assert(len(keys) == len(type))

        self.keys = keys
        self.type = type

    def __call__(self, sample, *args):

        for idx, k in enumerate(self.keys):

            assert(k in sample)

            t = self.type
            if isinstance(t, Iterable):
                t = t[idx]

            if isinstance(sample[k],Image.Image) or isinstance(sample[k],np.ndarray):
                sample[k] = F.to_tensor(sample[k])

                if t == torch.ByteTensor:
                    sample[k] = sample[k]*255

            elif not isinstance(sample[k],torch.Tensor):
                raise Exception("Invalid data format: must be PIL Image, numpy array or pytorch Tensor")

            sample[k] = sample[k].type(t)

        return sample

from scipy.ndimage.filters import gaussian_filter

class RandomGaussianBlur(object):

    def __init__(self, rate, sigma, keys=[]):

        self.rate = rate
        self.sigma_interval = sigma
        self.keys = keys

    @staticmethod
    def get_params(rate, sigma, rng=None):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        if rng is None:
            rng = np.random.default_rng()

        if rng.random() < rate:
            return rng.random()*(sigma[1]-sigma[0]) + sigma[0]
        else:
            return 0

    def __call__(self, sample, rng=None, *args):

        sigma = self.get_params(self.rate, self.sigma_interval, rng)

        if sigma > 0:
            for idx, k in enumerate(self.keys):
                assert (k in sample)
                #sample[k] = gaussian_filter(sample[k], sigma)
                sample[k] = F.gaussian_blur(sample[k], 2*round(3*sigma) + 1, sigma)

        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += ', rate={0}'.format(self.rate)
        format_string += ', sigma_interval=[{0},{1]]'.format(self.sigma_interval[0],self.sigma_interval[1])
        format_string += ')'
        return format_string


class Normalize(object):

    def __init__(self, keys=[], mean=[], std=[], shape=[]):

        if isinstance(type, Iterable):
            assert(len(keys) == len(mean))

        if isinstance(type, Iterable):
            assert(len(keys) == len(std))

        if isinstance(type, Iterable):
            assert(len(keys) == len(shape))


        self.keys = keys
        self.mean = mean
        self.std = std
        self.shape = shape

    def __call__(self, sample, *args):

        for idx, k in enumerate(self.keys):

            assert(k in sample)

            m = self.mean
            if isinstance(m, Iterable):
                m = m[idx]

            s = self.std
            if isinstance(s, Iterable):
                s = s[idx]

            sh = self.shape
            if isinstance(sh, Iterable):
                sh = sh[idx]

            sample[k] = (sample[k]  - m.reshape(sh) ) / s.reshape(sh)

        return sample

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, *args):
        for t in self.transforms:
            img = t(img, *args)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class Background(object):

    def __init__(self, bg_dir, bg_images, keys=('image',), p=0.5, mask_key='label', img_ext='.jpg'):
        self.keys = keys
        self.mask_key = mask_key
        self.images = []
        with open(bg_images) as f:
            lines = f.readlines()
            for l in lines:
                s = l.strip().split(' ')
                ok = (int(s[-1])==1) if len(s) > 1 else True
                if ok:
                    fn = os.path.join(bg_dir,f"{s[0]}{img_ext}")
                    self.images.append(fn)

        self.p = p

    def __call__(self, sample, rng=None, *args):

        augment = np.random.random()<self.p

        if not augment:
            return sample

        bg_fn = np.random.choice(self.images, 1)[0]

        label = sample[self.mask_key]
        if isinstance(label,torch.Tensor):
            label = label.permute((1, 2, 0)).numpy()

        bg = None

        for key in self.keys:
            to_output_format = lambda x: x
            im = sample[key]
            if isinstance(im,torch.Tensor):
                to_output_format = F.to_tensor
                if im.type() == torch.ByteTensor:
                    im = im/255
                    to_output_format = lambda x: (F.to_tensor(x)*255).type(torch.ByteTensor)

                im = (im*255).permute([1, 2, 0]).numpy()

            elif isinstance(im,Image.Image):
                to_output_format = Image.fromarray

            im = np.array(im)

            im_shape = im.shape[:2][::-1]

            # get bg
            if bg is None:
                bg = cv2.imread(bg_fn)
                bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
                bg = cv2.resize(bg, im_shape)
                label = np.dstack((label,label,label))
                field3d_mask = np.broadcast_to(label > 0, bg.shape)

            final = bg.copy()
            final[field3d_mask]=im[field3d_mask]
            sample[key] = to_output_format(final)

        return sample

def get_transform(transforms):
    transform_list = []

    for tr in transforms:
        name = tr['name']
        opts = tr['opts']
        transform_list.append(globals()[name](**opts))

    return Compose(transform_list)


def _record_transform(sample, name, params):
    """Record a transform and its parameters into sample['transforms'].
    
    Stored as a JSON string so that variable_len_collate (which cannot handle
    nested dicts containing None) treats it as a plain string and passes it
    through untouched.
    
    Each entry in the JSON list contains:
        - 'name': class name of the transform
        - 'params': dict of parameters needed to compute the inverse
    """
    # Decode existing history
    history = []
    if 'transforms' in sample and sample['transforms']:
        assert isinstance(sample['transforms'], str), f"Expected 'transforms' to be a JSON string."

        history = json.loads(sample['transforms'])


    history.append({'name': name, 'params': params})

    # Store as JSON string (collation-safe)
    sample['transforms'] = json.dumps(history)


# Registry mapping transform name -> class (populated at module end)

# Add inverse method for FixedCrop and CenterCrop (they inherit from RandomCrop,
# so their inverse logic is identical)
FixedCrop.inverse = staticmethod(RandomCrop.inverse)
CenterCrop.inverse = staticmethod(RandomCrop.inverse)

# Populate the transform registry for inverse lookup
_TRANSFORM_REGISTRY= {
    'CropRandomObject': CropRandomObject,
    'RandomCrop': RandomCrop,
    'FixedCrop': FixedCrop,
    'CenterCrop': CenterCrop,
    'RandomCustomRotation': RandomCustomRotation,
    'RandomRotation': RandomRotation,
    'RandomHorizontalFlip': RandomHorizontalFlip,
    'RandomVerticalFlip': RandomVerticalFlip,
    'RandomResize': RandomResize,
    'Resize': Resize,
    'Padding': Padding,
}

def inverse_transform(sample, images=None, points=None, bboxes=None):
    """Apply inverse of all recorded transforms in reverse order.
    
    Args:
        sample: dict containing 'transforms' key with recorded transform history.
        images: list of image arrays/tensors to inverse-transform (H,W or C,H,W).
        points: list of point arrays, each shaped (N, 2) as (x, y).
        bboxes: list of bbox arrays, each shaped (N, 4+) where columns are
                [x, y, ...] or [id, x, y, ...] (odd column count means first col is id).
    
    Returns:
        dict with keys 'images', 'points', 'bboxes' containing inverse-transformed lists.
    """
    data = dict(
        images=images,
        points=points,
        bboxes=bboxes
    )

    transforms_raw = sample.get('transforms', None)
    if not transforms_raw:
        return data

    # should always be single batch
    transforms_raw = transforms_raw[0]

    # Decode: JSON string (normal path) or legacy dict/list format.
    # After collation + extract_batch, transforms_raw may be a list with
    # one JSON string element, e.g. ['[{"name": ...}]'].
    assert isinstance(transforms_raw, str), f"Expected 'transforms' to be a JSON string, got {type(transforms_raw)}."

    transforms_history = json.loads(transforms_raw)

    # Apply in reverse order
    for entry in reversed(transforms_history):
        name = entry['name']
        params = entry['params']

        cls = _TRANSFORM_REGISTRY.get(name)
        if cls is None:
            raise ValueError(f"Unknown transform '{name}' — cannot compute inverse.")

        data = cls.inverse(params, **data)

    return data
