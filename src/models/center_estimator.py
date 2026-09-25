import torch
import torch.nn as nn

import numpy as np

from models.center_augmentator import CenterAugmentator

from models.localization.centers import Conv1dMultiscaleLocalization, Conv2dDilatedLocalization
from models.center_groundtruth import CenterDirGroundtruth

class CenterEstimator(nn.Module):
    def __init__(self, args=dict(), is_learnable=True):
        super().__init__()

        self.return_backbone_only = False

        self.suppression_by_mask = args.get('suppression_by_mask')
        self.use_magnitude_as_mask = args.get('use_magnitude_as_mask')
        self.add_edges_to_area_score = args.get('add_edges_to_area_score')
        self.mask_inference_on_gorundtruth_points_only = args.get('mask_inference_on_gorundtruth_points_only', False)

        instance_center_estimator_op = Conv1dMultiscaleLocalization
        if args.get('use_dilated_nn'):
            from functools import partial
            instance_center_estimator_op = partial(Conv2dDilatedLocalization,
                                                   **args.get('dilated_nn_args',{}))

        self.instance_center_estimator = instance_center_estimator_op(
            local_max_thr=args.get('local_max_thr', 0.1),
            mask_thr=args.get('mask_thr', 0.01),
            exclude_border_px=args.get('exclude_border_px', 5),
            learnable=is_learnable,
            allow_input_backprop=args.get('allow_input_backprop', True),
            backprop_only_positive=args.get('backprop_only_positive', True),
            apply_input_smoothing_for_local_max=args.get('apply_input_smoothing_for_local_max', 1),
            use_findcontours_for_local_max=args.get('use_findcontours_for_local_max', False),
        )

        self.instance_mask_estimator = None
        if self.suppression_by_mask:
            raise NotImplementedError("suppression_by_mask is not supported in the public release")
        if self.add_edges_to_area_score:
            raise NotImplementedError("add_edges_to_area_score is not supported in the public release")
        if self.mask_inference_on_gorundtruth_points_only:
            raise NotImplementedError("mask_inference_on_gorundtruth_points_only is not supported in the public release")
        if args.get('use_fourier_instance', False):
            raise NotImplementedError("use_fourier_instance is not supported in the public release")
        if args.get('augmentation'):
            self.center_augmentator = CenterAugmentator(
                **args.get('augmentation_kwargs'),
            )
        else:
            self.center_augmentator = None

        scale_r = 1024 if 'scale_r' not in args else args['scale_r']
        scale_r_gt = 1 if 'scale_r_gt' not in args else args['scale_r_gt']

        self.scale_r_fn = lambda x: x * scale_r
        self.scale_r_gt_fn = lambda x: x * scale_r_gt

        self.inverse_scale_r_fn = lambda x: x / scale_r

        self.use_log_r = args['use_log_r'] if 'use_log_r' in args else True
        use_log_r_base = args['use_log_r_base'] if 'use_log_r_base' in args else 'exp'

        if use_log_r_base.lower() in ['exp', 'e']:
            self.log_r_fn = lambda x: torch.log(x+1)
            self.inverse_log_r_fn = lambda x: torch.exp(x)-1
        elif use_log_r_base.lower() in ['decimal', '10']:
            self.log_r_fn = lambda x: torch.log10(x+1)
            self.inverse_log_r_fn = lambda x: torch.pow(10, x)-1
        elif use_log_r_base.lower() in ['pow10']:
            self.log_r_fn = lambda x: torch.log10(x+1)
            self.inverse_log_r_fn = lambda x: torch.pow(x, 10)-1
        else:
            raise Exception('Only "exp" and "10" are allowed logarithms for R')

        self.MAX_NUM_CENTERS = 4*512

        self.ignore_cls_prediction = args.get('ignore_cls_prediction', False)
        self.ignore_centerdir_magnitude = args.get('ignore_centerdir_magnitude', False)

    def set_return_backbone_only(self, val):
        self.return_backbone_only = val

    def is_return_backbone_only(self):
        return self.return_backbone_only

    def init_output(self, num_vector_fields=1):
        self.num_vector_fields = num_vector_fields

        assert self.num_vector_fields >= 3
        self.instance_center_estimator.init_output()

        return input

    def forward(self, input, ignore_gt=False, **gt):
        if self.center_augmentator is not None:
            input = self.center_augmentator(input, **gt)

        ignore = gt.get('ignore')

        assert input.shape[1] >= self.num_vector_fields

        predictions = input[:, 0:self.num_vector_fields]
        cls_mask = input[:, self.num_vector_fields:]  # 1 x h x w

        S = predictions[:, 0].unsqueeze(1)
        C = predictions[:, 1].unsqueeze(1)
        R = predictions[:, 2].unsqueeze(1)
        M = torch.sqrt(torch.pow(S, 2) + torch.pow(C, 2))

        R = self.inverse_log_r_fn(self.scale_r_fn(R))


        if self.ignore_cls_prediction:
            # just set to ZERO for legacy compatibility (original center models require this channel )
            cls_mask = torch.zeros_like(cls_mask, requires_grad=False)
        if self.ignore_centerdir_magnitude:
            # just set to ZERO for legacy compatibility (original center models require this channel )
            M = torch.zeros_like(M, requires_grad=False)

        if self.training:
            # during training only detect centers but do not do any filtering
            center_pred, conv_resp = self.instance_center_estimator(C, S, R, M, cls_mask)
        else:
            # during inference detect centers and do additional filtering and scoring if requested

            # apply R adjustment for log directly to input and GT data only during testing
            if self.use_log_r and 'centerdir_groundtruth' in gt and not ignore_gt:
                input[:, 2:3] = R
                gt['centerdir_groundtruth']['gt_R'] = self.scale_r_gt_fn(CenterDirGroundtruth.parse_groundtruth_map(gt.get('centerdir_groundtruth'),keys=['gt_R']))

            mask = M if self.use_magnitude_as_mask else (cls_mask * (cls_mask > 0).type(torch.float32))

            # use only ignore flag == 1 here, but have all remaining ignore types handled afterwards
            res, conv_resp = self.instance_center_estimator(C, S, R, M, mask,
                                                            ignore_region=ignore & 1 if ignore is not None else None)

            # need to apply relu to ignore raw negative values returned by net
            conv_resp = torch.relu(conv_resp)

            res = torch.cat((res, torch.ones((len(res),1),device=res.device)),dim=1)
            #res = [np.concatenate((x, [1.0]), axis=0) for x in res]

            res = res.cpu().numpy()
            if len(res) > 0:
                #idx = np.argsort(res[:, 3])
                idx = np.lexsort((res[:, 3],res[:, 4]))
                res = res[idx[::-1], :]

            # take only 2000 examples if too may
            if res.shape[0] > 2000:
                res = res[:2000, :]

            if len(res) > 0 and ignore is not None:
                selected_centers = np.ones(len(res), dtype=bool)
                ignore_np = ignore.clone().cpu().numpy()
                for b in range(len(input)):
                    batch_idx = res[:, 0] == b
                    centers_b = res[batch_idx][:, [2, 1, 4]]
                    ignored_pred = np.array([
                        ignore_np[b, 0, int(r[0]), int(r[1])] & (255 - 8 - 64 - 128) == 0
                        for r in centers_b
                    ]).astype(bool)
                    selected_centers[batch_idx] *= ignored_pred
                center_pred = res[selected_centers, :]
            else:
                center_pred = res

        center_pred = torch.tensor(center_pred).to(input.device)

        # convert center prediction list to tensor of fixed size so that it can be merger from parallel GPU processings
        center_pred = self._pack_center_predictions(center_pred, batch_size=len(input))

        return dict(output=input, center_pred=center_pred, center_heatmap=conv_resp, pred_mask=None)

    def _pack_center_predictions(self, center_pred, batch_size):
        center_pred_all = torch.zeros((batch_size, self.MAX_NUM_CENTERS, center_pred.shape[1] if len(center_pred) > 0 else 5),
                                      dtype=torch.float, device=center_pred.device)
        if len(center_pred) > 0:
            for b in center_pred[:, 0].unique().long():
                valid_centers_idx = torch.nonzero(center_pred[:, 0] == b.float()).squeeze(dim=1).long()

                if len(valid_centers_idx) > self.MAX_NUM_CENTERS:
                    valid_centers_idx = valid_centers_idx[:self.MAX_NUM_CENTERS]
                    print('WARNING: got more centers (%d) than allowed (%d) - removing last centers to meet criteria' % (len(valid_centers_idx), self.MAX_NUM_CENTERS))

                center_pred_all[b, :len(valid_centers_idx), 0] = 1
                center_pred_all[b, :len(valid_centers_idx), 1:] = center_pred[valid_centers_idx, 1:]

        return center_pred_all
