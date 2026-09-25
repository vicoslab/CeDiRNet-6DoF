import numpy as np
import torch

from utils.visualize.center import CentersVisualizeTest

class CenterInstanceVisualizeTest(CentersVisualizeTest):

    def __init__(self, *args, instance_thr, **kwargs):
        super(CenterInstanceVisualizeTest, self).__init__(*args, **kwargs)

        self.instance_thr = instance_thr

    def __call__(self, sample, result, pred_data,  pred_gt_match, difficult, base, save_dir=None, plot_bbox_all=False):

        pred_poly_mask = pred_data.get('polygon')
        if len(pred_data['center']) > 0:
            pred_list, pred_score = map(np.array, zip(*[(pred_data['center'][id],
                                                         pred_data['score'][id]) for id in sorted(pred_data['center'].keys())]))
        else:
            pred_list, pred_score = [], []


        im, output, output_cls, resnet_features, CAM, \
            pred_heatmap, centerdir_gt, gt_list, is_difficult_gt = self.parse_results(sample, result, difficult)

        for pred_idx in range(len(pred_list)):
            score = pred_score[pred_idx]

            if score <= self.instance_thr:
                continue

            selected_pred_score = pred_score[pred_idx:pred_idx + 1]
            selected_pred = pred_list[pred_idx:pred_idx + 1, :]
            selected_pred_match = pred_gt_match[pred_idx:pred_idx + 1, :]

            # set score of the prediction for filename prefix
            name_prefix = '%.5f_' % score

            # add IoU as prefix to filename
            iou = pred_poly_mask[pred_idx][2] if pred_idx in pred_poly_mask else 0
            name_prefix = '%.5f_%s' % (iou, name_prefix)

            selected_pred_poly_mask = []
            pred_gt = pred_list[pred_idx, 0]

            # apply ROI if needed
            sel_im, sel_output, sel_output_cls, sel_pred_heatmap, sel_centerdir_gt, \
                sel_resnet_features, sel_CAM, sel_difficult = self.apply_roi(im.clone(), output, selected_pred, selected_pred_poly_mask,
                                                                             output_cls, pred_heatmap, centerdir_gt, resnet_features, CAM, difficult)
            # apply border color
            sel_im = self.apply_border_color(sel_im, pred_gt)

            sel_sample = dict(image=[sel_im],
                              center_dict=sample['center_dict'],
                              centerdir_groundtruth=sel_centerdir_gt)
            sel_result = dict(output=sel_output,
                              resnet_features=sel_resnet_features,
                              CAM=sel_CAM,
                              pred_heatmap=sel_pred_heatmap)

            super(CenterInstanceVisualizeTest, self).__call__(sel_sample, sel_result,
                                                               selected_pred, selected_pred_score, selected_pred_poly_mask, selected_pred_match,
                                                               sel_difficult, '%s%s' % (name_prefix, base), save_dir)


    def apply_roi(self, im, output, selected_pred, selected_pred_poly_mask, output_cls, pred_heatmap, centerdir_gt, resnet_features, CAM, difficult):
        sel_im = im
        sel_output, sel_output_cls, sel_pred_heatmap = output, output_cls, pred_heatmap
        sel_centerdir_gt = centerdir_gt
        sel_resnet_features, sel_CAM = resnet_features, CAM
        sel_difficult = difficult

        # apply ROI as
        if len(selected_pred_poly_mask) > 0:
            poly = selected_pred_poly_mask[0][0]
            (x1, y1), (x2, y2) = poly.min(axis=0), poly.max(axis=0)
            w, h = max(50, x2 - x1), max(50, y2 - y1)
            x1, y1, x2, y2 = x1 - w, y1 - h, x2 + w, y2 + h
            x1, x2 = map(lambda x: np.clip(x, 0, im.shape[1]), [x1, x2])
            y1, y2 = map(lambda x: np.clip(x, 0, im.shape[2]), [y1, y2])
            sel_im = im[:, x1:x2, y1:y2]

            # crop all data with coordinates
            sel_output = sel_output[:, :, x1:x2, y1:y2]
            # sel_output_cls = sel_output_cls[:, :, x1:x2, y1:y2]
            sel_pred_heatmap = [h[:, x1:x2, y1:y2] for h in sel_pred_heatmap]
            sel_centerdir_gt = {k:v[:, :, x1:x2, y1:y2] for k,v in sel_centerdir_gt.items()}
            selected_pred[:, :2] = selected_pred[:, :2] - np.array([y1, x1])
            selected_pred_poly_mask[0][0] = selected_pred_poly_mask[0][0] - np.array([x1, y1])
            sel_difficult = difficult[:, :, x1:x2, y1:y2]

        return sel_im, sel_output, sel_output_cls, sel_pred_heatmap, sel_centerdir_gt, \
               sel_resnet_features, sel_CAM, sel_difficult

    def apply_border_color(self, im, pred_gt):
        border_mask = torch.ones_like(im[0])
        border_mask[5:-5, 5:-5] = 0

        return torch.stack((im[0, :, :] * (1 - border_mask) + (1 - pred_gt) * (border_mask),
                            im[1, :, :] * (1 - border_mask) + (pred_gt) * (border_mask),
                            im[2, :, :] * (1 - border_mask) + 0 * (border_mask)), dim=0)
