import os
import cv2
import json
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from utils.evaluation import NumpyEncoder


def fill_holes(image):
    dims = len(image.shape)
    if dims > 3:
        image = image[0]

    assert image.shape[0] == 1, "Currently only one segemntation channel is supported"

    components = cv2.connectedComponents(1 - image[0].cpu().numpy().astype(np.uint8))

    # 0 krpa
    # 1 background
    # 2-... clutter

    filled_mask = torch.tensor(1 - (components[1] == 1), dtype=torch.float32).unsqueeze(0).to(image.device)

    if dims > 3:
        filled_mask = filled_mask.unsqueeze(0)

    return filled_mask


class SegmentationIoUEval:
    def __init__(self, exp_name, thresholds, fill_holes) -> None:
        self.exp_name = exp_name
        self.thresholds = thresholds
        self.fill_holes = fill_holes
        self.ious = defaultdict(list)

    def save_str(self):
        return f"iou_filled={self.fill_holes}"

    def get_results_timestamp(self, save_dir):
        res_filename = os.path.join(save_dir, self.exp_name, self.save_str(), 'results.json')

        return os.path.getmtime(res_filename) if os.path.exists(res_filename) else 0

    def add_image_prediction(self, sample, im_name, segmentation, segmentation_gt):

        # TODO: dodaj to kr na model?
        segmentation = F.sigmoid(segmentation)

        if self.fill_holes:
            segmentation_gt = fill_holes(segmentation_gt)

        image_ious = {}
        thresholded_images = {}
        for thr in self.thresholds:
            thresholded = (segmentation > thr) * 1

            if self.fill_holes:
                thresholded = fill_holes(thresholded)

            overlap = segmentation_gt * thresholded
            union = (segmentation_gt + thresholded) > 0

            iou = overlap.sum() / union.sum()

            image_ious[thr] = iou.cpu()
            self.ious[thr].append(iou.cpu())

            thresholded_images[thr] = thresholded

        return {"ious": image_ious, "gt": segmentation_gt, "thresholded": thresholded_images}

    def calc_and_display_final_metrics(self, print_result=True, save_dir=None):
        metrics = {f"iou_{thr}": np.mean(ious) for thr, ious in self.ious.items()}

        if print_result:
            print(f"Segmentation IoU: {metrics}")

        out_dir = os.path.join(save_dir, self.exp_name, self.save_str())
        os.makedirs(out_dir, exist_ok=True)

        with open(os.path.join(out_dir, "results.json"), "w") as file:
            file.write(json.dumps(metrics, cls=NumpyEncoder))

        return metrics


def get_edge_mask(image, se):
    N = se.shape[0]
    while len(image.shape) < 4:
        image = image.unsqueeze(0)
    eroded = (F.conv2d(image, se[None,None,:,:], padding=N // 2) == N**2) * 1
    edge_mask = (image - eroded) > 0

    return edge_mask


class EdgeOverlapEval:
    def __init__(self, exp_name, threholds, edge_sizes, fill_holes) -> None:
        self.exp_name = exp_name
        self.thresholds = threholds
        self.edge_sizes = edge_sizes
        self.fill_holes = fill_holes
        self.edge_overlap = defaultdict(lambda: defaultdict(list))

    def save_str(self):
        return f"edge_iou_filled={self.fill_holes}"

    def get_results_timestamp(self, save_dir):
        res_filename = os.path.join(save_dir, self.exp_name, self.save_str(), 'results.json')

        return os.path.getmtime(res_filename) if os.path.exists(res_filename) else 0

    def add_image_prediction(self, im_name, segmentation, segmentation_gt):
        segmentation = F.sigmoid(segmentation)

        if self.fill_holes:
            segmentation_gt = fill_holes(segmentation_gt)

        for edge_px in self.edge_sizes:
            N = 2 * edge_px + 1
            se = torch.ones(N, N).to(segmentation.device)

            edge_mask = get_edge_mask(segmentation_gt, se)

            for thr in self.thresholds:
                thresholded = (segmentation > thr) * 1.0
                thresholded_edge = get_edge_mask(thresholded, se)

                overlap = edge_mask * thresholded_edge
                union = (edge_mask + thresholded_edge) > 0

                iou = overlap.sum() / union.sum()

                self.edge_overlap[thr][edge_px].append(iou.cpu())

    def calc_and_display_final_metrics(self, print_result=True, save_dir=None):
        metrics = {k1: {k2: np.mean(overlap) for k2, overlap in v1.items()} for k1, v1 in self.edge_overlap.items()}

        if print_result:
            print(f"Edge Overlap: {metrics}")

        out_dir = os.path.join(save_dir, self.exp_name, self.save_str())
        os.makedirs(out_dir, exist_ok=True)

        with open(os.path.join(out_dir, "results.json"), "w") as file:
            file.write(json.dumps(metrics, cls=NumpyEncoder))

        return metrics
