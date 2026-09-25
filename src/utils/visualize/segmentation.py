import os
import cv2
import numpy as np
import torch.nn.functional as F
from utils.visualize.vis import Visualizer

# RGB colors
TRUE_POSITIVE = np.array((0, 255, 0))
FALSE_POSITIVE = np.array((255, 0, 0))
FALSE_NEGATIVE = np.array((255, 255, 0))

class SegmentationVisualizeTest(Visualizer):
    def __init__(self, iou_cutoff=0.8, **kwargs):
        super().__init__(keys=[], **kwargs)
        self.iou_cutoff = iou_cutoff

    def visualize(self, image, gt, eval_obj_res, thr, iou, save_dir, base):
        if iou > self.iou_cutoff:
            return

        thresholded = eval_obj_res["thresholded"][thr][0].cpu().numpy()

        tp = (gt * thresholded).astype(bool)
        fp = (thresholded - gt).clip(min=0).astype(bool)
        fn = (gt - thresholded).clip(min=0).astype(bool)

        vis_img = image.copy()
        vis_img[tp] = (vis_img[tp] * 0.3 + TRUE_POSITIVE * 0.7).astype(np.uint8)
        vis_img[fp] = (vis_img[fp] * 0.3 + FALSE_POSITIVE * 0.7).astype(np.uint8)
        vis_img[fn] = (vis_img[fn] * 0.3 + FALSE_NEGATIVE * 0.7).astype(np.uint8)

        path = os.path.join(save_dir, f"{base}_iou_{iou:.2f}_thr_{thr:.2f}.png")
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, vis_img)

    def visualize_opencv(self, image, eval_obj_res, base, save_dir):
        image = image[0, :3].cpu().numpy().transpose(1, 2, 0) * 255
        gt = eval_obj_res["gt"][0, 0].cpu().numpy()

        best_thr = 0
        best_iou = 0
        for thr, iou in eval_obj_res["ious"].items():
            if iou > best_iou:
                best_iou = iou
                best_thr = thr

        self.visualize(image, gt, eval_obj_res, best_thr, best_iou, save_dir, base)
