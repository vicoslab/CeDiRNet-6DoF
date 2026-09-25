import numpy as np
from typing import Any

import os

from utils.transforms import inverse_transform


class CenterExport:

    def __init__(self, impath2name_fn=None, score_thr=0.01, attributes=None, attribute_inverse_fn=None):
        if impath2name_fn is None:
            self.impath2name_fn = lambda name: os.path.splitext(os.path.basename(name))[0]
        else:
            self.impath2name_fn = impath2name_fn

        self.score_thr = score_thr
        self.attributes = attributes
        if type(self.attributes) is dict:
            self.attribute_keys = self.attributes.keys()
        elif type(self.attributes) in [list, tuple]:
            self.attribute_keys = self.attributes
            self.attributes = {attr:[attr] for attr in self.attributes}
        else:
            self.attribute_keys = None

        self.attribute_inverse_fn = attribute_inverse_fn


    def __call__(self, sample, result, evaluation_lists, save_dir) -> Any:

        scoring_fn_list = []
        # find all scoring_fn from evaluation list
        for eval_args in evaluation_lists:
            scoring_fn_list.append(eval_args['scoring_fn'])

        # use only unique scoring_fn
        scoring_fn_list = set(scoring_fn_list)

        im_name = self.impath2name_fn(sample['im_name'])
        org_im_size = sample['org_im_size'][0]

        predictions = result['predictions']
        all_scores = predictions[:, 2:] if len(predictions) > 0 else []

        pred_centers_ = predictions[:,[0,1]]
        if self.attribute_keys is not None:
            assert 'pred_attributes' in result, "Missing 'pred_attributes' in result"
            pred_attributes = {attr: result['pred_attributes'][attr] for attr in self.attribute_keys}
        else:
            pred_attributes = {}

        # Use inverse_transform to undo all recorded transforms (Padding, Crop, Flip, etc.)
        if len(pred_centers_) > 0:
            inv_result = inverse_transform(sample, points=[pred_centers_])
            pred_centers_ = inv_result['points'][0]

        # Handle the initial dataset resize (org_im_size -> im_size) which is not
        # captured by the transforms list since it happens in the dataset __getitem__
        if len(pred_centers_) > 0:
            # After inverse_transform, centers are in the pre-transform coordinate
            # space, i.e. the dataset-resized image size. Scale to original image size.
            # im_size is (W, H) from dataset, org_im_size is (W, H) from original PIL image
            im_size = sample['im_size']
            # handle im_size being tensor, list of tensors, or tuple after collation
            if isinstance(im_size, (list, tuple)):
                im_w = im_size[0].item() if hasattr(im_size[0], 'item') else im_size[0]
                im_h = im_size[1].item() if hasattr(im_size[1], 'item') else im_size[1]
            else:
                im_w = im_size[0].item() if hasattr(im_size, 'item') else im_size[0]
                im_h = im_size[1].item() if hasattr(im_size, 'item') else im_size[1]
            org_w = org_im_size[0].item() if hasattr(org_im_size[0], 'item') else org_im_size[0]
            org_h = org_im_size[1].item() if hasattr(org_im_size[1], 'item') else org_im_size[1]
            pred_centers_[:, 0] *= org_w / im_w
            pred_centers_[:, 1] *= org_h / im_h

        # Apply inverse function for attributes if provided
        for attr in pred_attributes.keys():
            if self.attribute_inverse_fn is not None and attr in self.attribute_inverse_fn:
                pred_attributes[attr] = self.attribute_inverse_fn[attr](pred_attributes[attr], sample, org_im_size, im_size)

        for i,scoring_fn in enumerate(scoring_fn_list):
            pred_scores_ = scoring_fn(all_scores)

            idx = np.where((pred_scores_ > self.score_thr) * (predictions.sum(axis=1) != 0))[0]

            pred_centers = pred_centers_[idx,:]
            pred_scores = pred_scores_[idx]
            pred_attributes_filtered = {attr: attr_vals[idx] for attr, attr_vals in pred_attributes.items()}

            idx = np.argsort(pred_scores)
            pred_centers = pred_centers[idx[::-1],:]
            pred_scores = pred_scores[idx[::-1]]
            pred_attributes_filtered = {attr: attr_vals[idx[::-1]] for attr, attr_vals in pred_attributes_filtered.items()}

            columns = ['x', 'y', 'confidence_score']
            pred = np.concatenate([pred_centers, pred_scores.reshape(-1,1)], axis=1)

            if self.attribute_keys is not None:
                pred = np.concatenate(list([pred]) + list([pred_attributes_filtered[attr] for attr in self.attribute_keys]), axis=1)
                for attr_key in self.attribute_keys:
                    columns += self.attributes[attr_key]

            out_file = os.path.join(save_dir, "exported_detections" + (f"_{i}" if i > 0 else ""), f"{im_name}.csv")

            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            np.savetxt(out_file, pred, delimiter=',', fmt='%.4f', header=",".join(columns), comments='')
