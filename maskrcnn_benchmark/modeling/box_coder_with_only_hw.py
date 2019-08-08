#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 02:06:19 2019

@author: sgiit
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 23:34:40 2019

@author: sgiit
"""

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import torch

import numpy as np


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        """
        Arguments:
            weights (5-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

#matched_targets.bbox, proposals_per_image.bbox, gt_theta

    def encode(self, reference_boxes, proposals, pt_box):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        
        

        TO_REMOVE = 1  # TODO remove
        ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
        ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights
        

        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + TO_REMOVE
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights
            
        gt_pt_w = pt_box[:, 2] + TO_REMOVE
        gt_pt_h = pt_box[:, 3] + TO_REMOVE
        
        wx, wy, ww, wh, wl = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
#        
#        print("gt_widths: ", gt_widths)
#        print("gt_heights: ", gt_heights)
#        print("ex_widths: ", ex_widths)
#        print("ex_heights: ", ex_heights)
#        print("gt_pt_h: ", gt_pt_h)
#        print("gt_pt_w: ", gt_pt_w)
        
    
        targets_pth = wl * torch.log(gt_pt_h / ex_heights)
        targets_ptw = wl * torch.log(gt_pt_w / ex_widths)
        
#        print("targets_pth: ", targets_pth)
#        print("targets_ptw: ", targets_ptw)

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh, targets_ptw, targets_pth), dim=1)
        return targets
    
   

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        boxes = boxes.to(rel_codes.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh, wl = self.weights
        dx = rel_codes[:, 0::6] / wx
        dy = rel_codes[:, 1::6] / wy
        dw = rel_codes[:, 2::6] / ww
        dh = rel_codes[:, 3::6] / wh
        dptw = rel_codes[:, 4::6] / wl
        dpth = rel_codes[:, 5::6] / wl
        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)
        dptw = torch.clamp(dptw, max=self.bbox_xform_clip)
        dpth = torch.clamp(dpth, max=self.bbox_xform_clip)


        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]
        pred_ptw = torch.exp(dptw) * widths[:, None]
        pred_pth = torch.exp(dpth) * heights[:, None]
        pred_pthw = torch.stack((pred_pth, pred_ptw), dim=1)
        #pred_boxes = torch.zeros_like(rel_codes)
        shape_pre = rel_codes.size()
        #print("======================================: {}".format(int(shape_pre[1]/5*4)))
        pred_boxes = rel_codes.new(shape_pre[0], int(shape_pre[1]/5*4))
#        # x1
#        pred_boxes[:, 0::5] = pred_ctr_x - 0.5 * pred_w
#        # y1
#        pred_boxes[:, 1::5] = pred_ctr_y - 0.5 * pred_h
#        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
#        pred_boxes[:, 2::5] = pred_ctr_x + 0.5 * pred_w - 1
#        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
#        pred_boxes[:, 3::5] = pred_ctr_y + 0.5 * pred_h - 1
        
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w 
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h 
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

        return pred_boxes, pred_pthw
