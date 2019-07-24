# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
#from maskrcnn_benchmark.modeling.box_coder_with_constrained_and_diff_angle import BoxCoder
from maskrcnn_benchmark.modeling.box_coder_with_angle import BoxCoder


class PostProcessor_(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
        cls_agnostic_bbox_reg=False,
        bbox_aug_enabled=False,
        lambda_integrated=True
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor_, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.bbox_aug_enabled = bbox_aug_enabled
        self.lambda_integrated = lambda_integrated
        
    def forward_integrated(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """

        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)
            
       
        
        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]
        proposals, pre_thetas = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        
        # TODO cls_agnostic_bbox_reg for theta
        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        pre_thetas = pre_thetas.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_img, image_shape, lambda_ in zip(
            class_prob, proposals, image_shapes, pre_thetas
        ):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape, lambda_)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            if not self.bbox_aug_enabled:  # If bbox aug is enabled, we will do it later
                boxlist = self.filter_results(boxlist, num_classes)
            results.append(boxlist)      
            
        return results 
        
    
    def forward_no_integrated(self, x, boxes):   
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """

        class_logits, box_regression, lambda_regression = x
        class_prob = F.softmax(class_logits, -1)
        lambda_reg = lambda_regression
            
       
        
        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]
        
        proposals, angle = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), 
            lambda_reg.view(sum(boxes_per_image), -1), 
            concat_boxes
        )

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)
        angle = angle.split(boxes_per_image, dim=0)


        results = []
        for prob, boxes_per_img, image_shape, lambda_ in zip(
            class_prob, proposals, image_shapes, angle
        ):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape, lambda_)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            if not self.bbox_aug_enabled:  # If bbox aug is enabled, we will do it later
                boxlist = self.filter_results(boxlist, num_classes)
            results.append(boxlist)
        return results        
        
        
        

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        if self.lambda_integrated:
            results = self.forward_integrated(x, boxes)
        else:
            results = self.forward_no_integrated(x, boxes)

        return results

    def prepare_boxlist(self, boxes, scores, image_shape, lambda_):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        #print("+++++++++++++++++++++", scores.size())
        lambda_ = lambda_.reshape(-1)
        scores = scores.reshape(-1) 
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        
        boxlist.add_field("theta", lambda_)
        return boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        #boxlist = boxlist.convert("xyxy")
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)
        lambda_ = boxlist.get_field("theta").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            lambda_j = lambda_[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class.add_field("theta", lambda_j)
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result


def hrbb_anchor2obb_anchor(proposal, angle):
    
    hrbb_x_min = [proposal[0] - proposal[2]/2]
    hrbb_y_min = [proposal[1] - proposal[3]/2]
    hrbb_x_max = [proposal[0] + proposal[2]/2]
    hrbb_y_min = [proposal[1] + proposal[3]/2]   
    
    h = (proposal[3] - torch.tan(angle/180*np.pi)*proposal[2])/(1-(torch.tan(angle/180*np.pi))**2)
    w = h * torch.tanh(angle)
    
    obb_pt_1 = [hrbb_x_min, hrbb_y_max - h]
    obb_pt_2 = [hrbb_x_max - w, hrbb_y_min]
    obb_pt_3 = [hrbb_x_max, hrbb_y_min + h]
    obb_pt_4 = [hrbb_x_min + w, hrbb_y_max]
    
    obb_bbox = [obb_pt_1,
                obb_pt_2,
                obb_pt_3,
                obb_pt_4]
    
    
    
    return obb_bbox
    
    





def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    bbox_aug_enabled = cfg.TEST.BBOX_AUG.ENABLED
    
    lambda_integrated = cfg.MODEL.ROI_HEADS.LAMDA_INTEGRATED
        
    postprocessor = PostProcessor_(
        score_thresh,
        nms_thresh,
        detections_per_img,
        box_coder,
        cls_agnostic_bbox_reg,
        bbox_aug_enabled,
        lambda_integrated
    )
    return postprocessor
