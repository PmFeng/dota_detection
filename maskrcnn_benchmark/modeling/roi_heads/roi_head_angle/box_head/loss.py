# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F
import torch.nn as nn

from maskrcnn_benchmark.layers import smooth_l1_loss
#from maskrcnn_benchmark.modeling.box_coder_with_constrained_and_diff_angle import BoxCoder
from maskrcnn_benchmark.modeling.box_coder_with_only_hw import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self,
        proposal_matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg=False,
        lambda_integrated=True,
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields(["labels", "theta", "obb_boxes","pt_inbox"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            target_size = len(targets_per_image)
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            #ex_theta = matched_targets.get_field('theta')
            #gt_theta = matched_targets.get_field('theta')
            gt_pt_hw = matched_targets.get_field('pt_inbox')
#            for tl in range(1, target_size+1):
#                gt_theta[-tl] = 0
            
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox, gt_pt_hw.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets
    
    def prepare_targets_no_integrated(self, proposals, targets):
        labels = []
        regression_targets = []
        regression_angles = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            target_size = len(targets_per_image)
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            #ex_theta = matched_targets.get_field('theta')
            gt_theta = matched_targets.get_field('theta')
            for tl in range(1, target_size+1):
                gt_theta[-tl] = 0
            
            # compute regression targets
            regression_targets_per_image,  regression_angle_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox, gt_theta
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            regression_angles.append(regression_angle_per_image)

        return labels, regression_targets, regression_angles

    def subsample(self, proposals, targets, lambda_integrated):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:self
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        if lambda_integrated:
            labels, regression_targets = self.prepare_targets(proposals, 
                                                              targets)
        else:
            labels, regression_targets, regression_angles = self.prepare_targets_no_integrated(proposals, 
                                                                            targets)
            
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        if lambda_integrated:
            for labels_per_image, regression_targets_per_image, proposals_per_image in zip(
                labels, regression_targets, proposals
            ):
                proposals_per_image.add_field("labels", labels_per_image)
                proposals_per_image.add_field(
                    "regression_targets", regression_targets_per_image
                )
        else:
            for labels_per_image, regression_targets_per_image, regression_angles_per_image, proposals_per_image in zip(
                labels, regression_targets, regression_angles, proposals
            ):
                proposals_per_image.add_field("labels", labels_per_image)
                proposals_per_image.add_field(
                    "regression_targets", regression_targets_per_image
                )
                proposals_per_image.add_field(
                    "regression_angles", regression_angles_per_image
                )
        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def call_no_integrated(self, class_logits, box_regression, lambda_regression):


        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        lambda_regression = cat(lambda_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )
        regression_angles = cat(
            [proposal.get_field("regression_angles") for proposal in proposals], dim=0
        )

        classification_loss = F.cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)
            lambda_map_inds = labels_pos[:, None] 

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        box_loss = box_loss / labels.numel()
        
        ang_loss = smooth_l1_loss(
            lambda_regression[sampled_pos_inds_subset[:, None], lambda_map_inds],
            regression_angles[sampled_pos_inds_subset, None],
            size_average=False,
            beta=1,
        )
        ang_loss = ang_loss / labels.numel()

        return classification_loss, box_loss, ang_loss
 
    
    
    
    def call_integrated(self, class_logits, box_regression):

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        classification_loss = F.cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([5, 6, 7, 8, 9], device=device)
        else:
#            map_inds = 6 * labels_pos[:, None] + torch.tensor(
#                [0, 1, 2, 3, 4, 5], device=device)
            map_inds = 6 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3, 4, 5], device=device)
            
#        box_loss_ = box_regression[sampled_pos_inds_subset[:, None], map_inds] - regression_targets[sampled_pos_inds_subset]

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
#        print(box_loss_)
#        print(type(box_loss_))
        
#        loss_box = nn.L1Loss()
#        
#        box_loss = loss_box(box_regression[sampled_pos_inds_subset[:, None], map_inds],
#                            regression_targets[sampled_pos_inds_subset])  
        
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss
    
    
    def call_integrated_ratio(self, class_logits, box_regression):

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        classification_loss = F.cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([5, 6, 7, 8, 9], device=device)
        else:
#            map_inds = 6 * labels_pos[:, None] + torch.tensor(
#                [0, 1, 2, 3, 4, 5], device=device)
            map_inds = 5 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3, 4], device=device)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
#        
#        box_regression_ratio = box_regression[sampled_pos_inds_subset[:, None], map_inds][:, 4] / regression_targets[sampled_pos_inds_subset][:, 6]
#        box_ratio_loss = smooth_l1_loss(
#            box_regression_ratio,
#            regression_targets[sampled_pos_inds_subset][:, 5],
#            size_average=False,
#            beta=1,
#        )
#        box_ratio_loss = box_ratio_loss / labels.numel()
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss
    
    
    def __call__(self, class_logits, box_regression, lambda_integrated=None):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """
        
        if lambda_integrated is None:
            return self.call_integrated(class_logits, 
                                        box_regression)
        else:
            return self.call_no_integrated(class_logits, 
                                           box_regression, 
                                           lambda_integrated)
    
    
    


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg
    )

    return loss_evaluator
