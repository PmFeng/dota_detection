#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 23:51:55 2019

@author: sgiit
"""

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')
    
    
parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
parser.add_argument(
    "--config-file",
    default="/home/sgiit/disk_1T/sgiit/Pengming_Feng/GitClone/dota_detection/configs/ship_detection_net/ship_detection_e2e_faster_rcnn_dconv_constrained_R_50_FPN_1x.yaml",
    metavar="FILE",
    help="path to config file",
)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument(
    "--ckpt",
    help="The path to the checkpoint for test, default is the latest checkpoint.",
    default='/home/sgiit/disk_2T/Train_Models/HRSC/hrsc_dconv_constrained_rotation_hw_R_50_FPN_1x_weight_AL10_AR_added_th_10/model_best.pth',
)
parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)

args = parser.parse_args()

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
distributed = num_gpus > 1


distributed = False
if distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    synchronize()

cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

save_dir = ""
logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
logger.info("Using {} GPUs".format(num_gpus))
logger.info(cfg)

logger.info("Collecting env info (might take some time)")
logger.info("\n" + collect_env_info())

model = build_detection_model(cfg)
model.to(cfg.MODEL.DEVICE)

# Initialize mixed-precision if necessary
use_mixed_precision = cfg.DTYPE == 'float16'
amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

output_dir = cfg.OUTPUT_DIR
checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
_ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

iou_types = ("bbox",)
if cfg.MODEL.MASK_ON:
    iou_types = iou_types + ("segm",)
if cfg.MODEL.KEYPOINT_ON:
    iou_types = iou_types + ("keypoints",)
output_folders = [None] * len(cfg.DATASETS.TEST)
dataset_names = cfg.DATASETS.TEST
if cfg.OUTPUT_DIR:
    for idx, dataset_name in enumerate(dataset_names):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        mkdir(output_folder)
        output_folders[idx] = output_folder
data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)

output_folder, dataset_name, data_loader_val = next(iter(zip(output_folders, dataset_names, data_loaders_val)))



# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.utils.comm import is_main_process, get_world_size
from maskrcnn_benchmark.utils.comm import all_gather
from maskrcnn_benchmark.utils.comm import synchronize
from maskrcnn_benchmark.utils.timer import Timer, get_time_str
from maskrcnn_benchmark.engine.bbox_aug import im_detect_bbox_aug


def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images.to(device))
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions

data_loader = data_loader_val
box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY
expected_results=cfg.TEST.EXPECTED_RESULTS
device=cfg.MODEL.DEVICE
expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL
# convert to a torch.device for efficiency
device = torch.device(device)
num_devices = get_world_size()
dataset = data_loader.dataset
total_timer = Timer()
inference_timer = Timer()
total_timer.tic()
predictions = compute_on_dataset(model, data_loader, device, inference_timer)

total_time = total_timer.toc()
total_time_str = get_time_str(total_time)


predictions = _accumulate_predictions_from_multiple_gpus(predictions)



extra_args = dict(
    box_only=box_only,
    iou_types=iou_types,
    expected_results=expected_results,
    expected_results_sigma_tol=expected_results_sigma_tol,
)

# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py)
from __future__ import division
import math
from PIL import Image
import os
from collections import defaultdict
import numpy as np
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou


pred_boxlists = []
gt_boxlists = []
for image_id, prediction in enumerate(predictions):
    img_id = dataset.ids[image_id]
    img = Image.open(dataset._imgpath % img_id).convert("RGB")
    img_info = dataset.get_img_info(image_id)
    image_width = img_info["width"]
    image_height = img_info["height"]
#    image_width = img.size[0]
#    image_height = img.size[1]
    prediction = prediction.resize((image_width, image_height))
    pred_boxlists.append(prediction)

    gt_boxlist = dataset.get_groundtruth(image_id)
    gt_boxlists.append(gt_boxlist)
result = eval_detection_voc(
    pred_boxlists=pred_boxlists,
    gt_boxlists=gt_boxlists,
    iou_thresh=0.5,
    use_07_metric=True,
)


img, target, idx = dataset.__getitem__(0)
import cv2 as cv
from torchvision.transforms.transforms import ToPILImage


pred = pred_boxlists[0]



def hrbb_anchor2obb_anchor(proposal, angle):
    
    hrbb_x_min = [proposal[0] - proposal[2]/2]
    hrbb_y_min = [proposal[1] - proposal[3]/2]
    hrbb_x_max = [proposal[0] + proposal[2]/2]
    hrbb_y_max = [proposal[1] + proposal[3]/2]   
    
    if angle < 0:
        angle = 90 + angle
        h = (proposal[3] - np.tan(angle/180*np.pi)*proposal[2])/(1-(np.tan(angle/180*np.pi))**2)
        w = h * np.tan(angle/180*np.pi)
        if h > proposal[3] or w > proposal[2]:
            h = (proposal[3] - np.tan((90-angle)/180*np.pi)*proposal[2])/(1-(np.tan((90-angle)/180*np.pi))**2)
            w = h * np.tan(angle/180*np.pi)
        if h < 0:
            h = proposal[3]+h
        if w < 0:
            w = proposal[2]+w
#        h = abs(h)
#        if h > proposal[3]:
#            h = h-proposal[3]
#        w = h * np.tan(angle/180*np.pi)
#        w = abs(w)
        obb_pt_1 = np.array([hrbb_x_min, hrbb_y_min + h])
        obb_pt_2 = np.array([hrbb_x_min + w, hrbb_y_min])
        obb_pt_3 = np.array([hrbb_x_max, hrbb_y_max - h])
        obb_pt_4 = np.array([hrbb_x_max - w, hrbb_y_max])
    else:
        angle = 90 - angle
        h = (proposal[3] - np.tan(angle/180*np.pi)*proposal[2])/(1-(np.tan(angle/180*np.pi))**2)
        w = h * np.tan(angle/180*np.pi)
        if h < 0:
            h = proposal[3]+h
        if w < 0:
            w = proposal[2]+w
        obb_pt_1 = np.array([hrbb_x_min, hrbb_y_max - h])
        obb_pt_2 = np.array([hrbb_x_max - w, hrbb_y_min])
        obb_pt_3 = np.array([hrbb_x_max, hrbb_y_min + h])
        obb_pt_4 = np.array([hrbb_x_min + w, hrbb_y_max])

    
    obb_bbox = np.array([
            obb_pt_1,
            obb_pt_2,
            obb_pt_3,
            obb_pt_4
        ], dtype=np.int64)
    
    
    
    return obb_bbox, h, w



size_data = len(dataset)
i = 0
while i < size_data:   
    #img, label, idd = dataset.__getitem__(i)
    img_id = dataset.ids[i]
    label = dataset.get_groundtruth(i)
    img = Image.open(dataset._imgpath % img_id).convert("RGB")
    #image = ToPILImage()(img)
    img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
    boxes, theta, rc_boxes = label.get_field('obb_boxes').bbox.numpy(), label.get_field('theta').numpy(), label.bbox.numpy()
    for bbox, angle, rc_box in zip(boxes, theta, rc_boxes):
        bbox = np.array(bbox, np.int32)
        bbox = [bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2, bbox[2], bbox[3]]
        rc_box = np.array(rc_box, np.int32)
    #    rect_rota = pts.reshape(2,2)
    #    rect_rota.a
        rect_rota = ((bbox[0], bbox[1]), 
                    (bbox[2], bbox[3]), 
                   angle)
        rc_tota = (int(bbox[0]-(bbox[2]/2)), int(bbox[1]-(bbox[3]/2)), int(bbox[0]+(bbox[2]/2)), int(bbox[1]+(bbox[3]/2)))
        
        box = cv.boxPoints(rect_rota)
        box = np.int0(box)
        big_box = (np.min(box[:, 0]), np.min(box[:, 1]), np.max(box[:, 0]), np.max(box[:, 1]))
        big_box_xywh = [
                (big_box[0]+big_box[2]) // 2, 
                (big_box[1]+big_box[3]) // 2,
                big_box[2]-big_box[0],
                big_box[3]-big_box[1],
                ]
        cv.drawContours(img,[box],0,(0,255,255),2)
        label = "d:{:.1f}".format(angle)
#        cv.putText(img, label, (rc_box[0], rc_box[1] - 2), 0, 0.5, [225, 255, 255], 2)
#        cv.rectangle(img, (big_box[0], big_box[1]), (big_box[2], big_box[3]), (0,255,0), 2)
#        box_recover = hrbb_anchor2obb_anchor(big_box_xywh, angle)
#        cv.drawContours(img,[box_recover],0,(0,255,255),2)
        
        #cv.rectangle(img, (rc_box[0], rc_box[1]), (rc_box[2], rc_box[3]), (0,255,0), 2)
        #cv.rectangle(img, (rc_tota[0], rc_tota[1]), (rc_tota[2], rc_tota[3]), (255,0,0), 2)
        
    #    cv.polylines(img,[box],True,(0,0,255),2)
        #cv.polylines(img,[pts], True, (0,0,255), thickness = 5)
    bboxeslist = pred_boxlists[i]
    thetas = bboxeslist.get_field('theta').numpy()
    bboxes = bboxeslist.bbox
    for bbox, theta in zip(bboxes, thetas):
        bbox = np.array(bbox.numpy(), dtype=np.int64)
        
        box_xywh = [
                (bbox[0]+bbox[2]) // 2, 
                (bbox[1]+bbox[3]) // 2,
                bbox[2]-bbox[0],
                bbox[3]-bbox[1],
                ]
        box_recover, h, w = hrbb_anchor2obb_anchor(box_xywh, theta)
        print(box_recover.shape)
        cv.drawContours(img,[box_recover],0,(0,0,255),2)
        label = "pd:{:.1f},h:{:.1f},w:{:.1f}".format(theta, h, w)
#        cv.putText(img, label, (bbox[0], bbox[3] + 30), 0, 0.5, [225, 0, 255], 2)
        label = "H:{},W:{}".format(box_xywh[3], box_xywh[2])
#        cv.putText(img, label, (bbox[0]+ box_xywh[2]//2, bbox[1] - 5), 0, 0.5, [225, 255, 255], 2)
#        cv.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,255,0), 2)
    
    
    
    cv.imshow('image: {}'.format(i),img)
    k = cv.waitKey(0)
    cv.destroyAllWindows()
    if k == ord('n'):
        i+=1
        continue
        #print("image: {}".format(i))
    elif k == ord('m'):
        i-=1
        continue
    elif k == ord('q'):
        print('done')
        break
    
cv.destroyAllWindows() 