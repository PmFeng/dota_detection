#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 11:33:05 2019

@author: sgiit
"""

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')
parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
parser.add_argument(
    "--config-file",
    default="/home/sgiit/disk_1T/sgiit/Pengming_Feng/GitClone/dota_detection/configs/ship_detection_net/dota_e2e_faster_rcnn_dconv_R_50_FPN_1x.yaml",
    #default="/home/sgiit/disk_1T/sgiit/Pengming_Feng/GitClone/ship_detection_optical/configs/ship_detection_net/ship_detection_R_101_FPN_1x.yaml",
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument(
    "--skip-test",
    dest="skip_test",
    help="Do not test the final model",
    action="store_true",
)
parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)

args = parser.parse_args()

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
args.distributed = num_gpus > 1

if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    synchronize()


cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

output_dir = cfg.OUTPUT_DIR
if output_dir:
    mkdir(output_dir)



logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
logger.info("Using {} GPUs".format(num_gpus))
logger.info(args)

logger.info("Collecting env info (might take some time)")
logger.info("\n" + collect_env_info())

logger.info("Loaded configuration file {}".format(args.config_file))
with open(args.config_file, "r") as cf:
    config_str = "\n" + cf.read()
    logger.info(config_str)

logger.info("Running with config:\n{}".format(cfg))

output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
logger.info("Saving config into: {}".format(output_config_path))
# save overloaded model config in the output directory
save_config(cfg, output_config_path)
distributed = False
model = build_detection_model(cfg)
device = torch.device(cfg.MODEL.DEVICE)
model.to(device)

optimizer = make_optimizer(cfg, model)
scheduler = make_lr_scheduler(cfg, optimizer)

# Initialize mixed-precision training
use_mixed_precision = cfg.DTYPE == "float16"
amp_opt_level = 'O1' if use_mixed_precision else 'O0'
model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

if distributed:
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank,
        # this should be removed if we update BatchNorm stats
        broadcast_buffers=False,
    )


arguments = {}
arguments["iteration"] = 0

output_dir = cfg.OUTPUT_DIR

save_to_disk = get_rank() == 0
checkpointer = DetectronCheckpointer(
    cfg, model, optimizer, scheduler, output_dir, save_to_disk
)
extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
arguments.update(extra_checkpoint_data)

data_loader = make_data_loader(
    cfg,
    is_train=True,
    is_distributed=distributed,
    start_iter=arguments["iteration"],
)

checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD


import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from torch import cat
from apex import amp

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses
logger = logging.getLogger("maskrcnn_benchmark.trainer")
logger.info("Start training")
meters = MetricLogger(delimiter="  ")
max_iter = len(data_loader)
start_iter = arguments["iteration"]
model.train()
start_training_time = time.time()
end = time.time()
iteration = 0
data = iter(data_loader)
images, targets, _ = next(data)


data_time = time.time() - end
iteration = iteration + 1
arguments["iteration"] = iteration

scheduler.step()

images = images.to(device)
targets = [target.to(device) for target in targets]
with torch.no_grad():
    features = model.backbone(images.tensors)
    proposals, proposal_losses = model.rpn(images, features, targets)
    
    x, result, detector_losses = model.roi_heads(features, proposals, targets)
    # roi head debug
    
    # inside subsample
#    labels, regression_targets = model.roi_heads.box.loss_evaluator.prepare_targets(proposals, targets)
#    sampled_pos_inds, sampled_neg_inds = model.roi_heads.box.loss_evaluator.fg_bg_sampler(labels)
#    
#    proposals = list(proposals)
#    # add corresponding label and regression_targets information to the bounding boxes
#    for labels_per_image, regression_targets_per_image, proposals_per_image in zip(
#        labels, regression_targets, proposals
#    ):
#        proposals_per_image.add_field("labels", labels_per_image)
#        proposals_per_image.add_field(
#            "regression_targets", regression_targets_per_image
#        )

    # distributed sampled proposals, that were obtained on all feature maps
    # concatenated via the fg_bg_sampler, into individual feature map levels
#    for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
#        zip(sampled_pos_inds, sampled_neg_inds)
#    ):
#        img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
#        proposals_per_image = proposals[img_idx][img_sampled_inds]
#        proposals[img_idx] = proposals_per_image
#    
    proposals = model.roi_heads.box.loss_evaluator.subsample(proposals, targets, True)
        
    # checkout the match target   
    from maskrcnn_benchmark.modeling.utils import cat
    labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
    regression_targets = cat(
        [proposal.get_field("regression_targets") for proposal in proposals], dim=0
    )
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    
    
regression_targets[sampled_pos_inds_subset]