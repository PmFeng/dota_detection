#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 22:09:09 2019

@author: sgiit
"""


import os

import torch
import torch.utils.data
from PIL import Image
import sys
from maskrcnn_benchmark.structures.bounding_box import BoxList
import cv2 as cv
import math
import numpy as np

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
    


root = '/home/pengming/HRSC2016/FullDataSet'
class DOTA_Dataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background__ ",
        "large-vehicle",
        "swimming-pool",
        "helicopter",
        "bridge",
        "plane",
        "ship",
        "soccer-ball-field",
        "basketball-court",
        "airport",
        "container-crane",
        "ground-track-field",
        "small-vehicle",
        "harbor",
        "baseball-diamond",
        "tennis-court",
        "roundabout",
        "storage-tank",
        "helipad",
    )

    def __init__(self, data_dir, split, use_difficult=False, transforms=None):
        self.root = data_dir
        self.image_set = split
        if split == 'train':
            self.keep_difficult = True
        else:
            self.keep_difficult = use_difficult
        self.transforms = transforms

        self._annopath = os.path.join(self.root, "labelxml_voc", "%s.xml")
        self._imgpath = os.path.join(self.root, "images", "%s.png")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "%s.txt")

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = DOTA_Dataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        
        height, width = anno["im_info"]
        target = BoxList(anno["hrbb_boxes"], (width, height), mode="xyxy")        
        obb_target = BoxList(anno["obb_boxes"], (width, height), mode="xywh")
        target.add_field("obb_boxes", obb_target)   
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        target.add_field("theta", anno["theta"])
        

        return target
        
#        label = {"im_info": anno["im_info"], 
#                 "boxes": anno["boxes"], 
#                 "labels": anno["labels"], 
#                 "difficult": anno["difficult"],
#                 "theta": anno["theta"]}
#        return label

    def _preprocess_annotation(self, target):
        #boxes = []
        obb_boxes = []
        hbb_boxes = []
        gt_classes = []
        difficult_boxes = []
        thetas = []
        TO_REMOVE = 1
        
        

        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
#            HBB_box = [
#                bb.find("xmin").text,
#                bb.find("ymin").text,
#                bb.find("xmax").text,
#                bb.find("ymax").text,
#            ]
            
            OBB_box = [
                float(bb.find("center_x").text),
                float(bb.find("center_y").text),
                float(bb.find("box_width").text),
                float(bb.find("box_height").text),
            ]
            
            theta = float(bb.find("box_ang").text)
            
            
            
            hbb_box = ((OBB_box[0], OBB_box[1]),
                       (OBB_box[2], OBB_box[3]),
                       math.degrees(theta))
            hbb_box = cv.boxPoints(hbb_box)
            hbb_box = np.int0(hbb_box)
            
            pt_x_y_min = hbb_box.min(axis= 0)
            pt_x_y_max = hbb_box.max(axis= 0)
            
            hrbb_box = np.hstack((pt_x_y_min, pt_x_y_max))
            

            
            hrbb_bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, hrbb_box)))
            )
            obb_bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, OBB_box)))
            )
            
            obb_bndbox = [OBB_box[0]-OBB_box[2]/2,
                          OBB_box[1]-OBB_box[3]/2,
                          OBB_box[2], OBB_box[3]]
            
            
            
            
            theta = float(bb.find("box_ang").text)
            
            theta = math.degrees(theta)
            
            thetas.append(theta)
            
            
            obb_boxes.append(obb_bndbox)
            hbb_boxes.append(hrbb_bndbox)
            #gt_classes.append(self.class_to_ind[name])
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)
            
            
        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "obb_boxes": torch.tensor(obb_boxes, dtype=torch.float32),
            "hrbb_boxes": torch.tensor(hbb_boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "theta": torch.tensor(thetas, dtype=torch.float32),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return DOTA_Dataset.CLASSES[class_id]
    
root = "/home/sgiit/disk_2T/DataSet/rssrai2019_object_detection/data_split_800/train"

dataset = DOTA_Dataset(root, 'train', use_difficult=True)
names_count = {}
import tqdm
for i in tqdm.tqdm(range(len(dataset))):
    label = dataset.get_groundtruth(i)
    names = label.get_field('labels')
    for name in names:
        id_ = dataset.CLASSES[name]
        if id_ in names_count.keys():
            names_count[id_] += 1
        else:
            names_count[id_] = 1

import matplotlib.pyplot as plt
#%matplotlib qt 

plt.bar(list(names_count.keys()), names_count.values(), color='g')
plt.show()   