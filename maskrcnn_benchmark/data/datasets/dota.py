#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 10:58:51 2019

@author: sgiit
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 11:58:50 2019

@author: pengming
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
    



#from maskrcnn_benchmark.structures.bounding_box import BoxList

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

#        height, width = anno["im_info"]
#        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
#        target.add_field("labels", anno["labels"])
#        target.add_field("difficult", anno["difficult"])
        
#        img_id = self.ids[index]
#        anno = ET.parse(self._annopath % img_id).getroot()
#        anno = self._preprocess_annotation(anno)
#
#        height, width = anno["im_info"]
#        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
#        target.add_field("labels", anno["labels"])
#        target.add_field("difficult", anno["difficult"])
#        return target
        
#        if anno["hrbb_boxes"].size(0) > 700:
#            fix_ind = np.random.randint(0, high=anno["hrbb_boxes"].size(0), size=700, dtype='l')
#            anno["hrbb_boxes"] = anno["hrbb_boxes"][fix_ind]
#            anno["obb_boxes"] = anno["obb_boxes"][fix_ind]
#            anno["labels"] = anno["labels"][fix_ind]
#            anno["difficult"] = anno["difficult"][fix_ind]
#            anno["theta"] = anno["theta"][fix_ind]
        
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
            
            if theta > 90.0:
                theta -= 180
            elif theta < -90.0:
                theta += 180
            elif theta == -90.0:
                theta = 90
            
            
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
    
    
#for index in range(len(data_loader.dataset.ids)):
#    img_id = data_loader.dataset.ids[index]
#    anno = ET.parse(data_loader.dataset._annopath % img_id).getroot()  
#    anno = data_loader.dataset._preprocess_annotation(anno)
#    height, width = anno["im_info"]
#    target = BoxList(anno["hrbb_boxes"], (width, height), mode="xyxy")        
#    obb_target = BoxList(anno["obb_boxes"], (width, height), mode="xywh")
#    target.add_field("obb_boxes", obb_target)   
#    target.add_field("labels", anno["labels"])
#    target.add_field("difficult", anno["difficult"])
#    target.add_field("theta", anno["theta"])