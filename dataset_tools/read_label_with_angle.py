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

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import cv2 as cv

import numpy as np
import math
import tqdm
#from maskrcnn_benchmark.structures.bounding_box import BoxList



def hrbb_anchor2obb_anchor(proposal, angle):
    
    hrbb_x_min = [proposal[0] - proposal[2]/2]
    hrbb_y_min = [proposal[1] - proposal[3]/2]
    hrbb_x_max = [proposal[0] + proposal[2]/2]
    hrbb_y_max = [proposal[1] + proposal[3]/2]   
    
    if angle < 0:
        angle = abs(angle)
    else:
        angle = 90 - angle
        
    h = (proposal[3] - np.tan(angle/180*np.pi)*proposal[2])/(1-(np.tan(angle/180*np.pi))**2)
    w = h * np.tan(angle/180*np.pi)
    
    
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
    
    
    
    return obb_bbox
    




root = '/home/sgiit/disk_1T/sgiit/Pengming_Feng/Dataset/hrsc2016/HRSC2016/FullDataSet'
class HRSC_Dataset(torch.utils.data.Dataset):


    def __init__(self, data_dir, split, use_difficult=False, transforms=None):
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms

        self._annopath = os.path.join(self.root, "Annotations_voc", "%s.xml")
        self._imgpath = os.path.join(self.root, "AllImages", "%s.bmp")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "%s.txt")

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

    def __getitem__(self, index):
        img_id = self.ids[index]
#        img = Image.open(self._imgpath % img_id).convert("RGB")

        image = cv.imread(self._imgpath % img_id)

        target = self.get_groundtruth(index)
#        target = target.clip_to_image(remove_empty=True)


        return image, target, index

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
        
#        height, width = anno["im_info"]
#        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
#        target.add_field("labels", anno["labels"])
#        target.add_field("difficult", anno["difficult"])
#        target.add_field("theta", anno["theta"])
#        
#        return target
        
        label = {"im_info": anno["im_info"], 
                 "boxes": anno["boxes"], 
                 "rc_boxes": anno["rc_boxes"],
                 "labels": anno["labels"], 
                 "difficult": anno["difficult"],
                 "theta": anno["theta"]}
        return label

    def _preprocess_annotation(self, target):
        boxes = []
        rc_boxes = []
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
            box = [
#                bb.find("xmin").text,
#                bb.find("ymin").text,
#                bb.find("xmax").text,
#                bb.find("ymax").text,
                float(bb.find("center_x").text),
                float(bb.find("center_y").text),
                float(bb.find("box_width").text),
                float(bb.find("box_height").text),
            ]
            rc_box = [
                float(bb.find("xmin").text),
                float(bb.find("ymin").text),
                float(bb.find("xmax").text),
                float(bb.find("ymax").text),
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )
            rc_box = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, rc_box)))
            )
            theta = float(bb.find("box_ang").text)
            thetas.append(theta)
            boxes.append(bndbox)
            rc_boxes.append(rc_box)
            #gt_classes.append(self.class_to_ind[name])
            gt_classes.append(int(name)-100000000)
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": boxes,
            "rc_boxes": rc_boxes,
            "labels": gt_classes,
            "difficult": difficult_boxes,
            "theta": thetas,
            "im_info": im_info,
        }
        return res
    
    
dataset = HRSC_Dataset(root, 'train_without_overlap_and_empty')

img, label, idd = dataset.__getitem__(89)

boxes, theta = label['boxes'], label['theta']

all_theta = []
all_len = []
for data in dataset:
    img, label, idd = data
    boxes, theta = label['boxes'], label['theta']
    h, w, _ = img.shape
    max_len = max(w, h)
    all_len.append(max_len)
#    for bbox, angle in zip(boxes, theta):
#        all_theta.append(math.degrees(angle))
    
    
    
size_data = len(dataset)
i = 0
while i < size_data:   
    img, label, idd = dataset.__getitem__(i)
    boxes, theta, rc_boxes= label['boxes'], label['theta'], label['rc_boxes']
    for bbox, angle, rc_box in zip(boxes, theta, rc_boxes):
    #    pts = np.array(bbox, np.int32)
    #    rect_rota = pts.reshape(2,2)
    #    rect_rota.a
        rect_rota = ((bbox[0], bbox[1]), 
                    (bbox[2], bbox[3]), 
                    math.degrees(angle))
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
        label = "degree: {}".format(math.degrees(angle))
        cv.putText(img, label, (rc_box[0], rc_box[1] - 2), 0, 1, [225, 255, 255])
        cv.rectangle(img, (big_box[0], big_box[1]), (big_box[2], big_box[3]), (0,255,0), 2)
        box_recover = hrbb_anchor2obb_anchor(big_box_xywh, math.degrees(angle))
        cv.drawContours(img,[box_recover],0,(0,0,255),2)
        
        #cv.rectangle(img, (rc_box[0], rc_box[1]), (rc_box[2], rc_box[3]), (0,255,0), 2)
        #cv.rectangle(img, (rc_tota[0], rc_tota[1]), (rc_tota[2], rc_tota[3]), (255,0,0), 2)
        
    #    cv.polylines(img,[box],True,(0,0,255),2)
        #cv.polylines(img,[pts], True, (0,0,255), thickness = 5)
        
    
    
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
#dataset = HRSC_Dataset(root, 'train')
#
#cv.imwrite('image_19_with_label.jpg', image)