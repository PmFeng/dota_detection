#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 00:36:54 2019

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

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import cv2 as cv

import numpy as np
import math
import tqdm
#from maskrcnn_benchmark.structures.bounding_box import BoxList

def hrbb_anchor2obb_anchor_0_180(proposal, angle):
    
    hrbb_x_min = [proposal[0] - proposal[2]/2]
    hrbb_y_min = [proposal[1] - proposal[3]/2]
    hrbb_x_max = [proposal[0] + proposal[2]/2]
    hrbb_y_max = [proposal[1] + proposal[3]/2]   
    
    if angle <= 90:
        w = (proposal[2] - np.tan(angle/180*np.pi)*proposal[3])/(1-(np.tan(angle/180*np.pi))**2)
        h = w * np.tan(angle/180*np.pi)
        if h > proposal[3] or w > proposal[2]:
            w = (proposal[2] - np.tan((90-angle)/180*np.pi)*proposal[3])/(1-(np.tan((90-angle)/180*np.pi))**2)
            h = 2 * np.tan(angle/180*np.pi)
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
        angle = 180 - angle
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
    
    
    
    return obb_bbox#, h, w

def hrbb_anchor2obb_anchor(proposal, angle):
    
    hrbb_x_min = [proposal[0] - proposal[2]/2]
    hrbb_y_min = [proposal[1] - proposal[3]/2]
    hrbb_x_max = [proposal[0] + proposal[2]/2]
    hrbb_y_max = [proposal[1] + proposal[3]/2]   
    
    if angle < 0:
        angle = 90 + angle
        h = (proposal[3] - np.tan(angle/180*np.pi)*proposal[2])/(1-(np.tan(angle/180*np.pi))**2)
        w = h * np.tan(angle/180*np.pi)
        if abs(h) > proposal[3] or abs(w) > proposal[2]:
            h = (proposal[3] - np.tan((90-angle)/180*np.pi)*proposal[2])/(1-(np.tan((90-angle)/180*np.pi))**2)
            w = h * np.tan((90-angle)/180*np.pi)
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
        if abs(h) > proposal[3] or abs(w) > proposal[2]:
            h = (proposal[3] - np.tan((90-angle)/180*np.pi)*proposal[2])/(1-(np.tan((90-angle)/180*np.pi))**2)
            w = h * np.tan((90-angle)/180*np.pi)
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
    
    
    
    return obb_bbox#, h, w
    




root = "/home/sgiit/disk_2T/DataSet/rssrai2019_object_detection/data_split/val"
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
        #img = Image.open(self._imgpath % img_id).convert("RGB")
        img = cv.imread(self._imgpath % img_id)

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
                if theta > -46 and theta < -44:
                    theta = -42
            elif theta < -90.0:
                theta += 180
                if theta < 46 and theta > 44:
                    theta = 48
            elif theta == -90.0:
                theta = 90
            elif theta < 46 and theta > 44:
                    theta = 48
            elif theta > -46 and theta < -44:
                    theta = -42
            
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

    
    
dataset = DOTA_Dataset(root, 'val', use_difficult=True)

#img, label, idd = dataset.__getitem__(89)
#
#boxes, theta = label['boxes'], label['theta']
#
#all_theta = []
#all_len = []
#for data in dataset:
#    img, label, idd = data
#    boxes, theta = label['boxes'], label['theta']
#    h, w, _ = img.shape
#    max_len = max(w, h)
#    all_len.append(max_len)
#    for bbox, angle in zip(boxes, theta):
#        all_theta.append(math.degrees(angle))
    
    
    
size_data = len(dataset)
i = 0
while i < size_data:   
    img, label, idd = dataset.__getitem__(i)
    if label == None:
        continue
    boxes, theta, rc_boxes=  label.get_field('obb_boxes').bbox.numpy(), label.get_field('theta').numpy(), label.bbox.numpy()
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
        cv.drawContours(img,[box],0,(0,0,255),2)
        
        
    
        label = "d:{:.1f}".format(angle)
        cv.putText(img, label, (rc_box[0], rc_box[1] - 2), 0, 0.5, [225, 255, 255], 2)
        cv.rectangle(img, (big_box[0], big_box[1]), (big_box[2], big_box[3]), (0,255,0), 2)
        if i == 19:
            label = "H:{},W:{}".format(bbox[3], bbox[2])
            cv.putText(img, label, (big_box[0]+ 20, big_box[1] + 80), 0, 0.5, [225, 255, 255], 2)
        
        
        box_recover = hrbb_anchor2obb_anchor(big_box_xywh, angle)
        cv.drawContours(img,[box_recover],0,(0,255,255),2)
        
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