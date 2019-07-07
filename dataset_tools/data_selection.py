#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 00:35:41 2019

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

#from maskrcnn_benchmark.structures.bounding_box import BoxList

root = '/home/sgiit/disk_1T/sgiit/Pengming_Feng/Dataset/hrsc2016/HRSC2016/FullDataSet/'


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
                 "labels": anno["labels"], 
                 "difficult": anno["difficult"],
                 "theta": anno["theta"]}
        return label

    def _preprocess_annotation(self, target):
        boxes = []
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
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )
            theta = float(bb.find("box_ang").text)
            thetas.append(theta)
            boxes.append(bndbox)
            #gt_classes.append(self.class_to_ind[name])
            gt_classes.append(int(name)-100000000)
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": boxes,
            "labels": gt_classes,
            "difficult": difficult_boxes,
            "theta": thetas,
            "im_info": im_info,
        }
        return res
    
import tqdm
dataset = HRSC_Dataset(root, 'train_without_overlap_and_empty')

# count the label
count_r = 0

labels = []
index_labels = list(range(29))
class_labels = {}
for i, label in enumerate(labels):
    x = {label: i+1}
    class_labels.update(x)

for i in tqdm.tqdm(range(len(dataset))):
    img, label, idd = dataset.__getitem__(i)
    classes = label['labels'] 
    for id_class in classes:
        if id_class not in labels:
            count_r +=1
            labels.append(id_class)



# remove all the overlap label
count_r = 0

file2remove = []

for i in tqdm.tqdm(range(len(dataset))):
    

    img, label, idd = dataset.__getitem__(i)
    h, w, c = img.shape
    
    boxes, theta = label['boxes'], label['theta']
    out_flag = False
    for bbox, angle in zip(boxes, theta):
    #    pts = np.array(bbox, np.int32)
    #    rect_rota = pts.reshape(2,2)
    #    rect_rota.a
        rect_rota = ((bbox[0], bbox[1]), 
                    (bbox[2], bbox[3]), 
                    math.degrees(angle))
        
        box = cv.boxPoints(rect_rota)
        box = np.int0(box)
        
        for (x,y) in box:
            if x < 0 or y < 0 or x > w or y > h:
                out_flag = True
                
    if out_flag:
        count_r += 1
        file2remove.append(idd)
           
        
#        cv.drawContours(img,[box],0,(0,0,255),2)
    
#    cv.polylines(img,[box],True,(0,0,255),2)
    #cv.polylines(img,[pts], True, (0,0,255), thickness = 5)
    


#cv.imshow('image',img)
#cv.waitKey(0)   
#dataset = HRSC_Dataset(root, 'train')
#
#cv.imwrite('image_19_with_label.jpg', image)
    
file_names = [] 
for idd in file2remove:
    file_names.append(dataset.ids[idd])
count = 0  
f = open('train_without_overlap_and_empty.txt', 'w')
for idd in dataset.ids:
    if not idd in file_names:
        f.write(idd+'\n')
    else:
       count+=1 
f.close()



import glob
all_images = glob.glob(root+'/AllImages/*.bmp')
f = open('train_without_overlap.txt', 'w')
for image in all_images:
    file_name = image.split('/')[-1].split('.')[0]
    if not file_name in file_names:
        f.write(file_name+'\n')
    else:
       count+=1 
f.close()
    