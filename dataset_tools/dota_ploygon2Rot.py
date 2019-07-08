# -*- coding: utf-8 -*-

import sys
import codecs
import numpy as np
import shapely.geometry as shgeo
import os
import re
import math
from tqdm import tqdm


import glob

path = '/home/sgiit/disk_2T/DataSet/rssrai2019_object_detection/data_split_800/val/labelTxt'
out_path = '/home/sgiit/disk_2T/DataSet/rssrai2019_object_detection/data_split_800/val/labelTxt_voc/'

file_list  = glob.glob(path + '/*.txt')



def polygonToRotRectangle(bbox):
    """
    :param bbox: The polygon stored in format [x1, y1, x2, y2, x3, y3, x4, y4]
    :return: Rotated Rectangle in format [cx, cy, w, h, theta]
    """
    bbox = np.array(bbox,dtype=np.float32)
    bbox = np.reshape(bbox,newshape=(2,4),order='F')
    angle = math.atan2(-(bbox[0,1]-bbox[0,0]),bbox[1,1]-bbox[1,0])

    center = [[0],[0]]

    for i in range(4):
        center[0] += bbox[0,i]
        center[1] += bbox[1,i]

    center = np.array(center,dtype=np.float32)/4.0

    R = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)

    normalized = np.matmul(R.transpose(),bbox-center)

    xmin = np.min(normalized[0,:])
    xmax = np.max(normalized[0,:])
    ymin = np.min(normalized[1,:])
    ymax = np.max(normalized[1,:])

    w = xmax - xmin + 1
    h = ymax - ymin + 1

    return [float(center[0]),float(center[1]),w,h,angle]



for file in tqdm(file_list):
    file_name = file.split('/', -1)[-1]
    f = open(file, 'r')
    
    file_write = out_path+file_name
    f_w = open(file_write, 'w')
    
    file_data = f.readlines()
    
    Rot_data_lines = []
    
    if len(file_data) == 0:
        f.close()
        f_w.close()
        continue
    else:
        for i in range(len(file_data)):
            
            bbox = file_data[i]
            bbox = bbox.split()
            pt = bbox[0 : 8]
            pt = tuple(map(float, pt))
            cls = bbox[8]
            difficult = bbox[9]
            
            Rot_pt = polygonToRotRectangle(pt)
            
            Rot_data_line = [Rot_pt, cls, difficult]
            
            Rot_data_lines.append(Rot_data_line)
            
            
        for b_label in Rot_data_lines:
            Rot_pt_per, cls_per, difficult_per = b_label
            f_w.write("{} {} {} {} {} ".format(*tuple(Rot_pt_per)))
            f_w.write("{} {}\n".format(cls_per, difficult_per))
            
            
    f.close()
    f_w.close()
            
        
            
            

    




