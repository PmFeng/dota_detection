# -*- coding: utf-8 -*-

import sys
import codecs
import numpy as np
import shapely.geometry as shgeo
import os
import re
import math
from tqdm import tqdm
from lxml.etree import Element, SubElement, tostring, ElementTree
from xml.etree.ElementTree import parse
import xml.etree.ElementTree as ET

import glob
from PIL import Image

path = '/home/sgiit/disk_2T/DataSet/rssrai2019_object_detection/data_split/val/labelTxt'
out_path = '/home/sgiit/disk_2T/DataSet/rssrai2019_object_detection/data_split/val/labelTxt_Rot/'
out_xml_path = '/home/sgiit/disk_2T/DataSet/rssrai2019_object_detection/data_split/val/labelxml_voc_4point/'
img_path = '/home/sgiit/disk_2T/DataSet/rssrai2019_object_detection/data_split/val/images/'

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
    file_name = file.split('/')[-1].split('.')[0]
    f = open(file, 'r')
    
#    file_write = out_path+file_name
#    f_w = open(file_write, 'w')
    
    file_data = f.readlines()
    
    Rot_data_lines = []
    Rot_data_clcs = []
    Rot_data_difficults = []
    Pt_4points = []
    img = Image.open(img_path + file_name + '.png')
    img = np.asarray(img)
    image_width = img.shape[0]
    image_height = img.shape[1]
    image_depth = img.shape[2]
    if len(file_data) == 0:
        f.close()
        continue
    else:
        for i in range(len(file_data)):
            
            bbox = file_data[i]
            bbox = bbox.split()
            pt = bbox[0 : 8]
            pt = tuple(map(float, pt))
            cls = bbox[8]
            difficult = bbox[9]
            Pt_4points.append(pt)
            Rot_pt = polygonToRotRectangle(pt)
            
            #Rot_data_line = [Rot_pt, cls, difficult]
            
            Rot_data_lines.append(Rot_pt)
            Rot_data_clcs.append(cls)
            Rot_data_difficults.append(difficult)
            
            
#        for b_label in Rot_data_lines:
#            Rot_pt_per, cls_per, difficult_per = b_label
#            f_w.write("{} {} {} {} {} ".format(*tuple(Rot_pt_per)))
#            f_w.write("{} {}\n".format(cls_per, difficult_per))
            
          
            
        root = ET.Element('annotation')
        ET.SubElement(root, 'folder').text = 'JPEGImages' # set correct folder name
        ET.SubElement(root, 'filename').text = file_name
        
        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(image_width)
        ET.SubElement(size, 'height').text = str(image_height)
        ET.SubElement(size, 'depth').text = str(image_depth)
        
        ET.SubElement(root, 'segmented').text = '0'
        
        for obj_4point, obj_cls, obj_pt, obj_dif in zip(Pt_4points, Rot_data_clcs, Rot_data_lines, Rot_data_difficults):
            name = obj_cls
            x1 = obj_4point[0]
            y1 = obj_4point[1]
            x2 = obj_4point[2]
            y2 = obj_4point[3]
            x3 = obj_4point[4]
            y3 = obj_4point[5]
            x4 = obj_4point[6]
            y4 = obj_4point[7]
            center_x = obj_pt[0]
            center_y = obj_pt[1]
            box_width = obj_pt[2]
            box_height = obj_pt[3]
            box_ang = obj_pt[4]
            
            obj = ET.SubElement(root, 'object')
            ET.SubElement(obj, 'name').text = name
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'occluded').text = '0'
            ET.SubElement(obj, 'difficult').text = obj_dif
            
            bx = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bx, 'center_x').text = str(center_x)
            ET.SubElement(bx, 'center_y').text = str(center_y)
            ET.SubElement(bx, 'box_width').text = str(box_width)
            ET.SubElement(bx, 'box_height').text = str(box_height)
            
            ET.SubElement(bx, 'x1').text = str(x1)
            ET.SubElement(bx, 'y1').text = str(y1)
            ET.SubElement(bx, 'x2').text = str(x2)
            ET.SubElement(bx, 'y2').text = str(y2)
            ET.SubElement(bx, 'x3').text = str(x3)
            ET.SubElement(bx, 'y3').text = str(y3)
            ET.SubElement(bx, 'x4').text = str(x4)
            ET.SubElement(bx, 'y4').text = str(y4)
            
            
        xml_file = file_name + '.xml'
        tree = ET.ElementTree(root)
        tree.write(out_xml_path+xml_file)
        f.close()    
            
    
            
        
            
            

    




