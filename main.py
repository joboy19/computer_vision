import cv2
import argparse
import sys
import math
import numpy as np
import time as t
import os
from yolo import yolo_on_one_frame
from stereo_to_3d import stereo_to_3d, stereo_to_3d_wls

#info: use CLAHE histogram equalization on the input image and more equlaisation on the output of the density
### todo:
# - crop depth_map from the black bar on the side



def cropBottom(image):
    return image[:450]

def drawPred(image, class_name, distance, left, top, right, bottom, colour):
    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

    # construct labels
    label1 = class_name
    label2 = '%.2f m' % distance

    cv2.putText(image, label1, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(image, label2, (left, top+17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

def drawBoxes(frame, classes, boxes, distance_estimates):
    for x in enumerate(boxes):
        left = x[1][0]
        top = x[1][1]
        width = x[1][2]
        height = x[1][3]
        
        drawPred(frame, classes[x[0]], distance_estimates[x[0]], left, top, left + width, top + height, (255, 178, 50))



master_path_to_dataset = "C://Users//joebo//Documents//00uni//year3//vision//TTBB-durham-02-10-17-sub10"


#get left and right image paths
test_image = "1506943061.478682_L.png"

def imagePrepCLAHELAB(imgL): 
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)) 

    imgL_LAB = cv2.cvtColor(imgL, cv2.COLOR_BGR2LAB)
    imgL_LAB_Planes = cv2.split(imgL_LAB)
    imgL_LAB_Planes[0] = clahe.apply(imgL_LAB_Planes[0])
    imgL_LAB = cv2.merge(imgL_LAB_Planes)
    imgL_CLAHE = cv2.cvtColor(imgL_LAB, cv2.COLOR_LAB2BGR)

    return imgL_CLAHE

def imagePrepCLAHEGRAY(imgL): 
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)) 
    imgL_ = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgL_ = clahe.apply(imgL_)

    return imgL_


def yolo_and_depth(image_path):
    full_path_directory_left =  os.path.join(master_path_to_dataset, "left-images")
    full_path_directory_right =  os.path.join(master_path_to_dataset, "right-images")

    full_path_filename_left = os.path.join(full_path_directory_left, image_path)
    full_path_filename_right = os.path.join(full_path_directory_right, image_path.replace("_L", "_R"))

    #read the images in and crop the car from the bottom
    imgL = cropBottom(cv2.imread(full_path_filename_left))
    imgR = cropBottom(cv2.imread(full_path_filename_right))
 

    imgL_CLAHE = imagePrepCLAHEGRAY(imgL)
    imgR_CLAHE = imagePrepCLAHEGRAY(imgR)

    #get boxes of objects and their classes (filtered to relevant objects)
    classes, boxes = yolo_on_one_frame(imgL)

    #get the depth-map by calculating disaparity and converting to depth
    depth_map, depth_map2 = stereo_to_3d(imgL_CLAHE, imgR_CLAHE, max_disparity=128)

    distance_estimates = []





    for x in enumerate(boxes):
        left = x[1][0]
        top = x[1][1]
        width = x[1][2]
        height = x[1][3]
        cv2.waitKey()
        distance_estimates.append(np.median( depth_map[top:top+height, left:left+width] ))

    print(distance_estimates)


    drawBoxes(imgL, classes, boxes, distance_estimates)



    cv2.imshow("Yolo", imgL)
    cv2.imshow("Depth", depth_map2)

    cv2.waitKey()

def test(image_path):
    full_path_directory_left =  os.path.join(master_path_to_dataset, "left-images")
    full_path_directory_right =  os.path.join(master_path_to_dataset, "right-images")

    full_path_filename_left = os.path.join(full_path_directory_left, image_path)
    full_path_filename_right = os.path.join(full_path_directory_right, image_path.replace("_L", "_R"))

    #read the images in and crop the car from the bottom
    imgL = cropBottom(cv2.imread(full_path_filename_left))
    imgR = cropBottom(cv2.imread(full_path_filename_right))
 

    imgL_CLAHE = imagePrepCLAHEGRAY(imgL)
    imgR_CLAHE = imagePrepCLAHEGRAY(imgR)

    #get boxes of objects and their classes (filtered to relevant objects)
    classes, boxes = yolo_on_one_frame(imgL)

    #get the depth-map by calculating disaparity and converting to depth
    disparity_map = stereo_to_3d_wls(imgL_CLAHE, imgR_CLAHE, max_disparity=128)
    f = 399.9745178222656
    B = 0.2090607502
    depth_map = np.nan_to_num((f*B)/disparity_map, posinf=0, neginf=0)

    distance_estimates = []
    for x in enumerate(boxes):
        left = x[1][0]
        top = x[1][1]
        width = x[1][2]
        height = x[1][3]
        cv2.waitKey()
        distance_estimates.append(np.median( depth_map[top:top+height, left:left+width] ))

    print(distance_estimates)

    drawBoxes(imgL, classes, boxes, distance_estimates)

    cv2.imshow("Yolo", imgL)
    cv2.imshow("Depth", depth_map/256)

    cv2.waitKey()

#yolo_and_depth("1506943061.478682_L.png")

test("1506943061.478682_L.png")