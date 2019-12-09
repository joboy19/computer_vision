#######
####### Ensure that opencv-contrib is installed
#######

import cv2
import argparse
import sys
import math
import numpy as np
import time as t
import os
from yolo import yolo_on_one_frame
from stereo_to_3d import stereo_to_3d_wls



master_path_to_dataset = "C://Users//joebo//Documents//00uni//year3//vision//TTBB-durham-02-10-17-sub10"

#crops image bottom (to remove car)
def cropBottom(image):
    return image[:450]

#modified draw pred to take class name and distance, modified label and text made smaller
def drawPred(image, class_name, distance, left, top, right, bottom, colour):
    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

    # construct labels
    if class_name != "person":
        label = '%s @ %.2f m' % ("vehicle", distance)
    else:
        label = '%s @ %.2f m' % ("person", distance)


    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    top = max(top, labelSize[1])

    cv2.rectangle(image, (left, top - round(1.5*labelSize[1])),
        (left + round(1.5*labelSize[0]) - 20, top + baseLine), (255, 255, 255), cv2.FILLED)

    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1)

#runs drawPred for each box
def drawBoxes(frame, classes, boxes, distance_estimates):
    for x in enumerate(boxes):
        left = x[1][0]
        top = x[1][1]
        width = x[1][2]
        height = x[1][3]
        drawPred(frame, classes[x[0]], distance_estimates[x[0]], left, top, left + width, top + height, (255, 178, 50))


#takes an image and runs CLAHE on the light channel of the image in LAB colour space
def imagePrepCLAHELAB(imgL): 
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)) 

    imgL_LAB = cv2.cvtColor(imgL, cv2.COLOR_BGR2LAB)
    imgL_LAB_Planes = cv2.split(imgL_LAB)
    imgL_LAB_Planes[0] = clahe.apply(imgL_LAB_Planes[0])
    imgL_LAB = cv2.merge(imgL_LAB_Planes)
    imgL_CLAHE = cv2.cvtColor(imgL_LAB, cv2.COLOR_LAB2BGR)

    return imgL_CLAHE

#takes a grayscale image and runs CLAHE on it
def imagePrepCLAHEGRAY(imgL): 
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)) 
    imgL_ = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgL_ = clahe.apply(imgL_)

    return imgL_



#slice based on percentile
def betterSlice(a):
    a = np.sort(a.flatten())
    return a[int(len(a)*0.2):int(len(a)*0.5)]

#slice based on heuristic
def betterSlice2(a, name):
    if name == "person":
        print("here")
        return a[int(len(a)*0.5):int(len(a)*0.8), int(len(a[1])*0.3):int(len(a[1])*0.7)]
    else:
        return a[int(len(a)*0.2):int(len(a)*0.5)]


def test(image_path):
    #set-up directorys for the image stream
    full_path_directory_left =  os.path.join(master_path_to_dataset, "left-images")
    full_path_directory_right =  os.path.join(master_path_to_dataset, "right-images")

    full_path_filename_left = os.path.join(full_path_directory_left, image_path)
    full_path_filename_right = os.path.join(full_path_directory_right, image_path.replace("_L", "_R"))

    time1 = t.time()

    #read the images in and crop the car from the bottom
    imgL = cropBottom(cv2.imread(full_path_filename_left))
    imgR = cropBottom(cv2.imread(full_path_filename_right))

    #crop to adjust for the black bar
    yolo_imgL = np.copy(imgL)[:, 128:]



    #imgL = cv2.bilateralFilter(imgL, 10, 16, 32)
    #imgR = cv2.bilateralFilter(imgR, 10, 16, 32)


    #run the preprocessing CLAHE on both images
    imgL_CLAHE = imagePrepCLAHELAB(imgL)
    imgR_CLAHE = imagePrepCLAHELAB(imgR)

    #print("image pre-process time:", t.time() - time1)
    #time1 = t.time()

    #get boxes of objects and their classes (filtered to relevant objects)
    classes, boxes = yolo_on_one_frame(cv2.medianBlur(yolo_imgL, 5))
    #classes, boxes = yolo_on_one_frame(yolo_imgL)

    #print("object detection:", t.time() - time1)
    #time1 = t.time()

    #get the depth-map by calculating disaparity, using a wls filter, and converting to depth
    disparity_map = stereo_to_3d_wls(imgL_CLAHE, imgR_CLAHE, max_disparity=128)[:, 128:]
    f = 399.9745178222656
    B = 0.2090607502
    depth_map = np.nan_to_num((f*B)/disparity_map, nan=0, posinf=0, neginf=0)



    #print("dispartity and distance calculation:", t.time() - time1)
    #time1 = t.time()

    distance_estimates = []
    
    #run through each box to get the distance estimate at this point
    for x in enumerate(boxes):
        left = max(0, x[1][0]) # make sure the box doesnt run off the end, else slicing returns nothing and nan is returned when calcing the mean
        top = x[1][1]
        width = x[1][2]
        height = x[1][3]
        vals = depth_map[top:top+height, left:left+width]
        a = np.mean(betterSlice(vals))
        z = np.mean(betterSlice2(vals, classes[x[0]]))
        b = np.mean(vals)
        c = np.median(vals)
        
        distance_estimates.append(a)
    

    #print(distance_estimates)

    #draw the boxes and show the images
    drawBoxes(yolo_imgL, classes, boxes, distance_estimates)

    #print("distance estimation and drawing:", t.time() - time1)
    time1 = t.time()

    cv2.imshow("Image", yolo_imgL)

    cv2.imshow("Disparity Adjusted", cv2.equalizeHist(disparity_map))
    
    try:
        min_ = str(min(distance_estimates))
        print("running on image: \n" + image_path + "\n" + image_path.replace("_L", "_R") + " : nearest detected scene object " + min_ + "m" )
    except:
        print("running on image: \n" + image_path + "\n" + image_path.replace("_L", "_R"))
    
    

    #pause using "p"
    key = cv2.waitKey(100);
    if key == ord("p"):
        while True:
            key2 = cv2.waitKey()
            if key2 == ord("p"):
                break

    return imgL, depth_map


def all_files():
    for _, _, filen in os.walk(os.path.join(master_path_to_dataset, "left-images")):    
        for x in filen:
            time1 = t.time()
            image, depth = test(x)
            print("overall time for frame:", t.time() - time1)
            print("")
            print("")
        


all_files()


