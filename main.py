import cv2
import argparse
import sys
import math
import numpy as np
import time as t
from yolo import yolo_on_one_frame
from stereo_to_3d import stereo_to_3d

#info: use CLAHE histogram equalization on the input image and more equlaisation on the output of the density
master_path_to_dataset = "D://Users//Joe//Documents//00uni//coursework and notes//ear3//ssa//visionTTBB-durham-02-10-17-sub10"


test_img = "left-images//1506942473.484027_L.png"

def cropBottom(image):
    return image[:450]

def drawPred(image, class_name, confidence, left, top, right, bottom, colour):
    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

    # construct label
    label = '%s:%.2f' % (class_name, confidence)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(image, (left, top - round(1.5*labelSize[1])),
        (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

def drawBoxes(frame, classes, boxes):
    for x in enumerate(boxes):
        left = x[1][0]
        top = x[1][1]
        width = x[1][2]
        height = x[1][3]
        
        drawPred(frame, classes[x[0]], 0, left, top, left + width, top + height, (255, 178, 50))

#get img R:

imgR = cv2.imread()

#crop bottom to remove the car
test_img = cropBottom(test_img)

#get boxes
classes, boxes = yolo_on_one_frame(test_img)



distance_values =  stereo_to_3d(master_path_to_dataset, test_img, max_disp=128)



#draw boxes
drawBoxes(test_img, classes, boxes)

cv2.imshow("Yolo", test_img)
cv2.waitKey()