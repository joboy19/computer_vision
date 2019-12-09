import cv2
import argparse
import sys
import math
import numpy as np


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

### init YOLO CNN object detection model
##
confThreshold = 0.7  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image
##
### Load names of classes from file
##
classes = None
with open("./coco.names", 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

#load configuration and weight files for the model and load the network using them
net = cv2.dnn.readNetFromDarknet("./yolov3.cfg", "./yolov3.weights")
output_layer_names = getOutputsNames(net)

#set net backend and target
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



def postprocess(image, results, threshold_confidence, threshold_nms):
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]

    classIds = []
    confidences = []
    boxes = []

    # Scan through all the bounding boxes output from the network and..
    # 1. keep only the ones with high confidence scores.
    # 2. assign the box class label as the class with the highest score.
    # 3. construct a list of bounding boxes, class labels and confidence scores

    classIds = []
    confidences = []
    boxes = []
    for result in results:
        for detection in result:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > threshold_confidence:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences
    classIds_nms = []
    confidences_nms = []
    boxes_nms = []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold_confidence, threshold_nms)
    for i in indices:
        i = i[0]
        classIds_nms.append(classIds[i])
        confidences_nms.append(confidences[i])
        boxes_nms.append(boxes[i])

    # return post processed lists of classIds, confidences and bounding boxes
    return (classIds_nms, confidences_nms, boxes_nms)

#run yolo on a single frame, returning the boxes and classes for filtered images
def yolo_on_one_frame(frame):

    #prep tensor and net with information from the 
    tensor = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    net.setInput(tensor)

    #run the net and post-process the output
    results = net.forward(output_layer_names)
    classIDs, confidences, boxes = postprocess(frame, results, confThreshold, nmsThreshold)

    out_class = [] 
    out_boxes = []
    for x in enumerate(classIDs):
        if classes[x[1]] in ["person", "car", "motorbike", "bus", "truck", "bicycle"]:
            out_class.append(classes[x[1]])
            out_boxes.append(boxes[x[0]])
    return out_class, out_boxes



    
