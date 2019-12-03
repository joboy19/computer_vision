#####################################################################

# Example : project SGBM disparity to 3D points for am example pair
# of rectified stereo images from a  directory structure
# of left-images / right-images with filesname DATE_TIME_STAMP_{L|R}.png

# basic illustrative python script for use with provided stereo datasets

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2017 Deparment of Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import os
import numpy as np
import random
import csv

master_path_to_dataset = "D:/Users/Joe/Documents/00uni/coursework and notes/year3/ssa/vision/TTBB-durham-02-10-17-sub10" # ** need to edit this **
directory_to_cycle_left = "left-images"     # edit this if needed
directory_to_cycle_right = "right-images"   # edit this if needed

#####################################################################

# fixed camera parameters for this stereo setup (from calibration)

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

image_centre_h = 262.0;
image_centre_w = 474.5;

#####################################################################

## project_disparity_to_3d : project a given disparity image
## (uncropped, unscaled) to a set of 3D points with optional colour

def project_disparity_to_3d(disparity, max_disparity, imgL):
    values_out = np.zeros((len(imgL), len(imgL[0]), 1))
    values_out2 = np.zeros((len(imgL), len(imgL[0]), 1))

    points = [];

    f = camera_focal_length_px;
    B = stereo_camera_baseline_m;

    height, width = disparity.shape[:2];

    # assume a minimal disparity of 2 pixels is possible to get Zmax
    # and then we get reasonable scaling in X and Y output if we change
    # Z to Zmax in the lines X = ....; Y = ...; below

    # Zmax = ((f * B) / 2);

    for y in range(height): # 0 - height is the y axis index
        for x in range(width): # 0 - width is the x axis index

            # if we have a valid non-zero disparity

            if (disparity[y,x] > 0):

                # calculate corresponding 3D point [X, Y, Z]

                # stereo lecture - slide 22 + 25

                Z = (f * B) / disparity[y,x];

                X = ((x - image_centre_w) * Z) / f;
                Y = ((y - image_centre_h) * Z) / f;

                values_out[y][x] = Z
                values_out2[y][x] = Z / 256

    #1 is the depthmap, 2 is the depthmap so you can view it
    return values_out, values_out2;

#####################################################################

# project a set of 3D points back the 2D image domain

def project_3D_points_to_2D_image_points(points):

    points2 = [];

    # calc. Zmax as per above

    # Zmax = (camera_focal_length_px * stereo_camera_baseline_m) / 2;

    for i1 in range(len(points)):

        # reverse earlier projection for X and Y to get x and y again

        x = ((points[i1][0] * camera_focal_length_px) / points[i1][2]) + image_centre_w;
        y = ((points[i1][1] * camera_focal_length_px) / points[i1][2]) + image_centre_h;
        points2.append([x,y]);

    return points2;

#####################################################################

def stereo_to_3d(imgL, imgR, max_disparity=128):

    # remember to convert to grayscale (as the disparity matching works on grayscale)
    # N.B. need to do for both as both are 3-channel images
    try:
        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY);
    except:
        grayL = imgL
    try:
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY);
    except:
        gray = imgR

    # compute disparity image from undistorted and rectified stereo images
    # that we have loaded
    # (which for reasons best known to the OpenCV developers is returned scaled by 16)
    stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);

    disparity = stereoProcessor.compute(grayL,grayR);

    # filter out noise and speckles (adjust parameters as needed)

    dispNoiseFilter = 100; # increase for more agressive filtering
    cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);

    # scale the disparity to 8-bit for viewing
    # divide by 16 and convert to 8-bit image (then range of values should
    # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
    # so we fix this also using a initial threshold between 0 and max_disparity
    # as disparity=-1 means no disparity available

    _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO);
    disparity_scaled = (disparity / 16.).astype(np.uint8);

    # display image (scaling it to the full 0->255 range based on the number
    # of disparities in use for the stereo part)

    ##cv2.imshow("disparity", (disparity_scaled * (255. / max_disparity)).astype(np.uint8));

    # project to a 3D colour point cloud (with or without colour)

    # points = project_disparity_to_3d(disparity_scaled, max_disparity);
    values_out = project_disparity_to_3d(disparity_scaled, max_disparity, imgL);

    return values_out
    # close all windows

def stereo_to_3d_wls(imgL, imgR, max_disparity=128):
    window_size = 15                 
    
    #set up stereo sgbm with parameters as such
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=max_disparity,
        blockSize=5,
        P1=8 * 3 * window_size ** 2,    
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    #create the right matcher for the right-view disparity match, based on the left-view's parameters
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    visual_multiplier = 1.0
    
    #compute the disparity map and ensure they're using the right data type
    displ = left_matcher.compute(imgL, imgR)  
    dispr = right_matcher.compute(imgR, imgL)  
    displ = np.int16(displ)
    dispr = np.int16(dispr)

    #create the wls (Weighted Least Squares) filter object to run the filter
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(80000)
    wls_filter.setSigmaColor(1.2)

    #filter the image using the wls filter
    filteredImg = wls_filter.filter(displ, imgL, None, dispr) 

    _, disparity = cv2.threshold(filteredImg,0, max_disparity * 16, cv2.THRESH_TOZERO);
    disparity_scaled = (disparity / 16.).astype(np.uint8)

    return disparity_scaled

#####################################################################
