import cv2
import os
import numpy as np
import random
import csv

#run SGBM and WLS filter on L/R pair, with max_disparity set at 128 default
def stereo_to_3d_wls(imgL, imgR, max_disparity=128):
    window_size = 5             
    
    #set up stereo sgbm with parameters as such
    #http://timosam.com/python_opencv_depthimage for the tutorial and code
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

    #create the wls (Weighted Least Squares) filter object to run the filter, setting the parameters
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(80000)
    wls_filter.setSigmaColor(1.2)

    #filter the image using the wls filter and scale
    filteredImg = wls_filter.filter(displ, imgL, None, dispr) 
    disparity_scaled = (filteredImg / 16).astype(np.uint8)

    return disparity_scaled


