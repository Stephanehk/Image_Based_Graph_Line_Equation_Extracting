#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 13:31:45 2018

@author: 2020shatgiskessell
"""

import cv2
import numpy as np

img = cv2.imread("/Users/2020shatgiskessell/Desktop/Graph7.jpg")
h,w = img.shape[:2]
mask = np.zeros((h,w), np.uint8)
mask2 = mask = np.zeros((h,w), np.uint8)

def find_contours(image):
    # Transform to gray colorspace and threshold the image
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # erod then dialate image (for denoising)
    kernel = np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    #Find contours in order of hiarchy
    #CHAIN_APPROX_NONE gives all the points on the contour
    _, contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    return contours

#----------------------------------------------------------------------------------------------------------------
#CLEAN UP IMAGE AND JUST EXTRACT LINE

#get the biggest contour
cnt = max(find_contours(img), key=cv2.contourArea)
cv2.drawContours(mask, [cnt], 0, 255, -1)

# Perform a bitwise operation
res = cv2.bitwise_and(img, img, mask=mask)

# Threshold the image again
gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


# Find all non white pixels
non_zero = cv2.findNonZero(thresh)

# Transform all other pixels in non_white to white
for i in range(0, len(non_zero)):
    first_x = non_zero[i][0][0]
    first_y = non_zero[i][0][1]
    first = res[first_y, first_x]
    res[first_y, first_x] = 255


# Display the image
#cv2.imwrite("graph3.jpg", res)
#----------------------------------------------------------------------------------------------------------------
#GET CONTOUR OF LINE
i = 0
#figure out how to select the right contour
#get contour with largest area
# if area > certain amount
#   get next largest contour
for contour in find_contours(res):
    cv2.drawContours(mask2, [contour], 0, 255, -1)
    res2 = cv2.bitwise_and(res,res,mask=mask2)
    i = i+1

print ("number of contours: ", i)

cv2.imshow('before', res)
cv2.imshow('after', mask2)
#cv2.imwrite("after.png", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()