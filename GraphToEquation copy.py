#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 13:31:45 2018

@author: 2020shatgiskessell
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections
from matplotlib.pyplot import subplots
from numpy import linspace, random, sin, cos
from scipy import interpolate

img = cv2.imread("/Users/2020shatgiskessell/Desktop/Test_Graphs/Graph5.png")
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

#Display contours
for contour in find_contours(res):
    cv2.drawContours(mask2, [contour], 0, 255, -1)
    res2 = cv2.bitwise_and(res,res,mask=mask2)
    i = i+1

#Get coordinates of only white pixels (the threshold) - WORKS
indices = np.where(mask2 == [255])
coordinates = zip(indices[0], indices[1])



#-------------------------------------------------------------------------------------------------------------------------------
#EACH OF THESE ARE DIFFERENT WAYS TO GET THE EQUATION OF THE GRAPH LINE


#Get equation
#def get_equation(x,y):
#    degree = 2
#    coefs, res, _, _, _ = np.polyfit(x,y,degree, full = True)
#    ffit = np.poly1d(coefs)
#    print (ffit)
#    return ffit


#def get_equation (x,y):
#    # fit spline
#    spl = interpolate.InterpolatedUnivariateSpline(x, y)
#    # creates intervals from x.min to x.max (domain)
#    fitx = linspace(x.min(), x.max(), 100)
#    fig, ax = subplots()
#
#    ax.scatter(x, y)
#    
#    #create poly curve fitting equation
#    degree = 2
#    coefs, res, _, _, _ = np.polyfit(x,y,degree, full = True)
#    ffit = np.poly1d(coefs)
#    print (ffit)
#
#    ax.plot(fitx, spl(fitx))
#    fig.show()
#    return ffit

def get_equation (x,y):
    degree = 30
    #get equation coefficiants and residual value
    coefs, res, _, _, _ = np.polyfit(x,y,degree, full = True)
    #plug coefficients into polynomial
    ffit = np.poly1d(coefs)
    #Create domain from x mins and maxs
    xp = np.linspace(x.min(), x.max(),70)
    #works but prints the vertically flipped graph
    
    #plot everything
    pred_plot = ffit(xp)
    #plt.scatter(x, y, facecolor='None', edgecolor='k', alpha=0.3)
    plt.plot(xp, pred_plot)
    plt.show()

    
#def get_equation(x,y):
#    coef_vals = []
#    res_vals = []
#    #try curve fitting for degrees 1-10
#    for d in range(1,10):
#        coefs, res, _, _, _ = np.polyfit(x,y,d, full = True)
#        print (d, res)
#        coef_vals.append(coefs)
#        res_vals.append(res)
#    #sort the residual values from lowest to highest
#    sorted_res_vals = sorted(res_vals)
#    #get lowest res value
#    lowest_res = sorted_res_vals[0]
#    #get lowest res values corrosponding coef value
#    l_res_index = res_vals.index(lowest_res)
#    l_coefs = coef_vals[l_res_index]    
#    #create polynomial from coefficient
#    ffit = np.poly1d(l_coefs)
#    print (ffit)
##    return ffit
    
#-------------------------------------------------------------------------------------------------------------------------------

get_equation(indices[0], indices[1])

def print_matrix():
    f = open('graphcoordinates.txt', 'w')
    f.write(str(list(coordinates)))
    f.close()

#Graph equation - WORKS
def graph_found_equation(poly_fit_equation, number_of_xs):
    #Pass x values to the equation to get the y values for plotting
    x = []
    for i in range (-(int(number_of_xs/2)), int((number_of_xs/2))):
        x.append(i)
    y = poly_fit_equation(x)
    plt.plot(x,y)
    plt.show()

#Graoh the gotten equation
#graph_found_equation(get_equation(indices[0], indices[1]), len(indices[0]))



cv2.imshow('after', mask2)
#cv2.imwrite("after.png", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()