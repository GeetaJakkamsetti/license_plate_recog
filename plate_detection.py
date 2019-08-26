# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 13:41:14 2019

@author: HP
"""

from TOOLS import Functions
import cv2
import numpy as np
import math
from PIL import Image,ImageFilter

img=cv2.imread('G:\sem7\humAIn\luck.jpg')
#hsv transform - value = gray image
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue, saturation, value = cv2.split(hsv)
# cv2.imshow('gray', value)
# cv2.imwrite(temp_folder + '2 - gray.png', value)

# kernel to use for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# applying topHat/blackHat operations
topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)


# add and subtract between morphological operations
add = cv2.add(value, topHat)
subtract = cv2.subtract(add, blackHat)


# applying gaussian blur on subtract image
blur = cv2.GaussianBlur(subtract, (5, 5), 0)
# thresholding
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)

cv2MajorVersion = cv2.__version__.split(".")[0]
# check for contours on thresh
if int(cv2MajorVersion) >= 4:
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
else:
    imageContours, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# get height and width
height, width = thresh.shape

# create a numpy array with shape given by threshed image value dimensions
imageContours = np.zeros((height, width, 3), dtype=np.uint8)

# list and counter of possible chars
possibleChars = []
countOfPossibleChars = 0

# loop to check if any (possible) char is found
for i in range(0, len(contours)):

    # draw contours based on actual found contours of thresh image
    cv2.drawContours(imageContours, contours, i, (255, 255, 255))

    # retrieve a possible char by the result ifChar class give us
    possibleChar = Functions.ifChar(contours[i])

   
    if Functions.checkIfChar(possibleChar) is True:
        countOfPossibleChars = countOfPossibleChars + 1
        possibleChars.append(possibleChar)


imageContours = np.zeros((height, width, 3), np.uint8)

ctrs = []

# populating ctrs list with each char of possibleChars
for char in possibleChars:
    ctrs.append(char.contour)

# using values from ctrs to draw new contours
cv2.drawContours(imageContours, ctrs, -1, (255, 255, 255))


plates_list = []
listOfListsOfMatchingChars = []

for possibleC in possibleChars:

   
    def matchingChars(possibleC, possibleChars):
        listOfMatchingChars = []

        
        for possibleMatchingChar in possibleChars:
            if possibleMatchingChar == possibleC:
                continue

            # compute stuff to see if chars are a match
            distanceBetweenChars = Functions.distanceBetweenChars(possibleC, possibleMatchingChar)

            angleBetweenChars = Functions.angleBetweenChars(possibleC, possibleMatchingChar)

            changeInArea = float(abs(possibleMatchingChar.boundingRectArea - possibleC.boundingRectArea)) / float(
                possibleC.boundingRectArea)

            changeInWidth = float(abs(possibleMatchingChar.boundingRectWidth - possibleC.boundingRectWidth)) / float(
                possibleC.boundingRectWidth)

            changeInHeight = float(abs(possibleMatchingChar.boundingRectHeight - possibleC.boundingRectHeight)) / float(
                possibleC.boundingRectHeight)

            # check if chars match
            if distanceBetweenChars < (possibleC.diagonalSize * 5) and \
                    angleBetweenChars < 12.0 and \
                    changeInArea < 0.5 and \
                    changeInWidth < 0.8 and \
                    changeInHeight < 0.2:
                listOfMatchingChars.append(possibleMatchingChar)

        return listOfMatchingChars


    
    listOfMatchingChars = matchingChars(possibleC, possibleChars)

    listOfMatchingChars.append(possibleC)

    
    if len(listOfMatchingChars) < 3:
        continue

    # here the current list passed test as a "group" or "cluster" of matching chars
    listOfListsOfMatchingChars.append(listOfMatchingChars)

    listOfPossibleCharsWithCurrentMatchesRemoved = list(set(possibleChars) - set(listOfMatchingChars))

    recursiveListOfListsOfMatchingChars = []

    for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:
        listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)

    break

imageContours = np.zeros((height, width, 3), np.uint8)

for listOfMatchingChars in listOfListsOfMatchingChars:
    contoursColor = (255, 0, 255)

    contours = []

    for matchingChar in listOfMatchingChars:
        contours.append(matchingChar.contour)

    cv2.drawContours(imageContours, contours, -1, contoursColor)

for listOfMatchingChars in listOfListsOfMatchingChars:
    possiblePlate = Functions.PossiblePlate()

    # sort chars from left to right based on x position
    listOfMatchingChars.sort(key=lambda matchingChar: matchingChar.centerX)

    # calculate the center point of the plate
    plateCenterX = (listOfMatchingChars[0].centerX + listOfMatchingChars[len(listOfMatchingChars) - 1].centerX) / 2.0
    plateCenterY = (listOfMatchingChars[0].centerY + listOfMatchingChars[len(listOfMatchingChars) - 1].centerY) / 2.0

    plateCenter = plateCenterX, plateCenterY

    # calculate plate width and height
    plateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].boundingRectX + listOfMatchingChars[
        len(listOfMatchingChars) - 1].boundingRectWidth - listOfMatchingChars[0].boundingRectX) * 1.3)

    totalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        totalOfCharHeights = totalOfCharHeights + matchingChar.boundingRectHeight

    averageCharHeight = totalOfCharHeights / len(listOfMatchingChars)

    plateHeight = int(averageCharHeight * 1.5)

    # calculate correction angle of plate region
    opposite = listOfMatchingChars[len(listOfMatchingChars) - 1].centerY - listOfMatchingChars[0].centerY

    hypotenuse = Functions.distanceBetweenChars(listOfMatchingChars[0],
                                                listOfMatchingChars[len(listOfMatchingChars) - 1])
    correctionAngleInRad = math.asin(opposite / hypotenuse)
    correctionAngleInDeg = correctionAngleInRad * (180.0 / math.pi)

    # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    possiblePlate.rrLocationOfPlateInScene = (tuple(plateCenter), (plateWidth, plateHeight), correctionAngleInDeg)

    # get the rotation matrix for our calculated correction angle
    rotationMatrix = cv2.getRotationMatrix2D(tuple(plateCenter), correctionAngleInDeg, 1.0)

    height, width, numChannels = img.shape

    # rotate the entire image
    imgRotated = cv2.warpAffine(img, rotationMatrix, (width, height))

    # crop the image/plate detected
    imgCropped = cv2.getRectSubPix(imgRotated, (plateWidth, plateHeight), tuple(plateCenter))

    # copy the cropped plate image into the applicable member variable of the possible plate
    possiblePlate.Plate = imgCropped

    # populate plates_list with the detected plate
    if possiblePlate.Plate is not None:
        plates_list.append(possiblePlate)

    # draw a ROI on the original image
    for i in range(0, len(plates_list)):
        # finds the four vertices of a rotated rect - it is useful to draw the rectangle.
        p2fRectPoints = cv2.boxPoints(plates_list[i].rrLocationOfPlateInScene)

        # roi rectangle colour
        rectColour = (0,255,0)

        cv2.line(imageContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), rectColour, 3)
        cv2.line(imageContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), rectColour, 3)
        cv2.line(imageContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), rectColour, 3)
        cv2.line(imageContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), rectColour, 3)

        cv2.line(img, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), rectColour, 3)
        cv2.line(img, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), rectColour, 3)
        cv2.line(img, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), rectColour, 3)
        cv2.line(img, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), rectColour, 3)

       

        cv2.imshow("detectedOriginal", img)
        
cv2.waitKey(0)

'''
import pytesseract

img = Image.open('G:\sem7\humAIn\c3.jpg')
enhancer = ImageEnhance.Brightness(img)
enhanced_im = enhancer.enhance(1.8)
img2 = img.filter(ImageFilter.BLUR)
pixels = img2.load()
width, height = img2.size
x_ = []
y_ = []
for x in range(width):
    for y in range(height):
        if pixels[x, y] == (255, 255, 255):
            x_.append(x)
            y_.append(y)

img = img.crop((min(x_), min(y_),  max(x_), max(y_)))
text = pytesseract.image_to_string(img, lang='eng', config='-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
print(text)
'''
