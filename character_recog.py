# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 13:42:13 2019

@author: HP
"""

from PIL import Image, ImageFilter
import pytesseract

img = Image.open('G:\sem7\humAIn\c3.jpg')
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