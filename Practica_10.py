#Edwing Alexis Casillas Valencia.   19110113.   7E1.    Práctica 8 visión artificial
#Extracción del fondo y encontrar esquinas.
import os
from turtle import width
from cv2 import imread
import numpy as np
import cv2
from matplotlib import pyplot as plt

#im1 = cv2.imread('Perro azul.jpeg')
#im1 = cv2.imread('IMG_2656.jpg')
#im1 = cv2.imread('IMG_0164.jpg')
im1 = cv2.imread('Rubik.jpg')
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

scale = 70
width = int(im1.shape[1] * scale / 100)
height = int(im1.shape[0] * scale / 100)
dsize = (width, height)
im1 = cv2.resize(im1, dsize)

#plt.imshow(im1)
#plt.show()
mask = np.zeros(im1.shape[:2], np.uint8)

bgModel = np.zeros((1,65), np.float64)
fgModel = np.zeros((1,65), np.float64)

#rectangulo = (490,450,1900,2200)
#rectangulo = (1000,0,1000,1000)
#rectangulo = (520,0,1000,1300)  #rostro
#rectangulo = (600,0,1000,2700)
rectangulo = (500,120,2370,1790)

cv2.grabCut(im1, mask, rectangulo, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
im2 = im1*mask2[:,:,np.newaxis]

plt.imshow(im2)
plt.colorbar()
plt.show()

gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
gray = np.float32(gray)

corners = cv2.goodFeaturesToTrack(gray,100, 0.01, 10)
corners = np.int0(corners)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(im2, (x,y), 10, 255, -1)

plt.imshow(im2)
plt.show()
