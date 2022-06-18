#Edwing Alexis Casillas Valencia.   19110113.   7E1.    Práctica 8 visión artificial
#Extracción del fondo y encontrar esquinas.
import os
from cv2 import imread
import numpy as np
import cv2
from matplotlib import pyplot as plt

im1 = cv2.imread('Perro azul.jpeg')
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
#plt.imshow(im1)
mask = np.zeros(im1.shape[:2], np.uint8)

bgModel = np.zeros((1,65), np.float64)
fgModel = np.zeros((1,65), np.float64)

rectangulo = (490,450,1900,2200)

cv2.grabCut(im1, mask, rectangulo, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
im2 = im1*mask2[:,:,np.newaxis]

plt.imshow(im2)
plt.colorbar()
plt.show()
