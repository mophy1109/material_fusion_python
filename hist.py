import cv2
import numpy as np
import os

image_files = sorted(os.listdir("FQ2"))
for img in image_files:
    if img.split(".")[-1].lower() not in ["jpg", "jpeg", "png"]:
        image_files.remove(img)
for img in image_files:
    imgin = cv2.imread("FQ2/{}".format(img), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(imgin)
    # equ = cv2.equalizeHist(imgin)
    # res = np.hstack((imgin, equ))
    # cv2.imwrite("FQ/{}".format(img), res)
    # res = np.hstack((imgin, cl1))
    cv2.imwrite("FQ/{}".format(img), cl1)
