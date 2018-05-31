# coding=utf-8

import os
import cv2
import cv2.cv as cv
import fusion
import time
import numpy as np
import gc
from sampling import sampling

if __name__ == "__main__":
    start = time.clock()

    cap = cv2.VideoCapture("500px.avi")
    i = 0
    print cap.get(3)
    ret, frameImg = cap.read()
    result = cv2.cvtColor(frameImg, cv2.COLOR_BGR2GRAY)
    imgin1 = result

    f = 0
    while(ret == True):
        frame = cv2.cvtColor(frameImg, cv2.COLOR_BGR2GRAY)
        f = f + 1
        if f % 5 == 0:
            i += 1
            print "reading img:", i
            imgin2 = frame[0:frame.shape[0] - 2, 0:]
            result, imgin1 = fusion.detect(imgin1, imgin2, result)
            del imgin2
        ret, frameImg = cap.read()

    print "That's All Folks!"

    end = time.clock()
    run_time = end - start
    print run_time