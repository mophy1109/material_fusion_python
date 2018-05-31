# coding=utf-8

import os
import cv2
import fusion
import time
import gc
from sampling import sampling

save_name = "FQtest.jpg"

def fusionImages(image_files):
    # 读取第一张图作为初始结果
    result = cv2.imread("zirconSmall/1/{}".format(image_files[0]), 0)
    imgin1 = result

    for img in image_files:
        print ("Reading file {}".format(img))
        imgin2 = cv2.imread("zirconSmall/1/{}".format(img), 0)
        result, imgin1 = fusion.detect(imgin1, imgin2, result)
        del imgin2
    cv2.imwrite("result/" + save_name, result)

def getFilelist(dir):
    image_files = sorted(os.listdir(dir))
    for img in image_files:
        if img.split(".")[-1].lower() not in ["jpg", "jpeg", "png"]:
            image_files.remove(img)

    fulldir = []
    for img in image_files:
        fulldir.append(dir + format(img))
    return fulldir

if __name__ == "__main__":
    start = time.clock()
    # sampling("500px.avi")
    # samptime = time.clock() - start
    # print "sample time:",samptime

    image_files = sorted(os.listdir("zirconSmall/1"))
    for img in image_files:
        print(img.split(".")[-1])
        if img.split(".")[-1].lower() not in ["jpg", "jpeg", "png"]:
            image_files.remove(img)
    print(getFilelist("zirconSmall/1/"))
    fusionImages(image_files)

    print ("That's All Folks!")

    end = time.clock()
    run_time = end - start
    print (run_time)

