from numba import cuda
import numpy
import time
import cv2
import os

from timeit import default_timer as timer

@cuda.jit
def showimage(images):
    pos = cuda.grid(1)
    if pos < len(images):
        cv2.imread(images[pos], 0)
        cv2.imshow(pos, images)

def reading_file(dir):
    image_files = sorted(os.listdir(dir))
    images = []
    for img in image_files:
        if img.split(".")[-1].lower() not in ["jpg", "jpeg", "png"]:
            image_files.remove(img)
        images.append(cv2.imread(dir.format(img), 0))
    return images

def fusionImages(image_files):
    # 读取第一张图作为初始结果
    result = cv2.imread("FQ/{}".format(image_files[0]), 0)
    imgin1 = result
    img_files = []
    for img in image_files:
        print ("Reading file {}".format(img))
        img_files.append(cv2.imread("FQ/{}".format(img), 0))
    return

if __name__ == "__main__":

    start1 = time.clock()
    images=[]
    images = reading_file("out90")
    #define threads

    threadsperblock = 32
    blockspergrid = (len(images) + (threadsperblock - 1)) #threadperblock
    showimage[blockspergrid, threadsperblock](images)

    end1 = time.clock() - start1
    print(end1)
