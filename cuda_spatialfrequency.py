import numpy as np
import cv2
import math
import time
import numba
from numba import cuda


def genMatR1(img1, img2, edgelen):
    #comparison of NSEN of a pic to generate the matrix for blending
    #img1 and img2 should be gray images with the same size
    out = np.zeros((img1.shape[0], img1.shape[1]), dtype='uint8')
    n = edgelen
    cols = int(math.ceil(img1.shape[1]/n))
    rows = int(math.ceil(img1.shape[0]/n))
    miniout = np.zeros((rows,cols),dtype='uint8')

    for i in range(rows - 1):
        for j in range(cols - 1):
            if NSEN(img1[i * n:min((i + 1) * n, img1.shape[0]),j * n:min((j + 1) * n, img1.shape[1])]) > NSEN(img2[i * n:min((i + 1) * n, img1.shape[0]),j * n:min((j + 1) * n, img1.shape[1])]):
                miniout[i, j] = 1
    miniout = cor(miniout)
    for i in range(rows - 1):
        for j in range(cols - 1):
            if miniout[i, j] == 1:
                out[i * n:min((i + 1) * n, img1.shape[0]), j * n:min((j + 1) * n, img1.shape[1])] = 1
    out = stretchImage(out)
    # cv2.imwrite("test.png", out)
    return out

def genMatR2(img1, img2, edgelen):
    #comparison of Spacial Frequency of a pic to generate the matrix for blending
    #img1 and img2 should be gray images with the same size
    out = np.zeros((img1.shape[0], img1.shape[1]), dtype='uint8')
    n = edgelen
    cols = int(math.ceil(img1.shape[1]/n))
    rows = int(math.ceil(img1.shape[0]/n))
    miniout = np.zeros((rows,cols),dtype='uint8')

    # start = time.clock()
    #
    # for i in range(rows - 1):
    #     for j in range(cols - 1):
    #         if SF(img1[i * n:min((i + 1) * n, img1.shape[0]),j * n:min((j + 1) * n, img1.shape[1])]) >= SF(img2[i * n:min((i + 1) * n, img1.shape[0]),j * n:min((j + 1) * n, img1.shape[1])]):
    #             miniout[i,j] = 1
    # end = time.clock()
    #
    # print ("calculate SF time:",end - start)

    start = time.clock()

    # for i in range(rows - 1):
    #     for j in range(cols - 1):
    #         if SF_parallel(img1[i * n:min((i + 1) * n, img1.shape[0]),j * n:min((j + 1) * n, img1.shape[1])]) >= SF(img2[i * n:min((i + 1) * n, img1.shape[0]),j * n:min((j + 1) * n, img1.shape[1])]):
    #             miniout[i,j] = 1
    result = SF_parallel(img1, img2, n)

    end = time.clock()

    print ("calculate SF(para) time:",end - start)

    start1 = time.clock()
    miniout = cor(miniout)
    end1 = time.clock()
    print("correct time:", end1 - start1)

    start2 = time.clock()
    for i in range(rows - 1):
        for j in range(cols - 1):
            if miniout[i,j] == 1:
                out[i * n:min((i + 1) * n, img1.shape[0]), j * n:min((j + 1) * n, img1.shape[1])] = 1
    end2 = time.clock()
    print("transform time:", end2 - start2)

    return out

def NSEN(img):
    #calculate NSEN of an image
    n_num = 0.0
    xcols = int(img.shape[1])
    xrows = int(img.shape[0])
    if xcols==0 or xrows == 0:
        return 0
    # for i in range(xrows / 2 - 4, xrows / 2 + 4):
    #     for j in range(xcols / 2 - 4, xcols / 2 + 4):
    for i in range(xrows-1):
        for j in range(xcols-1):
            if (img[i, j] > img[i - 1, j] and img[i, j] > img[i + 1, j]) or (
                img[i, j] < img[i - 1, j] and img[i, j] < img[i + 1, j]) or (
                img[i, j] < img[i, j - 1] and img[i, j] < img[i, j + 1]) or (
                img[i, j] > img[i, j - 1] and img[i, j] > img[i, j + 1]) or (
                img[i, j] < img[i - 1, j - 1] and img[i, j] < img[i + 1, j + 1]) or (
                img[i, j] > img[i - 1, j - 1] and img[i, j] > img[i + 1, j + 1]) or (
                img[i, j] < img[i - 1, j + 1] and img[i, j] < img[i + 1, j - 1]) or (
                img[i, j] > img[i - 1, j + 1] and img[i, j] > img[i + 1, j - 1]):
                n_num = n_num + 1
    SEN = n_num/((xcols-2)*(xrows-2))
    # print SEN
    return SEN

def SF(img):

    RF_2 = 0
    CF_2 = 0
    xcols = int(img.shape[1])
    xrows = int(img.shape[0])
    for i in range(xrows):
        for j in range(xcols):
            #print (i,j,img[i, j - 1])
            RF_2 = int(RF_2) + (int(img[i, j]) - int(img[i, j - 1]))**2
            CF_2 = int(CF_2) + (int(img[i, j]) - int(img[i - 1, j]))**2
            # print img[i, j], img[i, j - 1], img[i - 1, j]
            # print RF_2, CF_2
            # print '---------'
    SF = math.sqrt(RF_2/xcols/xrows + CF_2/xcols/xrows)

    return SF

@cuda.jit()
def paraCalculate_SF(img, RF_result, CF_result):
    i, j = cuda.blockIdx.x,cuda.blockIdx.y
    if i < img.shape[0] and j < img.shape[1] and i>0 and j>0:
        RF_result[i][j] = (img[i, j] - img[i, j - 1]) ** 2
        CF_result[i][j] = (img[i, j] - img[i - 1, j]) ** 2
    return

@cuda.jit()
def calWeightSF(RF1, CF1, RF2, CF2, edgelen, result):
    i, j = cuda.blockIdx.x, cuda.blockIdx.y
    n = edgelen
    if i < edgelen and j < edgelen and i>0 and j>0:
        x = min((i + 1) * n, RF1.shape[0]) - i * n
        y = min((j + 1) * n, RF1.shape[1]) - j * n



        RF_1 = RF1[i*n:i * n:min((i + 1) * n, RF1.shape[0]),j * n:min((j + 1) * n, RF1.shape[1])].sum()
        CF_1 = CF1[i*n:i * n:min((i + 1) * n, RF1.shape[0]),j * n:min((j + 1) * n, RF1.shape[1])].sum()
        RF_2 = RF2[i*n:i * n:min((i + 1) * n, RF1.shape[0]),j * n:min((j + 1) * n, RF1.shape[1])].sum()
        CF_2 = CF2[i*n:i * n:min((i + 1) * n, RF1.shape[0]),j * n:min((j + 1) * n, RF1.shape[1])].sum()
        SF1 = math.sqrt(RF_1/xcols/xrows + CF_1/xcols/xrows)
        (img1[i * n:min((i + 1) * n, img1.shape[0]),j * n:min((j + 1) * n, img1.shape[1])]) >= SF(img2[i * n:min((i + 1) * n, img1.shape[0]),j * n:min((j + 1) * n, img1.shape[1])])
    return

def SF_parallel(img1, img2, edgelen):
    n = edgelen
    cols = int(math.ceil(img1.shape[1]/n))
    rows = int(math.ceil(img1.shape[0]/n))

    xcols = int(img1.shape[1])
    xrows = int(img1.shape[0])
    result = np.zeros((cols, rows))
    RF_result1 = np.zeros((xcols, xrows))
    CF_result1 = np.zeros((xcols, xrows))
    RF_result2 = np.zeros((xcols, xrows))
    CF_result2 = np.zeros((xcols, xrows))

    # print (result.shape)
    threadsperblock = 1
    blockspergrid = (xcols,xrows) #threadperblock
    paraCalculate_SF[blockspergrid, threadsperblock](np.ascontiguousarray(img1), RF_result1, CF_result1)
    paraCalculate_SF[blockspergrid, threadsperblock](np.ascontiguousarray(img2), RF_result2, CF_result2)

    blkpergrid = (cols, rows)
    #calWeightSF[blkpergrid, threadsperblock](RF_result1, CF_result1, RF_result2, CF_result2, edgelen, result)
    # RF_2 = RF_result.sum()
    # CF_2 = CF_result.sum()
    # print(RF_2, CF_2)
    # SF = math.sqrt(RF_2/xcols/xrows + CF_2/xcols/xrows)
    return SF

#correct the weight-mat with bilateralFilter
def cor(img):
    # start = time.clock()
    out = cv2.bilateralFilter(img, 40, 30, 30)
    # end = time.clock()
    # run_time = end - start
    # print run_time
    return out

def stretchImage(Region):
    minI = Region.min()
    maxI = Region.max()
    if maxI == minI:
        maxI = 1
    out = (Region - minI) / (maxI - minI) * 255
    return out

def genMat(img1, img2, len):
    return genMatR2(img1, img2, len)

if __name__ == "__main__":
    img1 = cv2.imread("out90/1-001.jpg",0)
    img2 = cv2.imread("out90/1-002.jpg",0)
    start1 = time.clock()
    genMat(img1, img2, 16)
    print (type(img1),"\n ===================\n", type(img2))
    end1 = time.clock() - start1
    print("total time:",end1)
