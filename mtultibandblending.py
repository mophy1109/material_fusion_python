# coding=utf-8
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

def BlendArbitrary(img1, img2, R, level):
    # img1 and img2 have the same size
    # R represents the region to be combined
    # level is the expected number of levels in the pyramid

    x, y = img1.shape[0], img1.shape[1]
    LA, GA = LaplacianPyramid(img1, level)
    LB, GB = LaplacianPyramid(img2, level)
    GR = GaussianPyramid(R, level)
    print GR
    ones = np.ones((x, y))
    zeros = np.zeros((x, y))
    GRN = []
    for i in range(level):
        GRN.append(np.ones((GR[i].shape[0], GR[i].shape[1])) - GR[i])
    LC = []
    for i in xrange(level):
        bA ,gA ,rA = cv2.split(LA[i])
        bB, gB, rB = cv2.split(LB[i])
        b, g, r = GR[level - 1 - i] * bA + GRN[level - 1 - i] * bB, GR[level - 1 - i] * gA + GRN[level - 1 - i] * gB, GR[level - 1 - i] * rA + GRN[level - 1 - i] * rB
        LC.append(cv2.merge((b, g, r)))
    result = reconstruct(LC)
    cv2.imwrite('result.jpg',result)
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.imshow('region', stretchImage(R))
    cv2.imshow('result', np.uint8(result))
    cv2.waitKey(0)

def GaussianPyramid(R, level):
    G = R.copy().astype(np.float64)
    gp = [G]
    for i in xrange(level):
        G = cv2.pyrDown(G)
        gp.append(G)
    return gp

def LaplacianPyramid(img, level):
    gp = GaussianPyramid(img, level)
    lp = [gp[level-1]]
    for i in xrange(level - 1, -1, -1):
        GE = cv2.pyrUp(gp[i])
        GE = cv2.resize(GE, (gp[i - 1].shape[1], gp[i - 1].shape[0]), interpolation=cv2.INTER_CUBIC)
        L = cv2.subtract(gp[i - 1], GE)
        lp.append(L)
    return lp, gp

def reconstruct(input_pyramid):
    out = input_pyramid[0]
    for i in xrange(1, len(input_pyramid)):
        # cv2.imshow('out', np.uint8(out))
        # cv2.waitKey(0)
        out = cv2.pyrUp(out)
        out = cv2.resize(out, (input_pyramid[i].shape[1],input_pyramid[i].shape[0]), interpolation = cv2.INTER_CUBIC)
        out = cv2.add(out, input_pyramid[i])
        print out
    return out

#权值矩阵归一化
def stretchImage(Region):
    minI = Region.min()
    maxI = Region.max()
    out = (Region - minI) / (maxI - minI) * 255
    return out

im1 = cv2.imread('out/sam00000.jpg')
im2 = cv2.imread('out/sam00001.jpg')
im3 = cv2.imread('tstimg/mask.bmp', cv2.CV_LOAD_IMAGE_GRAYSCALE)
C = BlendArbitrary(im1, im2, im3/255, 10)