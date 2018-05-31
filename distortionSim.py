# coding=utf-8
import numpy as np
import cv2
from scipy.optimize import root
from matplotlib import pyplot as plt
import math

#图片大小定义
width = 1600
height = 1200
U1 = []
U2 = []

#计算两点之间的距离（OK）
def sqrdist_pt(pt1, pt2):
    dist = (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2
    return dist

#对一组二维向量求和
def sumOfArr(A):
    SUM = [0, 0]
    for dim in A:
        SUM = np.add(SUM, dim)
    return SUM

#损失函数
def costFunc(k):
    F = 0
    U11, U21 = cordistor(k[0], k[1])
    #Δx、Δy 由校正后的值得到
    deltx, delty = caldelta(U1, U2)
    for n in xrange(len(U1)):
        x, y = U21[n][0] - (U11[n][0] + deltx), U21[n][1] - (U11[n][1] + delty)
        #print x,y
        F = F + x**2 + y**2
    #print deltx, delty
    #print "F = ", F
    return F, 0

#畸变校正函数(暂定)
def cordistor(k1, k2):
    UU1 = []
    UU2 = []
    for n in U1:
        r1 = math.sqrt(sqrdist_pt(n, [width/2, height/2]))
        # newx, newy = n[0] + (n[0] - width/2)*(k1*r1**2+k2*r1**4),n[1] + (n[1] - height/2)*(k1*r1**2+k2*r1**4)
        newx, newy = n[0] + (n[0] - width/2)*(k1*r1**2),n[1] + (n[1] - height/2)*(k1*r1**2)
        UU1.append([newx, newy])
        # print "k1, k2 = ", k1, k2
        # print "before = ", n
        # print "after = ", newx, newy
        # print "---------------------"
    for m in U2:
        r2 = math.sqrt(sqrdist_pt(m, [width/2, height/2]))
        # newx, newy = m[0] + (m[0] - width/2)*(k1*r2**2+k2*r2**4),m[1] + (m[1] - height/2)*(k1*r2**2+k2*r2**4)
        newx, newy = m[0] + (m[0] - width/2)*(k2*r2**2),m[1] + (m[1] - height/2)*(k2*r2**2)
        UU2.append([newx, newy])
    return UU1, UU2

#获得deltaX、deltaY的值,均值法（待改）
def caldelta(U11, U21):
    deltx, delty  = (sumOfArr(U21) - sumOfArr(U11))/len(U11)
    return deltx, delty

def detect(img1, img2):
    global DEVX
    global DEVY
    #   SIFT generally produces better results, but it is not FOSS, so chose the feature detector
    #   that suits the needs of your project.  ORB does OK
    use_sift = False

    if use_sift:
        detector = cv2.SURF()
    else:
        detector = cv2.ORB(1000)

    # keypoints as kp, descriptors as desc
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if use_sift:
        if len(kp2) < 5000:
            print "bad pic"
            return
        bf = cv2.BFMatcher()
        # This returns the top two matches for each feature point (list of list)
        pairMatches = bf.knnMatch(des1, des2, k=2)
        rawMatches = []
        for m, n in pairMatches:
            if m.distance < 0.6 * n.distance:
                rawMatches.append(m)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        rawMatches = bf.match(des1, des2)

    sortMatches = sorted(rawMatches, key=lambda x: x.distance)
    matches = sortMatches[0:300]

    U1 = []
    U2 = []
    if len(matches) >= 3:
        for mat in matches:

            U1.append(kp1[mat.queryIdx].pt)
            U2.append(kp2[mat.trainIdx].pt)
        return np.asarray(U1), np.asarray(U2)
    else:
        print len(matches)
        print "match failed,pass"
        return

if __name__ == "__main__":
    img1 = cv2.imread("distortion/sam00373.jpg", 0)
    img2 = cv2.imread("distortion/sam00374.jpg", 0)
    U1, U2 = detect(img1,img2)
    k = [0, 0]
    sol = root(costFunc, k)
    #fun, x0, args = (), method = 'hybr', jac = None, tol = None, callback = None, options = None)
    k1, k2 = sol.x
    U1X, U2X = cordistor(k1, k2)
    print k1, k2, caldelta(U1X, U2X)

    # for n in xrange(len(U1X)):
    #     print U2X[n] - U1X[n]