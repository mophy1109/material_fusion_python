import numpy as np
import cv2
import math
import spatialfrequency as SF
from scipy.stats import mode
from matplotlib import pyplot as plt

input_height = 1198
input_width = 1600
def detect(img1, img2):
    global DEVX
    global DEVY
    #   SIFT generally produces better results, but it is not FOSS, so chose the feature detector
    #   that suits the needs of your project.  ORB does OK
    use_sift = True

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

    #tmpimg = drawMatches.drawMatches(img1, kp1, img2, kp2, matches)
    #plt.imshow(tmpimg), plt.show()

    x_list = []
    y_list = []
    dx = []
    dy = []
    if len(matches) >= 3:
        for mat in matches:
            (x1, y1) = kp1[mat.queryIdx].pt
            (x2, y2) = kp2[mat.trainIdx].pt
            newptx, newpty  = int(x1 - x2), int(y1 - y2)
            x_list.append(newptx)
            y_list.append(newpty)
            dx.append(x1-x2)
            dy.append(y1-y2)
    else:
        print len(matches)
        print "match failed,pass"
        # img3 = drawMatches.drawMatches(img1,kp1,img2,kp2,matches)
        # cv2.imshow('test',img3)
        # cv2.waitKey(0)
        return

    hom = findHomography(kp1, kp2, matches)
    print hom
    out_width = int(math.ceil(max(hom[2][0],np.dot([input_width,input_height,1],hom[0]))))
    out_height = int(math.ceil(max(hom[2][1],np.dot([input_width,input_height,1],hom[1]))))
    print out_width,out_height
    out = cv2.warpPerspective(img2, hom, (out_width, out_height), flags=cv2.INTER_LINEAR)
    cv2.imwrite("Affine.jpg", out)
    cv2.waitKey()
    return out

def findHomography(image_1_kp, image_2_kp, matches):
    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)

    for i in range(0, len(matches)):
        image_1_points[i] = image_1_kp[matches[i].queryIdx].pt
        image_2_points[i] = image_2_kp[matches[i].trainIdx].pt

    homography, mask = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0)

    return homography

if __name__ == "__main__":
    img1 = cv2.imread("distortion/sam00364.jpg",0)
    img2 = cv2.imread("distortion/sam00367.jpg",0)

    out = detect(img1, img2)
    