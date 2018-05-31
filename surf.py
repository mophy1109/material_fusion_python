import numpy as np
import drawMatches
import cv2
import time
from matplotlib import pyplot as plt

img1 = cv2.imread('out/sam00000.jpg',0)          # queryImage
img2 = cv2.imread('out/sam00001.jpg',0) # trainImage

start = time.clock()
# Initiate SIFT detector
surf = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = surf.detectAndCompute(img1,None)
kp2, des2 = surf.detectAndCompute(img2,None)
end = time.clock()
run_time = end - start
print run_time
# create BFMatcher object
bf = cv2.BFMatcher()
# This returns the top two matches for each feature point (list of list)
pairMatches = bf.knnMatch(des1, des2, k=2)

rawMatches = []
for m, n in pairMatches:
    if m.distance < 0.7 * n.distance:
        rawMatches.append(m)

sortMatches = sorted(rawMatches, key=lambda x: x.distance)
goodmatches = sortMatches[0:128]

img3 = drawMatches.drawMatches(img1,kp1,img2,kp2,goodmatches)



plt.imshow(img3)
plt.show()