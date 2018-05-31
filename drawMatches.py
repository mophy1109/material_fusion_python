import numpy as np
import cv2
from matplotlib import pyplot as plt

def drawMatches(img1, kp1, img2, kp2, matches):

    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1 + cols2,3), dtype='uint8')

    out[:rows1,:cols1] = np.dstack([img1, img1, img1])
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    # Also return the image if you'd like a copy
    return out

def stitch(img1, img2, matches):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]


img1 = cv2.imread('out/sam00000.jpg',0)          # queryImage
img2 = cv2.imread('out/sam00001.jpg',0) # trainImage

# Initiate SIFT detector
orb = cv2.SURF()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# create BFMatcher object
bf = cv2.BFMatcher()

# Match descriptors.
pairMatches = bf.knnMatch(des1, des2, k=2)
rawMatches = []
for m, n in pairMatches:
    if m.distance < 0.6 * n.distance:
        rawMatches.append(m)

# Sort them in the order of their distance.
matches = sorted(rawMatches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = drawMatches(img1,kp1,img2,kp2,matches[:100])

# cv2.imshow('233',img3)
# cv2.waitKey(0)
plt.imshow(img3)
plt.show()
