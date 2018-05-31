import cv2
import numpy as np

A = cv2.imread('out/sam00000.jpg')
B = cv2.imread('out/sam00010.jpg')

# generate Gaussian pyramid for A
G = A.astype(np.float64)
gpA = [G]

for i in xrange(8):
    G = cv2.pyrDown(G)
    gpA.append(G)

# generate Gaussian pyramid for B
G = B.copy().astype(np.float64)
gpB = [G]
for i in xrange(8):
    G = cv2.pyrDown(G)
    gpB.append(G)

# generate Laplacian Pyramid for A
lpA = [gpA[7]]
for i in xrange(7,0,-1):
    GE = cv2.pyrUp(gpA[i])
    GE = cv2.resize(GE, (gpA[i-1].shape[1],gpA[i-1].shape[0]), interpolation = cv2.INTER_CUBIC)
    L = cv2.subtract(gpA[i-1],GE)
    lpA.append(L)

# generate Laplacian Pyramid for B
lpB = [gpB[7]]
for i in xrange(7,0,-1):
    GE = cv2.pyrUp(gpB[i])
    print GE
    GE = cv2.resize(GE, (gpB[i-1].shape[1],gpB[i-1].shape[0]), interpolation = cv2.INTER_CUBIC)
    L = cv2.subtract(gpB[i-1],GE)
    lpB.append(L)

# Now add left and right halves of images in each level
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,:cols/2], lb[:,cols/2:]))
    LS.append(ls)

# now reconstruct
ls_ = LS[0]
for i in xrange(1,8):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.resize(ls_, (LS[i].shape[1],LS[i].shape[0]), interpolation = cv2.INTER_CUBIC)
    ls_ = cv2.add(ls_, LS[i])

# image with direct connecting each half
real = np.hstack((A[:,:cols/2],B[:,cols/2:]))

cv2.imwrite('Pyramid_blending2.jpg',ls_)
cv2.imwrite('Direct_blending.jpg',real)

print ls_.shape
print real.shape