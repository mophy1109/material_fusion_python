# coding=utf-8
import cv2

# 视频采样算法
def sampling(src):
    capture = cv2.cv.CaptureFromFile(src)

    nbFrames = int(cv2.cv.GetCaptureProperty(capture, cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    fps = cv2.cv.GetCaptureProperty(capture, cv2.cv.CV_CAP_PROP_FPS)


    print ('Num. Frames = ', nbFrames)
    print ('Frame Rate = ', fps, 'fps')
    i = 0
    for f in range( nbFrames ):
        frameImg = cv2.cv.QueryFrame(capture)
        ROI = frameImg[0:frameImg.height - 2, 0:]
        if f % int(fps/3) == 0:
           cv2.cv.SaveImage("out/sam"+ str(i).zfill(5) +".jpg", ROI)
           i += 1
