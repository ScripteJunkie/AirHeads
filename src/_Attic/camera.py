import cv2
import numpy as np
# from scipy.misc import bytescale

cap = cv2.VideoCapture('src/assets/Test.mp4')
# cap = cv2.VideoCapture(0)

first_iter = True
backSub = cv2.createBackgroundSubtractorMOG2()
points = []

print(np.shape(cap))
while(1):

    _, frame = cap.read()
    if (_ == False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        frame = frame[0:1800,0:1800]

        fgMask = backSub.apply(frame)    
        cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        cv2.imshow('FG Mask', fgMask)
        cut = cv2.bitwise_and(frame, frame, mask=fgMask)
        cut_blur = cv2.medianBlur(cut, 5)
        cv2.imshow("Cut", cut_blur)

        hsv = cv2.cvtColor(cut, cv2.COLOR_BGR2HSV)

        # define range of white color in HSV
        # change it according to your need !
        # lower_white = np.array([133, 0, 0], dtype=np.uint8)
        # upper_white = np.array([179, 255, 115], dtype=np.uint8)
        lower_white = np.array([0, 0, 0], dtype=np.uint8)
        upper_white = np.array([179, 255, 82], dtype=np.uint8)

        # Threshold the HSV image to get only white colors
        mask = cv2.inRange(hsv, lower_white, upper_white)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)

        contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #sorting the contour based of area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        # Blur the image to reduce noise
        img_blur = cv2.medianBlur(gray, 5)
        # Apply hough transform on the image
        circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, frame.shape[0]/64, param1=200, param2=10, minRadius=5, maxRadius=30)
        # Draw detected circles
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # for i in circles[0]:
            #     # Draw outer circle
            #     # blur = cv2.GaussianBlur(frame, (5, 5), 25)
            #     cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 255), 2)
            #     points.append((i[0], i[1]))
            #     # cv2.imshow('blur',blur)
            #     # Draw inner circle
            #     # cv2.circle(frame, (i[0], i[1]), 2, (255, 0, 0), 3)
        
        if contours:
            #if any contours are found we take the biggest contour and get bounding box
            (x_min, y_min, box_width, box_height) = cv2.boundingRect(contours[0])
            #drawing a rectangle around the object with 15 as margin
            if (box_width < 100):
                cv2.rectangle(frame, (x_min - 10, y_min - 10),(x_min + box_width + 10, y_min + box_height + 10),(0,255,0), 4)
        if first_iter:
            avg = np.float32(frame)
            first_iter = False

        for i in range(1, len(points)):
            cv2.circle(frame, (points[i][0], points[i][1]), 2, (0, 0, 255), 2)
        # cv2.accumulateWeighted(frame, avg, 0.005)
        # result = cv2.convertScaleAbs(avg)
        # cv2.imshow('avg',result)

        cv2.imshow('res',res)
        cv2.imshow('mask',mask)
        cv2.imshow('frame',frame)

        # fps = cap.get(cv2.CAP_PROP_FPS)
        # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1) #wait until any key is pressed

cv2.destroyAllWindows()