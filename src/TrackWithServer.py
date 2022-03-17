#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np
# import imutils
import time
import flask as Flask
import csv



#server shit
from base64 import encode
import socketserver
import time
import random


class MyTCPHandler(socketserver.BaseRequestHandler):

    fieldnames = [ 'X', 'Y' ]

    rows = []

    f = open('Coords.csv', 'w')
    index = 0
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define source and output
    camRgb = pipeline.createColorCamera()
    xoutRgb = pipeline.createXLinkOut()

    xoutRgb.setStreamName("rgb")

    # Properties
    camRgb.setPreviewSize(1920, 1080)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    # Linking
    camRgb.preview.link(xoutRgb.input)

    # used to record the time when we processed last frame
    prev_frame_time = 0
    
    # used to record the time at which we processed current frame
    new_frame_time = 0

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:

        print('Connected cameras: ', device.getConnectedCameras())
        # Print out usb speed
        print('Usb speed: ', device.getUsbSpeed().name)

        # Output queue will be used to get the rgb frames from the output defined above
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        first_iter = True
        backSub = cv2.createBackgroundSubtractorMOG2()
        points = []

        print(np.shape(qRgb))
        frame = None
        rownum = 1
        print("Row,Time,X,Y")
        while True:
            inRgb = qRgb.tryGet()

            if inRgb is not None:
                # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
                frame = inRgb.getCvFrame()

            # Retrieve 'bgr' (opencv format) frame

            if frame is not None:
            
                framed = frame #frame[100:980, 200:1720]
                

            

                hsv = cv2.cvtColor(framed, cv2.COLOR_BGR2HSV)

                
                lower_white = np.array([43, 67, 78], dtype=np.uint8)
                upper_white = np.array([86, 255, 255], dtype=np.uint8)

                # Threshold the HSV image to get only white colors
                mask = cv2.inRange(hsv, lower_white, upper_white)
                # Bitwise-AND mask and original image
                res = cv2.bitwise_and(framed,framed, mask= mask)

                contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                #sorting the contour based of area
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                tracked = framed.copy()
                ppp = []
                if contours:
                    #if any contours are found we take the biggest contour and get bounding box
                    (x_min, y_min, box_width, box_height) = cv2.boundingRect(contours[0])
                    #drawing a rectangle around the object with 15 as margin
                    if (30 < box_width < 55 and 30 < box_height < 55 ):
                        ppp = [int(x_min + (box_width/2)), int(y_min + (box_height/2))]
                        points.append(ppp) # FP added ppx/y to reuse below
                        cv2.circle(tracked, (points[-1][0], points[-1][1]), 20, (100, 150, 255), -1)
                        # cv2.rectangle(framed, (x_min - 10, y_min - 10),(x_min + box_width + 10, y_min + box_height + 10),(0,255,0), 4)
                if first_iter:
                    avg = np.float32(framed)
                    first_iter = False

            
                new_frame_time = time.time()
                output = []
                

                ppp = np.array(ppp)
                ppp = ppp/1192
                if( len(ppp) == 2 ):
                    print(str(rownum)+","+str(new_frame_time)+","+str(ppp[0])+","+str(ppp[1]))
                    rownum = rownum + 1



                    #server shit
                    def handle(self):
                        # self.request is the TCP socket connected to the client
                        self.data = self.request.recv(1024).strip()
                        print("{} wrote:".format(self.client_address[0]))
                        print(self.data)
                        # just send back the same data, but upper-cased
                        # self.request.sendall(self.data.upper())
                        thesx=str(ppp[0])+','+str(ppp[1])
                        self.request.sendall(bytearray(thesx.encode()))
                    

                    
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time
            
                # converting the fps into integer
                fps = int(fps)
            
                
                fps = str(fps)
            
                # putting the FPS count on the frame
                cv2.putText(tracked, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 0), 2, cv2.LINE_AA)

                cv2.imshow('Tracking',tracked)

                # fps = qRgb.get(cv2.CAP_PROP_FPS)
                # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('p'):
                cv2.waitKey(-1) #wait until any key is pressed
            if key == ord('c'):
                points=[]


    cv2.destroyAllWindows()





if __name__ == "__main__":
    HOST, PORT = "129.21.55.174", 9998

    # Create the server, binding to localhost on port 9999
    server = socketserver.TCPServer((HOST, PORT), MyTCPHandler)

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    print("go!")
    server.serve_forever()
    print("done")

