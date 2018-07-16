# import the necessary packages
from imutils.video import WebcamVideoStream
import imutils
import cv2
import numpy as np
import time

stop_signCascade = cv2.CascadeClassifier('D:\MyDocuments\python_code\opencv\Stopsign_HAAR_19Stages.xml')
traffic_lightsCascade = cv2.CascadeClassifier('D:\MyDocuments\python_code\opencv\TrafficLight_HAAR_16Stages.xml')

# initialize the camera and grab a reference to the raw camera capture
vs = WebcamVideoStream(src=0).start()

# allow the camera to warmup
time.sleep(0.1)

lower = {'red':(166, 84, 141), 'green':(50, 100, 100), 'yellow':(20, 100, 100)}
upper = {'red':(186,255,255), 'green':(70,255,255), 'yellow':(30,255,255)}

#lower = {'red':(166, 84, 141), 'green':(66, 122, 129), 'yellow':(23, 59, 119)}
#upper = {'red':(186,255,255), 'green':(86,255,255), 'yellow':(54,255,255)}

colors = {'red':(0,0,255), 'green':(0,255,0), 'yellow':(0, 255, 217)}

def circle_color_detect():
    for key, value in upper.items():
            #construct a mask for the color dict then erode mask to remove speckles/distortions in the mask
            kernel = np.ones((9,9),np.uint8)
            mask = cv2.inRange(hsv, lower[key], upper[key])
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            #find contours in the mask and initialize the current center of the circle
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            center = None

            # proceed only if one contour is found
            if len(cnts) > 0:
                #find the largest contour in the mask, then compute the minimum enclosing circle and center
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # proceed if circle radius meets min size
                if radius > 0.5:
                    cv2.circle(frame, (int(x), int(y)), int(radius), colors[key], 2)
                    cv2.putText(frame,key + " light", (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[key],2)
                    #template for programming based on traffic light state
                    if(key == 'red'):
                        print("STOP")
                    elif(key == 'yellow'):
                        print("SLOW DOWN")
                    elif(key == 'green'):
                        print("GO")
                    else:
                        pass

def stop_sign_detect():
        label = "Stop Sign"
        #Haar cascade detection
        stop_signs = stop_signCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )

    # Draw a rectangle around the faces
        for (x, y, w, h) in stop_signs:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                Y = y - 15 if y - 15 > 15 else y + 15
                cv2.putText(frame, label, (x, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                print("STOP SIGN")

def traffic_light_detect():
            label = "Traffic Light"
            #Haar cascade detection
            traffic_lights = traffic_lightsCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )

            #circle_color_detect()

    # Draw a rectangle around the faces and detect colored circles for traffic light status
            for (x, y, w, h) in traffic_lights:
                circle_color_detect()
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                Y = y - 15 if y - 15 > 15 else y + 15
                cv2.putText(frame, label, (x, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 2, cv2.LINE_AA)
                

# capture frames from the camera
while True:

        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        frame = vs.read()
        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
        # detection functions are placed here
        stop_sign_detect()
        traffic_light_detect()
        #circle_color_detect()
        
        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # clear the stream in preparation for the next frame


        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
                break

vs.stop()

cv2.destroyAllWindows()