#!/usr/bin/env python
# coding: utf-8
#Anurag Kanase
#Wink Counter

# In[1]:


import json
import time
import keyboard
import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
import pafy
from ffpyplayer.player import MediaPlayer


# In[ ]:



#Use Youtube video diirectly
url = "https://www.youtube.com/watch?v=eIhp7VhLa-o"
videoPafy = pafy.new(url)
best = videoPafy.getbest()
# In[2]:


def eye_aspect_ratio(eye):
    """Computes the EAR"""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2 * C)
    return ear


# In[3]:


def play():
    """Play game"""

    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_EYE_POINTS = list(range(42, 48))

    EYE_AR_THRESH = 0.22
    EYE_AR_CONSEC_FRAMES = 3
    EAR_AVG = 0

    # detection of the facial region
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


    COUNTER = 0
    TOTAL = 0
    #You have 3 optioins:
    # Inside cv2.VideoCapture(), insert best.url for Youtbe video
    # Inside cv2.VideoCapture(), insert 0 for camera, to use your Webcam
    # Inside cv2.VideoCapture(), insert file location
    # OpenCV - live video frame
    cap = cv2.VideoCapture("hrk.mp4")#best.url)#"http://192.168.1.238:4747/video")
    player = MediaPlayer("hrk.mp4")
    result = cv2.VideoWriter('hrk.avi',cv2.VideoWriter_fourcc(*'MJPG'),30,(320, 568))
    while True:
        ret, frame = cap.read() 
        height , width , layers =  frame.shape
        new_h=int(height/2)
        new_w=int(width/2)
        frame = cv2.resize(frame, (new_w, new_h)) 
        audio_frame, val = player.get_frame()
        if ret:
            # convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            for rect in rects:
                x = rect.left()
                y = rect.top()
                x1 = rect.right()
                y1 = rect.bottom()

                landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])

                left_eye = landmarks[LEFT_EYE_POINTS]
                right_eye = landmarks[RIGHT_EYE_POINTS]

                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)

                ear_left = eye_aspect_ratio(left_eye)
                ear_right = eye_aspect_ratio(right_eye)

                ear_avg = (ear_left + ear_right) / 2.0

                # detect of the blink
                if ear_avg < EYE_AR_THRESH: 
                    COUNTER += 1
                else:
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        TOTAL += 1
                        # press space bar when blinked
                        #keyboard.press_and_release('space')
                        print("Eye blinked")
                    COUNTER = 0

                cv2.putText(frame, "# winks {}".format(TOTAL), (int(frame.shape[1]*0.3),int(frame.shape[0]*0.85)), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, "AI #1 dezember, Wink Counter".format(TOTAL), (int(frame.shape[1]*.01),int(frame.shape[0]*0.98)), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 250, 255), 1)
                if val != 'eof' and audio_frame is not None:
                    img, t = audio_frame
            cv2.imshow(" AI#1 dezember, # Wink Counter", frame)
            key = cv2.waitKey(1) & 0xFF

            if key is ord('q'):
                break
            result.write(frame)

    cap.release()
    result.release()
    cv2.destroyAllWindows()


# In[4]:

#run the program
play()    


