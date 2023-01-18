# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import pyopenpose as op
import numpy as np
import math
import cv2


try:
    params = dict()
    params["model_folder"] = "/openpose/models"

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    cap = cv2.VideoCapture('/openpose/examples/media/test.mp4')
    count = 0
    while True:
        count += 1
        success, imageToProcess = cap.read()
        if not success:
            print('[INFO] Failed to read Video')
            break
        # Process Image
        datum = op.Datum()
        # imageToProcess = cv2.imread(img)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])

        keypoints = datum.poseKeypoints
        for person in keypoints:
            # print(len(person))
            # print('Each Person')

            # Each person
            points_x = []
            points_y = []
            n1_x, n1_y, n2_x, n2_y = 0, 0, 0, 0
            norm_dist = 0
            
            # Head
            head1_x, head1_y, head2_x, head2_y = 0, 0, 0, 0

            for id, key_id in enumerate(person):
                # print(f'Id: {id}, Key: {key_id}')
                if key_id[2] > 0.5:

                    # Normalize
                    if id == 1:
                        n1_x, n1_y = key_id[0], key_id[1]
                    if id == 8:    
                        n2_x, n2_y = key_id[0], key_id[1]
                        
                    # Head
                    if id == 17:
                        head1_x, head1_y = key_id[0], key_id[1]
                    if id == 18:    
                        head2_x, head2_y = key_id[0], key_id[1]
                
            ### Pre_processing
            if n1_x > 0 and n1_y > 0 and n2_x > 0 and n2_y > 0:
                norm_dist = np.sqrt(((n1_x-n2_x)**2 + (n1_y-n2_y)**2))
                # print('Norm Dist: ', norm_dist)

            if norm_dist > 0:
                if head1_x > 0 and head2_x > 0:
                    # Mid point
                    mid_x, mid_y = int((head1_x+head2_x)/2), int((head1_y+head2_y)/2)
                elif (head1_x > 0 and head2_x == 0):
                    mid_x, mid_y = head1_x, head1_y
                elif (head1_x == 0 and head2_x > 0):
                    mid_x, mid_y = head2_x, head2_y
                elif (head1_x == 0 and head2_x == 0):
                    continue
                # head_dist = np.sqrt(((head1_x-head2_x)**2 + (head1_y-head2_y)**2))
                # print('Head Dist: ', head_dist)
                # thresh_head = head_dist/norm_dist
                # print('Thresh: ', thresh_head)

                cv2.rectangle(
                    imageToProcess, 
                    (int(mid_x-norm_dist//3), int(mid_y-norm_dist//3)),
                    (int(mid_x+norm_dist//3), int(mid_y+norm_dist//3)),
                    (0, 255, 0), 2
                )

                print(
                    'head1_x, head1_y, head2_x, head2_y\n',
                    head1_x, head1_y, head2_x, head2_y
                )

                # cv2.circle(
                #     imageToProcess, (mid_x, mid_y),
                #     5, (0, 255, 0), cv2.FILLED
                # )
    

        # Display Image
        # print("Body keypoints: \n" + str(datum.poseKeypoints))
        # cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)
        imageToProcess = cv2.resize(imageToProcess, (1000, 700))
        cv2.imshow("img", imageToProcess)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
except Exception as e:
    print(e)
    sys.exit(-1)
