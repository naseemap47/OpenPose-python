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
    dir_path = os.path.dirname(os.path.realpath(__file__))
    params = dict()
    params["model_folder"] = "/openpose/models"
    
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    cap = cv2.VideoCapture('/openpose/examples/media/00000001541000000.mp4')
    count = 0
    
    while True:
        count += 1
        success, imageToProcess = cap.read()
        img_copy = imageToProcess.copy()
        if not success:
            print('[INFO] Failed to read Video')
            break
    
        # Process Image
        datum = op.Datum()
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])

        keypoints = datum.poseKeypoints
        for person in keypoints:

            # Each person
            points_x = []
            points_y = []
            n1_x, n1_y, n2_x, n2_y = 0, 0, 0, 0
            norm_dist = 0

            h1_x1, h1_y1, h2_x1, h2_y1 = 0, 0, 0, 0

            for id, key_id in enumerate(person):
                # print(f'Id: {id}, Key: {key_id}')
                if key_id[2] > 0.5:

                    # Normalize
                    if id == 1:
                        n1_x, n1_y = key_id[0], key_id[1]
                    if id == 8:    
                        n2_x, n2_y = key_id[0], key_id[1]
                        
                    # Hand - Right
                    if id == 4:    
                        h1_x1, h1_y1 = key_id[0], key_id[1]
                    
                    # Hand - Left
                    if id == 7:    
                        h2_x1, h2_y1 = key_id[0], key_id[1]
                    
                    # Mid Point
                    if id == 1:
                        m_x, m_y = key_id[0], key_id[1]
                    
                    # Nose
                    if id == 0:
                        n_x, n_y = key_id[0], key_id[1]

                
            ### Pre_processing
            if n1_x > 0 and n1_y > 0 and n2_x > 0 and n2_y > 0:
                norm_dist = np.sqrt(((n1_x-n2_x)**2 + (n1_y-n2_y)**2))
                # head = (n1_x, int(n1_y-norm_dist*0.4))
                # offset = 
                # print('Norm Dist: ', norm_dist)

            if norm_dist > 0 and m_x > 0 and m_y > 0:
                
                # Nose in side BBox
                # if int(m_x-norm_dist//2) < n_x < int(m_x+norm_dist//3) and int(m_y-norm_dist//1.3) < n_y < int(m_y+norm_dist//8):
                ## BBox - ROI
                # Head ROI
                cv2.rectangle(
                    img_copy, 
                    (int(m_x-norm_dist//2), int(m_y-norm_dist//1.3)),
                    (int(m_x+norm_dist//3), int(m_y+norm_dist//8)),
                    (0, 255, 0), 2
                )

                # Head ROI
                path_to_save_dir_head = '/openpose/examples/media/head'
                hand_roi = imageToProcess[int(m_y-norm_dist//1.3):int(m_y+norm_dist//8), int(m_x-norm_dist//2):int(m_x+norm_dist//3)]
                
                # save
                try:
                    if count % 10 == 0:
                        path_name = f'{path_to_save_dir_head}/{len(os.listdir(path_to_save_dir_head))}.jpg'
                        cv2.imwrite(path_name, hand_roi)
                        print(f'[INFO] Successfully saved {path_name}')
                except:
                    print(f'[INFO] Failed to save {path_name}')
                

            if norm_dist > 0 and h1_x1 > 0 and h1_y1 > 0:
                cv2.rectangle(
                    img_copy,
                    (int(h1_x1-norm_dist//3), int(h1_y1-norm_dist//3)),
                    (int(h1_x1+norm_dist//3), int(h1_y1+norm_dist//3)),
                    (0, 255, 0), 2
                )
                ## BBox - ROI
                # Hand ROI
                path_to_save_dir_hand = '/openpose/examples/media/hand'
                hand1_roi = imageToProcess[int(h1_y1-norm_dist//3):int(h1_y1+norm_dist//3), int(h1_x1-norm_dist//3):int(h1_x1+norm_dist//3)]
                
                # save
                try:
                    if count % 10 == 0:
                        path_name = f'{path_to_save_dir_hand}/{len(os.listdir(path_to_save_dir_hand))}.jpg'
                        cv2.imwrite(path_name, hand1_roi)
                        print(f'[INFO] Successfully saved {path_name}')
                except:
                    print(f'[INFO] Failed to save {path_name}')


            if norm_dist > 0 and h2_x1 > 0 and h2_y1 > 0:
                cv2.rectangle(
                    img_copy,
                    (int(h2_x1-norm_dist//3), int(h2_y1-norm_dist//3)),
                    (int(h2_x1+norm_dist//3), int(h2_y1+norm_dist//3)),
                    (0, 255, 0), 2
                )

                ## BBox - ROI
                # Hand ROI
                path_to_save_dir_hand = '/openpose/examples/media/hand'
                hand2_roi = imageToProcess[int(h1_y1-norm_dist//3):int(h1_y1+norm_dist//3), int(h1_x1-norm_dist//3):int(h1_x1+norm_dist//3)]
                
                # save
                try:
                    if count % 10 == 0:
                        path_name = f'{path_to_save_dir_hand}/{len(os.listdir(path_to_save_dir_hand))}.jpg'
                        cv2.imwrite(path_name, hand2_roi)
                        print(f'[INFO] Successfully saved {path_name}')
                except:
                    print(f'[INFO] Failed to save {path_name}')    
                

        # Display Image
        # print("Body keypoints: \n" + str(datum.poseKeypoints))
        # cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)
        img_copy = cv2.resize(img_copy, (1000, 700))
        cv2.imshow("img", img_copy)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
except Exception as e:
    print(e)
    sys.exit(-1)
