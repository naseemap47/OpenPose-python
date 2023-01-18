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
import pyshine as ps


try:
    
    ############################
    white_pixel_value_head = 255
    white_pixel_value_hand = 255
    pixel_thresh_head = 1500
    pixel_thresh_hand = 1500
    ############################

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

        helmet = 0
        glove = 0

        keypoints = datum.poseKeypoints
        for person in keypoints:

            # Each person
            n1_x, n1_y, n2_x, n2_y = 0, 0, 0, 0
            norm_dist = 0

            h1_x1, h1_y1, h2_x1, h2_y1 = 0, 0, 0, 0
            
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
                        
                    # Hand - Right
                    if id == 4:    
                        h1_x1, h1_y1 = key_id[0], key_id[1]
                    
                    # Hand - Left
                    if id == 7:    
                        h2_x1, h2_y1 = key_id[0], key_id[1]
                    
                    # Head
                    if id == 17:
                        head1_x, head1_y = key_id[0], key_id[1]
                    if id == 18:    
                        head2_x, head2_y = key_id[0], key_id[1]

                
            ### Pre_processing
            if n1_x > 0 and n1_y > 0 and n2_x > 0 and n2_y > 0:
                norm_dist = np.sqrt(((n1_x-n2_x)**2 + (n1_y-n2_y)**2))
            
            if norm_dist > 0:
                try:
                    if head1_x > 0 and head2_x > 0:
                        # Mid point
                        mid_x, mid_y = int((head1_x+head2_x)/2), int((head1_y+head2_y)/2)
                    elif (head1_x > 0 and head2_x == 0):
                        mid_x, mid_y = head1_x, head1_y
                    elif (head1_x == 0 and head2_x > 0):
                        mid_x, mid_y = head2_x, head2_y
                    elif (head1_x == 0 and head2_x == 0):
                        continue

                    # Head ROI
                    head_roi = imageToProcess[int(mid_y-norm_dist//3):int(mid_y+norm_dist//3), int(mid_x-norm_dist//3):int(mid_x+norm_dist//3)]

                    # Save
                    cv2.imwrite(f"/openpose/examples/media/head/{len(os.listdir('/openpose/examples/media/head'))}.jpg", head_roi)

                    gray_hand = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
                    ret,thresh1 = cv2.threshold(gray_hand,127,255,cv2.THRESH_BINARY)

                    n_white_pix = np.sum(thresh1 == white_pixel_value_head)
                    print(n_white_pix)

                    if 2800 > n_white_pix > 1100:
                        cv2.rectangle(
                            img_copy, 
                            (int(mid_x-norm_dist//3), int(mid_y-norm_dist//3)),
                            (int(mid_x+norm_dist//3), int(mid_y+norm_dist//3)),
                            (0, 255, 0), 2
                        )
                    else:
                        cv2.rectangle(
                            img_copy, 
                            (int(mid_x-norm_dist//3), int(mid_y-norm_dist//3)),
                            (int(mid_x+norm_dist//3), int(mid_y+norm_dist//3)),
                            (0, 0, 255), 2
                        )
                except:
                    pass

            # if norm_dist > 0 and h1_x1 > 0 and h1_y1 > 0:
            #     try:
            #         # Hand ROI - Right
            #         hand1_roi = imageToProcess[int(h1_y1-norm_dist//3):int(h1_y1+norm_dist//3), int(h1_x1-norm_dist//3):int(h1_x1+norm_dist//3)]

            #         gray_hand = cv2.cvtColor(hand1_roi, cv2.COLOR_BGR2GRAY)
            #         ret,thresh1 = cv2.threshold(gray_hand,127,255,cv2.THRESH_BINARY)
            #         n_white_pix1 = np.sum(thresh1 == white_pixel_value_hand)
            #         print(n_white_pix1)

            #         if n_white_pix1 > 1500:
            #             # Head ROI
            #             cv2.rectangle(
            #                 img_copy,
            #                 (int(h1_x1-norm_dist//3), int(h1_y1-norm_dist//3)),
            #                 (int(h1_x1+norm_dist//3), int(h1_y1+norm_dist//3)),
            #                 (0, 255, 0), 2
            #             )
            #             glove += 1
            #         else:
            #             cv2.rectangle(
            #                 img_copy,
            #                 (int(h1_x1-norm_dist//3), int(h1_y1-norm_dist//3)),
            #                 (int(h1_x1+norm_dist//3), int(h1_y1+norm_dist//3)),
            #                 (0, 0, 255), 2
            #             )
            #     except:
            #         pass


            # if norm_dist > 0 and h2_x1 > 0 and h2_y1 > 0:
            #     try:

            #         # Hand ROI - Left
            #         hand2_roi = imageToProcess[int(h1_y1-norm_dist//3):int(h1_y1+norm_dist//3), int(h1_x1-norm_dist//3):int(h1_x1+norm_dist//3)]

            #         gray_hand = cv2.cvtColor(hand2_roi, cv2.COLOR_BGR2GRAY)
            #         ret,thresh1 = cv2.threshold(gray_hand,127,255,cv2.THRESH_BINARY)
            #         n_white_pix2 = np.sum(thresh1 == white_pixel_value_hand)
            #         print(n_white_pix2)

            #         if n_white_pix2 > 1500:
            #             # Head ROI
            #             cv2.rectangle(
            #             img_copy,
            #             (int(h2_x1-norm_dist//3), int(h2_y1-norm_dist//3)),
            #             (int(h2_x1+norm_dist//3), int(h2_y1+norm_dist//3)),
            #             (0, 255, 0), 2
            #         )
            #             glove += 1
            #         else:
            #             cv2.rectangle(
            #             img_copy,
            #             (int(h2_x1-norm_dist//3), int(h2_y1-norm_dist//3)),
            #             (int(h2_x1+norm_dist//3), int(h2_y1+norm_dist//3)),
            #             (0, 0, 255), 2
            #         )
            #     except:
            #         pass

        # Display Image
        # print("Body keypoints: \n" + str(datum.poseKeypoints))
        # ps.putBText(img_copy,label,text_offset_x=20,text_offset_y=40,vspace=20,hspace=10, font_scale=1.0,background_RGB=(240,16,255),text_RGB=(255,255,255))
        
        # ps.putBText(img_copy,f'Number of people detected: {len(keypoints)}',text_offset_x=20,text_offset_y=101,vspace=20,hspace=10, font_scale=1.0,background_RGB=(10,20,222),text_RGB=(255,255,255))
        # ps.putBText(img_copy,f'Number of people wearing Helmet: {helmet}',text_offset_x=20,text_offset_y=162,vspace=20,hspace=10, font_scale=1.0,background_RGB=(0,250,250),text_RGB=(255,255,255)) # 210,20,4 red
        # ps.putBText(img_copy,f'Number of people wearing Gloves: {glove}',text_offset_x=20,text_offset_y=223,vspace=20,hspace=10, font_scale=1.0,background_RGB=(228,225,222),text_RGB=(0,0,0))
        
        # cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)
        img_copy = cv2.resize(img_copy, (1000, 700))
        cv2.imshow("img", img_copy)
        # cv2.imshow("hand", hand_roi)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
except Exception as e:
    print(e)
    sys.exit(-1)