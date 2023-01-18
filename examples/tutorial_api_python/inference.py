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
    dir_path = os.path.dirname(os.path.realpath(__file__))
    params = dict()
    params["model_folder"] = "/openpose/models"
    
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Model
    # saved_model = tf.keras.models.load_model('model.h5')
    # img_size = 240
    # class_names = ['withGlove', 'withHelmet', 'withoutGlove', 'withoutHelmet']

    cap = cv2.VideoCapture('/openpose/examples/tutorial_api_python/00000001541000000.mp4')
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
            if norm_dist > 0 and m_x > 0 and m_y > 0:
                try:
                    # Head ROI
                    hand_roi = imageToProcess[int(m_y-norm_dist//1.3):int(m_y+norm_dist//8), int(m_x-norm_dist//2):int(m_x+norm_dist//3)]
                    gray_hand = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
                    ret,thresh1 = cv2.threshold(gray_hand,127,255,cv2.THRESH_BINARY)
                    n_white_pix = np.sum(thresh1 == 255)
                    print(n_white_pix)

                    # Load Model
                    # img_resize = cv2.resize(hand_roi, (img_size, img_size))
                    # h, w, _ = img_copy.shape
                    # img_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
                    # img = tf.keras.preprocessing.image.img_to_array(img_rgb)
                    # img = np.expand_dims(img, axis=0)
                    # img = tf.keras.applications.efficientnet.preprocess_input(img)
                    # prediction = saved_model.predict(img)[0]
                    # predict = class_names[prediction.argmax()]

                    if n_white_pix > 1500:
                        # Head ROI
                        cv2.rectangle(
                            img_copy, 
                            (int(m_x-norm_dist//2), int(m_y-norm_dist//1.3)),
                            (int(m_x+norm_dist//3), int(m_y+norm_dist//8)),
                            (0, 255, 0), 2
                        )
                        helmet += 1
                    else:
                        cv2.rectangle(
                            img_copy, 
                            (int(m_x-norm_dist//2), int(m_y-norm_dist//1.3)),
                            (int(m_x+norm_dist//3), int(m_y+norm_dist//8)),
                            (0, 0, 255), 2
                        )
                except:
                    pass
                

            if norm_dist > 0 and h1_x1 > 0 and h1_y1 > 0:
                try:
                    # Hand ROI
                    hand1_roi = imageToProcess[int(h1_y1-norm_dist//3):int(h1_y1+norm_dist//3), int(h1_x1-norm_dist//3):int(h1_x1+norm_dist//3)]

                    gray_hand = cv2.cvtColor(hand1_roi, cv2.COLOR_BGR2GRAY)
                    ret,thresh1 = cv2.threshold(gray_hand,127,255,cv2.THRESH_BINARY)
                    n_white_pix1 = np.sum(thresh1 == 255)
                    print(n_white_pix1)

                    if n_white_pix1 > 1500:
                        # Head ROI
                        cv2.rectangle(
                            img_copy,
                            (int(h1_x1-norm_dist//3), int(h1_y1-norm_dist//3)),
                            (int(h1_x1+norm_dist//3), int(h1_y1+norm_dist//3)),
                            (0, 255, 0), 2
                        )
                        glove += 1
                    else:
                        cv2.rectangle(
                            img_copy,
                            (int(h1_x1-norm_dist//3), int(h1_y1-norm_dist//3)),
                            (int(h1_x1+norm_dist//3), int(h1_y1+norm_dist//3)),
                            (0, 0, 255), 2
                        )
                except:
                    pass


            if norm_dist > 0 and h2_x1 > 0 and h2_y1 > 0:
                try:

                    # Hand ROI
                    hand2_roi = imageToProcess[int(h1_y1-norm_dist//3):int(h1_y1+norm_dist//3), int(h1_x1-norm_dist//3):int(h1_x1+norm_dist//3)]

                    gray_hand = cv2.cvtColor(hand2_roi, cv2.COLOR_BGR2GRAY)
                    ret,thresh1 = cv2.threshold(gray_hand,127,255,cv2.THRESH_BINARY)
                    n_white_pix2 = np.sum(thresh1 == 255)
                    print(n_white_pix2)

                    if n_white_pix2 > 1500:
                        # Head ROI
                        cv2.rectangle(
                        img_copy,
                        (int(h2_x1-norm_dist//3), int(h2_y1-norm_dist//3)),
                        (int(h2_x1+norm_dist//3), int(h2_y1+norm_dist//3)),
                        (0, 255, 0), 2
                    )
                        glove += 1
                    else:
                        cv2.rectangle(
                        img_copy,
                        (int(h2_x1-norm_dist//3), int(h2_y1-norm_dist//3)),
                        (int(h2_x1+norm_dist//3), int(h2_y1+norm_dist//3)),
                        (0, 0, 255), 2
                    )
                except:
                    pass

        # Display Image
        # print("Body keypoints: \n" + str(datum.poseKeypoints))
        # ps.putBText(img_copy,label,text_offset_x=20,text_offset_y=40,vspace=20,hspace=10, font_scale=1.0,background_RGB=(240,16,255),text_RGB=(255,255,255))
        ps.putBText(img_copy,f'Number of people detected: {len(keypoints)}',text_offset_x=20,text_offset_y=101,vspace=20,hspace=10, font_scale=1.0,background_RGB=(10,20,222),text_RGB=(255,255,255))
        ps.putBText(img_copy,f'Number of people wearing Helmet: {helmet}',text_offset_x=20,text_offset_y=162,vspace=20,hspace=10, font_scale=1.0,background_RGB=(0,250,250),text_RGB=(255,255,255)) # 210,20,4 red
        ps.putBText(img_copy,f'Number of people wearing Gloves: {glove}',text_offset_x=20,text_offset_y=223,vspace=20,hspace=10, font_scale=1.0,background_RGB=(228,225,222),text_RGB=(0,0,0))
        cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)
        img_copy = cv2.resize(img_copy, (1000, 700))
        cv2.imshow("img", img_copy)
        # cv2.imshow("hand", hand_roi)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
except Exception as e:
    print(e)
    sys.exit(-1)