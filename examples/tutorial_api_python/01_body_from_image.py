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
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # try:
    #     # Windows Import
    #     if platform == "win32":
    #         # Change these variables to point to the correct folder (Release/x64 etc.)
    #         sys.path.append(dir_path + '/../../python/openpose/Release');
    #         os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            
    #     else:
    #         # Change these variables to point to the correct folder (Release/x64 etc.)
    #         sys.path.append('../../python');
    #         # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
    #         # sys.path.append('/usr/local/python')
    #         # from openpose import pyopenpose as op
    # except ImportError as e:
    #     print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    #     raise e

    # Flags
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--image_path", default="../../examples/media/test.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    # args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "/openpose/models"

    # Add others in path?
    # for i in range(0, len(args[1])):
    #     curr_item = args[1][i]
    #     if i != len(args[1])-1: next_item = args[1][i+1]
    #     else: next_item = "1"
    #     if "--" in curr_item and "--" in next_item:
    #         key = curr_item.replace('-','')
    #         if key not in params:  params[key] = "1"
    #     elif "--" in curr_item and "--" not in next_item:
    #         key = curr_item.replace('-','')
    #         if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    cap = cv2.VideoCapture('/openpose/examples/media/test.mp4')

    while True:
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
            print('Each Person')

            # Each person
            n1_x, n1_y, n2_x, n2_y = 0, 0, 0, 0
            norm_dist = 0
            # thresh_head = 0
            
            # Nose
            n_x, n_y = 0, 0
            

            # Mid Point
            m_x, m_y = 0, 0

            for id, key_id in enumerate(person):
                # print(f'Id: {id}, Key: {key_id}')
                # if key_id[2] > 0.2:
                    
                # Normalize
                if id == 1:
                    n1_x, n1_y = key_id[0], key_id[1]
                if id == 8:    
                    n2_x, n2_y = key_id[0], key_id[1]

                # Mid Point
                if id == 1:
                    m_x, m_y = key_id[0], key_id[1]
                
                # Nose
                if id == 0:
                    n_x, n_y = key_id[0], key_id[1]
                
            ### Pre_processing
            ## head    
            if n1_x > 0 and n1_y > 0 and n2_x > 0 and n2_y > 0:
                norm_dist = np.sqrt(((n1_x-n2_x)**2 + (n1_y-n2_y)**2))
                print('Norm Dist: ', norm_dist)

            if norm_dist > 0 and m_x > 0 and m_y > 0 and n_x > 0 and n_y > 0:
                
                # Nose in side BBox
                if int(m_x-norm_dist//2) < n_x < int(m_x+norm_dist//3) and int(m_y-norm_dist//1.3) < n_y < int(m_y+norm_dist//8):
                    ## BBox - ROI
                    # Head ROI
                    cv2.rectangle(
                        imageToProcess, 
                        (int(m_x-norm_dist//2), int(m_y-norm_dist//1.3)),
                        (int(m_x+norm_dist//3), int(m_y+norm_dist//8)),
                        (0, 255, 0), 2
                    )

        # Display Image
        # print("Body keypoints: \n" + str(datum.poseKeypoints))
        cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)
        imageToProcess = cv2.resize(imageToProcess, (1000, 700))
        cv2.imshow("img", imageToProcess)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
except Exception as e:
    print(e)
    sys.exit(-1)
