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
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            # from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../models/"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    keypoints = datum.poseKeypoints
    for person in keypoints:
        # print(len(person))
        print('Each Person')

        # Each person
        n1_x, n1_y, n2_x, n2_y = 0, 0, 0, 0
        
        # Head
        head1_x, head1_y, head2_x, head2_y = 0, 0, 0, 0

        for id, key_id in enumerate(person):
            # print(f'Id: {id}, Key: {key_id}')
            if key_id[2] > 0.7:
                
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
            
        norm_dist = np.sqrt(((n1_x-n2_x)**2 + (n1_y-n2_y)**2))
        print('Norm Dist: ', norm_dist)

        # Head Cap
        head_dist = np.sqrt(((head1_x-head2_x)**2 + (head1_y-head2_y)**2))
        print('Head Dist: ', head_dist)
        thresh_head = head_dist/norm_dist
        print('Thresh: ', thresh_head)

        # cv2.rectangle(
        #     datum.cvOutputData, ()
        # )

    # Display Image
    # print("Body keypoints: \n" + str(datum.poseKeypoints))
    cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
except Exception as e:
    print(e)
    sys.exit(-1)
