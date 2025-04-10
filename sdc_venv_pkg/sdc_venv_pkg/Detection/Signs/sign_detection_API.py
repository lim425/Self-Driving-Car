import cv2
import numpy as np
import os
import math
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from sdc_venv_pkg.config import config

# Saving Variables
save_dataset = True
iter = 0
saved_no = 0


# Model and Sign Classes
model_loaded = False # Initially, model is not loaded
model = None
sign_classes = ["speed_sign_30","speed_sign_60","speed_sign_90","stop","left_turn","No_Sign"] # Trained CNN Classes


class SignTracking:

    def __init__(self):
        print("Initialized Object of signTracking class")

    mode = "Detection"

    max_allowed_dist = 100
    feature_params = dict(maxCorners=100,qualityLevel=0.3,minDistance=7,blockSize=7)
    lk_params = dict(winSize=(15, 15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10, 0.03))  
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    known_centers = []
    known_centers_confidence = []
    old_gray = 0
    p0 = []
    # [NEW]: If no Sign Tracked ==> Then default is Unknown
    Tracked_class = "Unknown"
    mask = 0

    def Distance(self,a,b):
        #return math.sqrt( ( (a[1]-b[1])**2 ) + ( (a[0]-b[0])**2 ) )
        return math.sqrt( ( (float(a[1])-float(b[1]))**2 ) + ( (float(a[0])-float(b[0]))**2 ) )

    def MatchCurrCenter_ToKnown(self,center):
        match_found = False
        match_idx = 0
        for i in range(len(self.known_centers)):
            if ( self.Distance(center,self.known_centers[i]) < self.max_allowed_dist ):
                match_found = True
                match_idx = i
                return match_found, match_idx
        # If no match found as of yet return default values
        return match_found, match_idx

    def Reset(self):
        
        self.known_centers = []
        self.known_centers_confidence = []
        self.old_gray = 0
        self.p0 = []

signTrack = SignTracking()

def image_forKeras(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)# Convert back to RGB for Keras
    image = cv2.resize(image,(30,30)) # Resize to model size requirement
    image = np.expand_dims(image, axis=0) # Adjust dimensions for the model: [Batch size, input_row, input_col, input_channel]
    return image

def sign_detection_and_tracking(gray,cimg,frame_draw,model):
    
    # IF Mode of SignTrack is Detection , Proceed
    if (signTrack.mode == "Detection"):
        cv2.putText(frame_draw,"Sign Detected => "+str(signTrack.Tracked_class),(20,60),cv2.FONT_HERSHEY_PLAIN, 1,(255,0,0),1)

        # gray_img: The input image (grayscale)
        # cv2.HOUGH_GRADIENT: The detection method (gradient-based method)
        # 1: The inverse ratio of resolution (1 means no change)
        minDistBtwCircles = 100 # Minimum distance between circle centers
        CannyHighthresh = 250 # Higher threshold for edge detection
        NumOfVotesForCircle = 30 # Threshold for circle center detection
        minRadius = 10 # The minimum radius for the circles to be detected.
        maxRadius = 100 # The maximum radius for the circles to be detected.
                        # As signs are right besides road so they will eventually be in view so ignore circles larger than said limit

        
        # >>>>>>>>>>>>>>>>>>> Step 2a: Detection (Localization) >>>>>>>>>>>>>>>>>>>
        # cv2.HoughCircles, the result circles is a NumPy array of shape (1, N, 3)
        # 1: This dimension exists because cv2.HoughCircles returns a list of circle arrays, even if there’s only one set of circles detected.
        # N: The number of detected circles.
        # 3: Each detected circle is represented by 3 values: [x_center, y_center, radius].
        circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,minDistBtwCircles,param1=CannyHighthresh,param2=NumOfVotesForCircle,minRadius=minRadius,maxRadius=maxRadius)

        # >>>>>>>>>>>>>>>>>>> Step 2b: Detection (Localization): Checking if circular regions were localized >>>>>>>>>>>>>>>>>>>
        if circles is not None:
            circles = np.uint16(np.around(circles))

            # >>>>>>>>>>>>>>>>>>> Step 2c: Detection (Localization): Looping over each localized circle >>>>>>>>>>>>>>>>>>>
            # By using circles[0, :], the shape becomes (N, 3) (a 2D array), making it easier to iterate through the individual circle parameters.
            for i in circles[0,:]: # i represents one circle: [x_center, y_center, radius]
                center =(i[0],i[1]) # (x, y)
                radius = i[2]
                match_found,match_idx = signTrack.MatchCurrCenter_ToKnown(center)

                
                try:
                    startP = (center[0] - radius, center[1] - radius)  # The top-left corner of the bounding square of the circle.
                    endP = (center[0] + radius, center[1] + radius)  # The bottom-right corner of the bounding square.
                    
                    # >>>>>>>>>>>>>>>>>>> step 2d: Detection (Localization) Extracting Roi from localized circle >>>>>>>>>>>>>>>>>>>
                    detected_sign = cimg[startP[1]:endP[1],startP[0]:endP[0]]
                    
                    # >>>>>>>>>>>>>>>>>>> step 2e: Detection (Localization) Indicating localized potential sign on frame and also displaying seperatly >>>>>>>>>>>>>>>>>>>
                    cv2.circle(frame_draw, (i[0], i[1]), i[2], (0, 255, 0), 1)  # Draw outer circle
                    cv2.circle(frame_draw, (i[0], i[1]), 2, (0, 0, 255), 3)  # Draw center of circle
                    cv2.imshow("ROI", detected_sign)


                    # >>>>>>>>>>>>>>>>>>> step 3a: Detection (Classification) Classifying sign in the ROi >>>>>>>>>>>>>>>>>>>
                    if model is not None:
                        # Predict the sign from the localized region
                        sign = sign_classes[np.argmax(model(image_forKeras(detected_sign)))]

                        # Check if Classified Region is a Sign
                        if(sign != "No_Sign"):
                            # Display the predicted sign
                            cv2.putText(frame_draw, sign, (endP[0] - 80, startP[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1) # Display class
                            cv2.circle(frame_draw, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Draw outer circle in green
                            

                            # >>>>>>>>>>>>>>>>>>> step 4a: Detection (Tracking): If match found , Increment ... known centers confidence >>>>>>>>>>>>>>>>>>>
                            if match_found:
                                signTrack.known_centers_confidence[match_idx] += 1

                                # >>>>>>>>>>>>>>>>>>> 4b. Detection (Tracking): Check if same sign detected 3 times, if yes initialize Optical Flow Tracker >>>>>>>>>>>>>>>>>>>
                                if(signTrack.known_centers_confidence[match_idx] > 3):
                                    # cv2.imshow("Detected_SIGN",detected_sign)
                                    circle_mask = np.zeros_like(gray)
                                    circle_mask[startP[1]:endP[1],startP[0]:endP[0]] = 255
                                    if not config.Training_CNN:
                                        signTrack.mode = "Tracking" # Set mode to tracking
                                    signTrack.Tracked_class = sign # keep tracking frame sign name
                                    signTrack.old_gray = gray.copy()
                                    signTrack.p0 = cv2.goodFeaturesToTrack(signTrack.old_gray, mask=circle_mask, **signTrack.feature_params)
                                    signTrack.mask = np.zeros_like(frame_draw)

                            # >>>>>>>>>>>>>>>>>>> 4b. Detection (Tracking): If sign detected first time ... Update sign detection and its detected count >>>>>>>>>>>>>>>>>>>
                            else:
                                signTrack.known_centers.append(center)
                                signTrack.known_centers_confidence.append(1)

                            

                        # Saving sign dataset
                        if save_dataset:
                            global iter, saved_no
                            iter += 1
                            # Save every 5th image
                            if((iter%5) == 0):
                                saved_no += 1
                                img_dir = "/home/junyi/potbot_venv_ws/src/sdc_venv_pkg/sdc_venv_pkg/data/live_dataset"
                                img_name = img_dir + "/" + str(saved_no) + ".png"
                                if not os.path.exists(img_dir):
                                    os.makedirs(img_dir)
                                cv2.imwrite(img_name, detected_sign)

                except Exception as e:
                    print("Error", e)
                    pass
            
            
            # cv2.imshow("detected Signs",frame_draw)

    # >>>>>>>>>>>>>>>>>>> 5a. Detection (Tracking): IF Mode of SignTrack is Tracking , Proceed >>>>>>>>>>>>>>>>>>>
    else:
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(signTrack.old_gray, gray, signTrack.p0, None,**signTrack.lk_params)
        
        # If no flow, look for new points
        if ( (p1 is None) or ( len(p1[st == 1])<3 ) ):
        #if p1 is None:
            signTrack.mode = "Detection"
            signTrack.mask = np.zeros_like(frame_draw)
            signTrack.Reset()

        # If flow , Extract good points ... Update SignTrack class
        else:
            # Select good points
            good_new = p1[st == 1]
            good_old = signTrack.p0[st == 1]
            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = (int(x) for x in new.ravel())
                c, d = (int(x) for x in old.ravel())
                signTrack.mask = cv2.line(signTrack.mask, (a, b), (c, d), signTrack.color[i].tolist(), 2)
                frame_draw = cv2.circle(frame_draw, (a, b), 5, signTrack.color[i].tolist(), -1)
            frame_draw_ = frame_draw + signTrack.mask # Display the image with the flow lines
            np.copyto(frame_draw,frame_draw_) #important to copy the data to same address as frame_draw
            signTrack.old_gray = gray.copy()  # Update the previous frame and previous points
            signTrack.p0 = good_new.reshape(-1, 1, 2)
    
           

def detect_signs(frame,frame_draw):
    """Extract required data from the traffic signs on the road

    Args:
        frame (numpy nd array): Prius front-cam view
        frame_draw (numpy nd array): for displaying detected signs

    Returns:
        string: Current mode of signtracker class
        string: detected speed sign (e.g speed sign 30)
    """    
        
    global model_loaded
    if not model_loaded:
        try:
            print(">>>>>>>> Loading Model >>>>>>>>")

            # >>>>>>>>>>>>>>>>>>> step 1: Load CNN model >>>>>>>>>>>>>>>>>>>
            global model
            model = load_model("/home/junyi/potbot_venv_ws/src/sdc_venv_pkg/sdc_venv_pkg/data/saved_model2.h5")

            # summarize model.
            model.summary()
            model_loaded = True
        except Exception as e:
            print("Failed to load model:", e)
            return


    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Localizing Potetial Candidates and Classifying them in SignDetection
    cv2.putText(frame_draw,signTrack.mode,(20,85),cv2.FONT_HERSHEY_PLAIN, 1,(0,0,0),1)
    print("Mode: ", signTrack.mode)
    print("Sign: ", signTrack.Tracked_class)
    sign_detection_and_tracking(gray.copy(),frame.copy(),frame_draw,model)

    return signTrack.mode , signTrack.Tracked_class