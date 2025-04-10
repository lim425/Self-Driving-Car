# until classification gpt

import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# Saving Variables
save_dataset = False
iter = 0
saved_no = 0

# Model and Sign Classes
model_loaded = False  # Initially, model is not loaded
model = None
sign_classes = ["speed_sign_30", "speed_sign_60", "speed_sign_90", "stop", "left_turn", "No Sign"] # Trained CNN classes

def image_forKeras(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert back to RGB for Keras
    image = cv2.resize(image, (30, 30))  # Resize to model size requirement
    image = np.expand_dims(image, axis=0)  # Adjust dimensions for the model: [Batch size, input_row, input_col, input_channel]
    return image

def sign_detection_and_track(gray_img, frame, frame_draw):

    # Detect circles using Hough Circles, the result circles is a NumPy array of shape (1, N, 3)
    circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 100, param1=250, param2=30, minRadius=10, maxRadius=100)

    # >>>>>>>>>>>>>>>>>>> Step 2a: Check if any circle regions were localized >>>>>>>>>>>>>>>>>>>
    if circles is not None:
        circles = np.uint16(np.around(circles))  # Round the circle parameters to integers
        
        # >>>>>>>>>>>>>>>>>>> step 2b: Lopping over each localized circle and extract its center and radius >>>>>>>>>>>>>>>>>>>
        # By using circles[0, :], the shape becomes (N, 3) (a 2D array), making it easier to iterate through the individual circle parameters.
        for i in circles[0, :]:  # i represents one circle: [x_center, y_center, radius]
            center = (i[0], i[1])  # (x, y)
            radius = i[2]

            # >>>>>>>>>>>>>>>>>>> step 2c: Extracting ROI from the localized circle >>>>>>>>>>>>>>>>>>>
            try:
                startP = (center[0] - radius, center[1] - radius)  # The top-left corner of the bounding square of the circle.
                endP = (center[0] + radius, center[1] + radius)  # The bottom-right corner of the bounding square.
                print("startP", startP, " endP", endP)
                localization_sign = frame[startP[1]:endP[1], startP[0]:endP[0]]  # Extract ROI

                # >>>>>>>>>>>>>>>>>>> step 2d: Indicating localized potential sign on frame and also displaying seperatly >>>>>>>>>>>>>>>>>>>
                cv2.circle(frame_draw, (i[0], i[1]), i[2], (0, 255, 0), 1)  # Draw outer circle
                cv2.circle(frame_draw, (i[0], i[1]), 2, (0, 0, 255), 3)  # Draw center of circle
                cv2.imshow("ROI", localization_sign)

                if model is not None:
                    # Predict the sign from the localized region
                    sign = sign_classes[np.argmax(model(image_forKeras(localization_sign)))]

                    # >>>>>>>>>>>>>>>>>>> 3a. Check if classification region is a sign >>>>>>>>>>>>>>>>>>>
                    if(sign != "No_Sign"):
                        # Display the predicted sign
                        cv2.putText(frame_draw, sign, (endP[0] - 80, startP[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1) # Display class
                        cv2.circle(frame_draw, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Draw outer circle in green

                        # Saving dataset
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
                                cv2.imwrite(img_name, localization_sign)
            except Exception as e:
                print("Error", e)
                pass
        
        cv2.imshow("Signs Localized", frame_draw)

def detect_signs(frame, frame_draw):
    global model_loaded
    global model

    # Check if the model is loaded, if not load it
    if not model_loaded:
        # >>>>>>>>>>>>>>>>>>> step 1: Load CNN model >>>>>>>>>>>>>>>>>>>
        try:
            print(">>>>>>>> Loading Model >>>>>>>>")
            model = load_model("/home/junyi/potbot_venv_ws/src/sdc_venv_pkg/sdc_venv_pkg/data/saved_model.h5")
            model_loaded = True
            print("Model loaded successfully")
            model.summary()  # Print the model summary to confirm it's loaded correctly
        except Exception as e:
            print("Failed to load model:", e)
            return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sign_detection_and_track(gray, frame, frame_draw)
