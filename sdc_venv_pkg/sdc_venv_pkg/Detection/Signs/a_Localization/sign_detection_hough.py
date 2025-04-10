import cv2
import numpy as np

def sign_detection_and_track(gray_img, frame, frame_draw):

    # gray_img: The input image (grayscale).
    # cv2.HOUGH_GRADIENT: The detection method (gradient-based method).
    # 1: The inverse ratio of resolution (1 means no change).
    # 100: Minimum distance between circle centers.
    # param1=250: Higher threshold for edge detection.
    # param2=30: Threshold for circle center detection.
    # minRadius=10, maxRadius=100: The minimum and maximum radius for the circles to be detected.
    circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 100, param1=250, param2=30, minRadius=10, maxRadius=100)

    # cv2.HoughCircles, the result circles is a NumPy array of shape (1, N, 3)
    # 1: This dimension exists because cv2.HoughCircles returns a list of circle arrays, even if there’s only one set of circles detected.
    # N: The number of detected circles.
    # 3: Each detected circle is represented by 3 values: [x_center, y_center, radius].

    # >>>>>>>>>>>>>>>>>>> Step1: Check if any circle regions were localized >>>>>>>>>>>>>>>>>>>
    if circles is not None:
        circles = np.uint16(np.around(circles)) # Round the Circle Parameters into integer
        
        # >>>>>>>>>>>>>>>>>>> step2: Lopping over each localized circle and extract its center and radius >>>>>>>>>>>>>>>>>>>
        # By using circles[0, :], the shape becomes (N, 3) (a 2D array), making it easier to iterate through the individual circle parameters.
        for i in circles[0, :]: # i represents one circle: [x_center, y_center, radius]
            center = (i[0], i[1]) # (x,y)
            radius = i[2]

            # >>>>>>>>>>>>>>>>>>> step3: Extracting ROI from the localized circle >>>>>>>>>>>>>>>>>>>
            try:
                startP = (center[0]-radius, center[1]-radius) # The top-left corner of the bounding square of the circle.
                endP = (center[0]+radius, center[1]+radius) # The bottom-right corner of the bounding square.
                print("startP", startP, " endP", endP)
                localization_sign = frame[startP[1] : endP[1], startP[0] : endP[0]] # startP[1] : endP[1] → Vertical range (y-axis) & startP[0] : endP[0] → Horizontal range (x-axis).

                # >>>>>>>>>>>>>>>>>>> step4: Indicating localized potential sign on frame and also displaying seperatly >>>>>>>>>>>>>>>>>>>
                cv2.circle(frame_draw, (i[0], i[1]), i[2], (0, 255, 0), 1) # draw the outer circle
                cv2.circle(frame_draw, (i[0], i[1]), 2, (0, 0, 255), 3) # draw the center of the circle
                cv2.imshow("ROI", localization_sign)
            except Exception as e:
                print(e)
                pass
        
        cv2.imshow("Signs Localized", frame_draw)

            

def detect_signs(frame, frame_draw):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sign_detection_and_track(gray, frame, frame_draw)