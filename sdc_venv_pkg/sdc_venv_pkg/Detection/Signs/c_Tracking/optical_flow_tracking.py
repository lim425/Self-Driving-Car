import math
import numpy as np
import cv2
from sdc_venv_pkg.config import config

class Tracker:
    def __init__(self):
        print("Initialized Object of Sign Tracking Class")
        # State variables
        self.mode = "Detection"
        self.Tracked_class = 0
        # Proximity variable, all the detection has done in previous frame
        self.known_centers = []
        self.known_centers_confidence = []
        self.known_centers_classes_confidence = []
        # Init variables
        self.old_gray = 0
        self.p0 = []
        # Draw variables
        self.mask = 0
        self.color = np.random.randint(0, 255, (100, 3))

    
    # Variable shared across instance of class
    max_allowed_dist = 100 # Allowed distance between two detected ROI to be considered same object
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) # lucas kanade optical flow parameter

    def Distance(self, a, b):
        # return math.sqrt( ( (float(a[1])-float(b[1]))**2 ) + ( (float(a[0])-float(b[0]))**2 ) )
        return math.sqrt( ( (float(a[1])-float(b[1]))**2 ) + ( (float(a[0])-float(b[0]))**2 ) )
    
    # Check the validity of previous detection
    # compare the current center with previous known center
    def MatchCurrCenter_ToKnown(self, center):
        match_found = False
        match_idx = 0
        for i in range(len(self.known_centers)):
            if (self.Distance(center, self.known_centers[i]) < self.max_allowed_dist):
                match_found = True
                match_idx = i
                return match_found, match_idx
            
        # If no match found, then return default value
        return match_found, match_idx
    
    def init_tracker(self, sign, gray, frame_draw, startP, endP):
        
        sign_mask = np.zeros_like(gray)
        sign_mask[startP[1]:endP[1], startP[0]:endP[0]] = 255
        self.mode = "Tracking"
        self.Tracked_class = sign
        self.old_gray = gray

        self.p0 = cv2.goodFeaturesToTrack(gray, mask=sign_mask, **self.feature_params)



    def track(self, gray, frame_draw):
        p1, status, _ = cv2.calcOpticalFlowPyrLK(self.old_gray, gray, self.p0, **self.lk_params)

        # If no flow, look for new points
        if ( (p1 is None) or (len(p1[status==1])<3) ):
            # if p1 is None:
            self.mode = "Detection"
            self.mask = np.zeros_like(frame_draw)
            self.Reset()
        # If flow, extract good points ... update SignTrack Class
        else:
            # Select good points
            good_new = p1[status == 1]
            good_old = self.p0[status == 1]
            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = (int(x) for x in new.ravel())
                c, d = (int(x) for x in old.ravel())
                self.mask = cv2.line(self.mask, (a,b), (c,d), self.color[i].tolist(), 2)
                frame_draw = cv2.circle(frame_draw, (a,b), 5, self.color[i].tolist(), -1)
            frame_draw_ = frame_draw + self.mask # Display the image with the flow lines
            np.copyto(frame_draw, frame_draw_) # Important to copy the data to the same address as frame_draw
            self.old_gray = gray.copy() # Update the previous frame and previous points
            self.p0 = good_new.reshape(-1, 1, 2)

    def Reset(self):
        self.known_centers = []
        self.known_centers_confidence = []
        self.known_centers_classes_confidence = []
        self.old_gray = 0
        self.p0 = []

