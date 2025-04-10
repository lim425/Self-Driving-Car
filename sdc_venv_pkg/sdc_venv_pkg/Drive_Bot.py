import cv2
import numpy as np

# venv lib path
from sdc_venv_pkg.Detection.Lanes.lane_detection import detect_lanes
# from sdc_venv_pkg.Detection.Signs.c_Tracking.sign_detection_tracking import detect_signs
from sdc_venv_pkg.Detection.Signs.sign_detection_API import detect_signs



class Control():

    def __init__(self):
        # Lane assist variable
        self.angle = 0.0
        self.speed = 80.0
        self.motor_speed = self.speed

        # Cruise control variable
        self.prev_Mode = "Detection"
        self.IncreaseTireSpeedInTurns = False

        # Navigate T-junction variable
        self.prev_Mode_LT = "Detection"
        self.Left_turn_iterations = 0
        self.Frozen_Angle = 0
        self.Detected_LeftTurn = False
        self.Activate_LeftTurn = False


    def follow_lane(self, max_sane_dist, dist, curv, mode, tracked_class):
        
        # Cruise control speed adjustment to match the road speed limit
        if((tracked_class != 0) and (self.prev_Mode == "Tracking") and (mode == "Detection")):
            if (tracked_class == "speed_sign_30"):
                self.speed = 30
                print("speed sign 30 detected")
            elif (tracked_class == "speed_sign_60"):
                self.speed = 60
                print("speed sign 60 detected")
            elif (tracked_class == "speed_sign_90"):
                self.speed = 90
                print("speed sign 90 detected")
            elif (tracked_class == "stop"):
                self.speed = 0
                print("Stopping Car")

        self.prev_Mode = mode # set the previous mode to current mode

        max_turn_angle = 90
        max_turn_angle_neg = -90
        req_turn_angle = 0

        # Positive dist: The car center is at the left of the lane center.
        # Negative dist: The car center is at the right of the lane center.
        # Positive curv: Lane curves right.
        # Negative curv: Lane curves left.

        # Not normal condition: The car's offset exceeds the maximum "sane" range (either too far left or too far right)
        if ((dist > max_sane_dist) or (dist < (-1) * max_sane_dist)):
            if (dist > max_sane_dist): # car is too far left
                req_turn_angle = max_turn_angle + curv # The car should turn maximum right, this attempts to bring the car sharply back to the lane.
            else: # car is too far right
                req_turn_angle = max_turn_angle_neg + curv
        # normal condition
        else: 
            car_offset_angle = np.interp(dist, [-max_sane_dist, max_sane_dist], [-max_turn_angle ,max_turn_angle]) # calculate a proportional turning angle based on the car's lateral offset (dist)
            req_turn_angle = car_offset_angle + curv # Add curv to car_offset to account for lane curvature.

        # Handle overflow - Ensure the required turn angle (req_turn_angle) stays within the allowable range ([-max_turn_angle, max_turn_angle])
        # Prevents oversteering
        if ((req_turn_angle > max_turn_angle) or (req_turn_angle < max_turn_angle_neg)):
            if (req_turn_angle > max_turn_angle):
                req_turn_angle = max_turn_angle
            else:
                req_turn_angle = max_turn_angle_neg

        
        # Handle max car turn ability: the car can only turn maximum 45 degree in both direction
        self.angle = np.interp(req_turn_angle, [max_turn_angle_neg, max_turn_angle], [-45, 45])
        
        # The car increases speed on smooth turns for better performance.
        if (self.IncreaseTireSpeedInTurns and (tracked_class != "left_turn")):
            if (self.angle > 30):
                Car_speed_turn = np.interp(self.angle, [30,45], [80,100])
                self.speed = Car_speed_turn
            elif(self.angle < (-30)):
                Car_speed_turn = np.interp(self.angle, [-45,-30], [100,80])
                self.speed = Car_speed_turn



    def Obey_LeftTurn(self, mode):
        """
        Smooth and gradual turning instead of an abrupt left turn.
        Time-based execution ensures the car doesn’t turn too quickly or for too long.
        Prevents multiple detections by tracking the previous mode.
        """
        self.speed = 50.0
        # Car starts tracking left turn...
        if ( (self.prev_Mode_LT == "Detection") and (mode == "Tracking") ):
            self.prev_Mode_LT = "Tracking"
            self.Detected_LeftTurn = True

        elif ( (self.prev_Mode_LT == "Tracking") and (mode == "Detection") ):
            self.Detected_LeftTurn = False
            self.Activate_LeftTurn = True
            # Move left by 7 degree every 20th iteration after a few waiting a bit(100-iteration delay) 
            if ( ((self.Left_turn_iterations % 20) == 0) and (self.Left_turn_iterations > 100) ):
                self.Frozen_Angle = self.Frozen_Angle -7

            # After a time period has passed [Deactivate Left Turn + Reset Left Turn Variable]
            if (self.Left_turn_iterations == 250):
                self.prev_Mode_LT = "Detection"
                self.Activate_LeftTurn = False
                self.Left_turn_iterations = 0
            
            self.Left_turn_iterations = self.Left_turn_iterations + 1

        # Angle of car adjustment after detect left-turn sign
        if (self.Activate_LeftTurn or self.Detected_LeftTurn):
            # Follow previously saved route
            self.angle = self.Frozen_Angle



    def drive(self, Current_State):
        [dist, curv, img, mode, tracked_class] = Current_State

        if ((dist != 10000) and (curv != 10000)):
            self.follow_lane(img.shape[1]/4, dist, curv, mode, tracked_class)
        else:
            self.speed = 0.0 # stop the car

        if (tracked_class == "left_turn"):
            self.Obey_LeftTurn(mode)

        # Interpolating the angle and speed from real world to the motor world
        self.angle = np.interp(self.angle, [-45,45], [0.5,-0.5])
        # self.motor_speed = np.interp(self.speed, [30,90], [1,2])

        if self.speed == 0.0:  # Car should stop
            self.motor_speed = 0.0
        else:
            self.motor_speed = np.interp(self.speed, [30, 90], [1, 2])

        print("Car Speed:" , self.speed)
        print("")
        
        
class Car():
    def __init__(self):
        self.Control = Control()

    def display_state(self, frame_disp, angle_of_car, current_speed):

        # Translate ROS car control range to Real world angle and speed:
        angle_of_car = np.interp(angle_of_car, [-0.5,0.5], [45,-45])
        
    
        if (current_speed != 0.0):
            current_speed = np.interp(current_speed, [1,2], [30,90])

        ###################################################  Displaying CONTROL STATE ####################################

        if (angle_of_car <-10):
            direction_string="[ Left ]"
            color_direction=(120,0,255)
        elif (angle_of_car >10):
            direction_string="[ Right ]"
            color_direction=(120,0,255)
        else:
            direction_string="[ Straight ]"
            color_direction=(0,255,0)

        if(current_speed>0):
            direction_string = "Moving --> "+ direction_string
        else:
            color_direction=(0,0,255)


        cv2.putText(frame_disp,str(direction_string),(20,40),cv2.FONT_HERSHEY_DUPLEX,0.4,color_direction,1)

        angle_speed_str = "[ Angle ,Speed ] = [ " + str(int(angle_of_car)) + "deg ," + str(int(current_speed)) + "mph ]"
        cv2.putText(frame_disp,str(angle_speed_str),(20,20),cv2.FONT_HERSHEY_DUPLEX,0.4,(0,0,255),1)    
        

    def drive_car(self, frame):
        img = frame[0:640, 238:1042]
        # resizeing to minimize computational time
        img = cv2.resize(img, (320, 240))
        img_original = img.copy()

        distance, curvature = detect_lanes(img)

        mode, tracked_class = detect_signs(img_original, img)

        Current_State = [distance, curvature, img, mode, tracked_class]
        
        self.Control.drive(Current_State)

        self.display_state(img, self.Control.angle, self.Control.motor_speed)

        return self.Control.angle, self.Control.motor_speed, img
        