import cv2
import numpy as np
from sdc_venv_pkg.Detection.Lanes.utilities import Distance_, Cord_Sort
from sdc_venv_pkg.config import config


def IsPathCrossingMid(Midlane,Mid_cnts,Outer_cnts):

    #  To check if the car's path, based on its current trajectory and reference to the midline, intersects with the midline. 
    # It also determines if the car's path is to the left or right of the reference point on the midline.

    # contour.shape will often be something like (n, 1, 2), where:
    # n is the number of points in the contour,
    # 1 is just an extra dimension used by OpenCV for storage,
    # 2 represents the x and y coordinates of each point.
    
	is_Ref_to_path_Left = 0
	Ref_To_Path_Image = np.zeros_like(Midlane)
	Midlane_copy = Midlane.copy()

	Mid_cnts_Rowsorted = Cord_Sort(Mid_cnts,"rows") # Sorted output(2D arrays), example: [[2, 1], [1, 2], [1, 3], [2, 3]].
	Outer_cnts_Rowsorted = Cord_Sort(Outer_cnts,"rows")

	if not Mid_cnts:
		print("[Warning!!!] NO Midlane detected")

	Mid_Rows = Mid_cnts_Rowsorted.shape[0] # shape[0] gives the total number of points in the sorted midline, if [1] gives the [x, y]
	Outer_Rows = Outer_cnts_Rowsorted.shape[0] # shape[0] gives the total number of points in the sorted outerline, if [1] gives the [x, y]
    
    # Mid_lowP = Mid_cnts_Rowsorted[Mid_Rows - 1, :] accesses the last row of the contour, i.e., the point with the largest y-coordinate, which represents the "lowest" point (bottommost in the image).
    # The notation [:] indicates that we want all columns (x and y) for that row.
    # The structure of Mid_lowP & Outer_lowP is a 1D array with two values: the x and y coordinates of the lowest point.
	Mid_lowP = Mid_cnts_Rowsorted[Mid_Rows-1,:] # selects the last (lowest) point from the sorted midline contour.
	Outer_lowP = Outer_cnts_Rowsorted[Outer_Rows-1,:] # selects the last (lowest) point from the outer lane contour.

	Traj_lowP = ( int( (Mid_lowP[0] + Outer_lowP[0]  ) / 2 ) , int( (Mid_lowP[1]  + Outer_lowP[1] ) / 2 ) )
	
    # Line 1 is drawing the projected path of the car from the midpoint of the current lane to the bottom-center of the image, helping to see if the car's trajectory crosses into the lane or moves off course.
    # Line 2 is drawing the vertical alignment of the midline, showing the lane's path from the low point to the bottom of the image. This helps visualize how well the car is aligned with the center of the lane.
	cv2.line(Ref_To_Path_Image,Traj_lowP,(int(Ref_To_Path_Image.shape[1]/2),Ref_To_Path_Image.shape[0]),(255,255,0),2)# distance of car center with lane path
	cv2.line(Midlane_copy,tuple(Mid_lowP),(Mid_lowP[0],Midlane_copy.shape[0]-1),(255,255,0),2)# distance of car center with lane path
    
    # If the result is positive, it means that Traj_lowP is to the left of the center, so is_Ref_to_path_Left will be True. If negative, is_Ref_to_path_Left will be False
	is_Ref_to_path_Left = ( (int(Ref_To_Path_Image.shape[1]/2) - Traj_lowP[0]) > 0 )
    
    # Checks whether the car's path (represented by Ref_To_Path_Image) intersects with the midline (Midlane_copy)
	if( np.any( (cv2.bitwise_and(Ref_To_Path_Image,Midlane_copy) > 0) ) ):
        # True: Midlane and CarPath Intersets (MidCrossing)
        # is_Ref_to_path_Left: indicating whether the car's trajectory is to the left or right of the reference point on the midline.
		return True,is_Ref_to_path_Left 
	else:
         return False,is_Ref_to_path_Left

def GetYellowLaneEdge(OuterLanes, MidLane, OuterLane_Points):

    """Fetching closest outer lane (side) to mid lane 

	Args:
		OuterLanes (numpy_1d_array): detected outerlane
		MidLane (numpy_1d_array): estimated midlane trajectory
		OuterLane_Points (list): points one from each side of detected outerlane

	Returns:
		numpy_1d_array: outerlane (side) closest to midlane
		list[List[tuple]]: refined contours of outerlane
		list[List[tuple]]: refined contours of midlane
		int: Offset to compensate for **removal of either midlane or outerlane 
			                 **(incase of false-positives)
	"""	


    Offset_correction = 0
    Outer_Lanes_ret = np.zeros_like(OuterLanes)

    # 1. Extracting MidLane and OuterLanes contours
    Mid_cnts = cv2.findContours(MidLane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    Outer_cnts = cv2.findContours(OuterLanes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # 2. Checking if OuterLanes was present initially or not
    if not Outer_cnts:
        NoOuterLane_before = True
    else:
        NoOuterLane_before = False

    # 3. Setting the first contour of MidLane as reference
    Ref = (0, 0) # This sets a default reference point in case no contours are found in MidLane
    if (Mid_cnts): # if mid contours are present, use the first contour point as Ref to find nearest yellow lane contour
        Ref = tuple(Mid_cnts[0][0][0])  # Mid_cnts[0]: Accesses the first contour in Mid_cnts.
                                        # Mid_cnts[0][0]: Gets the first point in this contour, which is represented as a coordinate in the format [[x, y]].
                                        # Mid_cnts[0][0][0]: Extracts just the x and y values (e.g., [x, y]), and tuple() converts this list into a tuple (x, y).


    if Mid_cnts:
        # 4a. >>>>>>>>>> Condition 1: if both MidLane and outlane is detected >>>>>>>>>>

        # (i) len(OuterLane_Points) == 2
        # i, step1: Fetching side of OuterLanes nearest to MidLane
        if (len(OuterLane_Points) == 2):
            point_a = OuterLane_Points[0]  # lane side 1
            point_b = OuterLane_Points[1]  # lane side 2

            closest_index = 0
            if(Distance_(point_a, Ref) <= Distance_(point_b, Ref)):
                closest_index = 0
            elif (len(Outer_cnts) > 1):
                closest_index = 1
            
            Outer_Lanes_ret = cv2.drawContours(Outer_Lanes_ret, Outer_cnts, closest_index, 255, 1)
            Outer_cnts_ret = [Outer_cnts[closest_index]] # Outer_cnts_ret contains only the closest contour, making it easier to pass this specific contour to other function

        # i step2: Check if correct OuterLanes was detected
            IsPathCrossing , IsCrossingLeft = IsPathCrossingMid(MidLane, Mid_cnts, Outer_cnts_ret)
            if (IsPathCrossing): 
                OuterLanes = np.zeros_like(OuterLanes) # if crossing mid meaning incorrect outerlane, remove the outerlane
            else:
                return Outer_Lanes_ret, Outer_cnts_ret, Mid_cnts, 0
            
        # (ii) len(OuterLane_Points != 2)
        elif (np.any(OuterLanes>0)):
            IsPathCrossing, IsCrossingLeft = IsPathCrossingMid(MidLane, Mid_cnts, Outer_cnts)

            if(IsPathCrossing):
                OuterLanes = np.zeros_like(OuterLanes) # if crossing mid meaning incorrect outerlane, remove the outerlane
            else:
                return OuterLanes, Outer_cnts, Mid_cnts, 0
        
        

        # 4b. >>>>>>>>>>>>>> Condition 2 : if MidLane is present but no Outlane detected (Or Outlane got zerod because of crossings Midlane) >>>>>>>>>>>>>> 
        # Action: Create Outlane on Side that represent the larger Lane as seen by camera
        if (not np.any(OuterLanes > 0)):
            # Fetching the column of the lowest point of the midlane
            # Mid_lowP = Mid_cnts_Rowsorted[Mid_Rows - 1, :] accesses the last row of the contour, i.e., the point with the largest y-coordinate, which represents the "lowest" point (bottommost in the image).
            # The notation [:] indicates that we want all columns (x and y) for that row.
            # The structure of Mid_lowP & Outer_lowP is a 1D array with two values: the x and y coordinates of the lowest point.
            Mid_cnts_Rowsorted = Cord_Sort(Mid_cnts,"rows") # Sorted output(2D arrays), example: [[2, 1], [1, 2], [1, 3], [2, 3]]
            Mid_Rows = Mid_cnts_Rowsorted.shape[0] # gives the total number of points in the sorted midline 
            Mid_lowP = Mid_cnts_Rowsorted[Mid_Rows-1,:] # selects the last (lowest) point from the sorted midline contour.
            Mid_highP = Mid_cnts_Rowsorted[0,:] # selects the first (highest) point from the sorted midline contour.
            Mid_low_Col = Mid_lowP[0] # Extract the x-Coordinate of the Lowest Point

            # Addressing which side to draw the outerlane considering it was present before or not
            DrawRight = False
            if NoOuterLane_before:
                if(Mid_low_Col < int(MidLane.shape[1]/2)): # MidLane on left side of Col/2 of image --> Bigger side is right side, draw there
                    DrawRight = True
            # If Outerlane was present before and got EKIA: >>> DrawRight because it was Crossing LEFt
            else:
                if IsCrossingLeft: # trajectory from reflane to lane path is crossing MidLane while moving left --> Draw Right
                    DrawRight = True
            
            #Offset Correction wil be set here to correct for the yellow lane not found 
            # IF we are drawing right then  we need to correct car to move right to find that OuterLanes
            # Else Move Left
            # 4. [Midlane But , No OuterLanes!!!] D : Calculate Offset Correction
            if not DrawRight:
                low_Col=0
                high_Col=0
                Offset_correction = -20 # the car should mov eto left
            else:
                low_Col=(int(MidLane.shape[1])-1) # Since arrays start at 0, MidLane.shape[1] - 1 correctly points to the final column.
                high_Col=(int(MidLane.shape[1])-1)
                Offset_correction = 20 # the car should mobe to right
 
            Mid_lowP[1] = MidLane.shape[0]# setting mid_trajectory_lowestPoint_Row to MaxRows of Image

            LanePoint_lower =  (low_Col , int( Mid_lowP[1] ) )
            LanePoint_top   =  (high_Col, int( Mid_highP[1]) )

            # Draw OuterLAnes according to MidLane information
            OuterLanes = cv2.line(OuterLanes,LanePoint_lower,LanePoint_top,255,1)

            # Find OuterLane Contours	
            Outer_cnts = cv2.findContours(OuterLanes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

            return OuterLanes, Outer_cnts, Mid_cnts, Offset_correction
        
    # 4c. >>>>>>>>>>>>>> Condition 3: No Midlane >>>>>>>>>>>>>>
    else:
        return OuterLanes, Outer_cnts, Mid_cnts, Offset_correction

	
# Step 2 cleaning stage: Extending the short lane

def ExtendShortLane(MidLane,Mid_cnts,Outer_cnts,OuterLane):
     
    # >>>>>>>>>>>>>> step 1: Sorting the mid and outer contours of row (Ascending) >>>>>>>>>>>>>>
    if (Mid_cnts and Outer_cnts):
        Mid_cnts_Rowsorted = Cord_Sort(Mid_cnts, "rows")
        Outer_cnts_Rowsorted = Cord_Sort(Outer_cnts, "rows")
        Image_bottom = MidLane.shape[0]
        total_no_of_cnts_midlane = Mid_cnts_Rowsorted.shape[0] # shape[0] gives the total number of points in the sorted outerline, if shape[1] gives the [x, y]
        total_no_of_cnts_outerlane = Outer_cnts_Rowsorted.shape[0] # shape[0] gives the total number of points in the sorted outerline, if shape[1] gives the [x, y]


        # >>>>>>>>>>>>>> step 2: Connect Midlane to the image bottom by drawing a vertical line if it is not connected >>>>>>>>>>>>>>
        
        # Mid_cnts_Rowsorted[total_no_of_cnts_midlane - 1, :] accesses the last row (the lowest point) 
        # The : specifies that we want all elements (both x and y coordinates) of this row.
        # For example, if Mid_cnts_Rowsorted is:
        # [[10, 5],    # A point with x=10, y=5
        # [15, 10],   # A point with x=15, y=10
        # [20, 20]]   # A point with x=20, y=20 (bottommost point)
        # Then total_no_of_cnts_midlane would be 3, and Mid_cnts_Rowsorted[total_no_of_cnts_midlane - 1, :] would access [20, 20], the lowest point in this array.
        
        BottomPoint_Mid = Mid_cnts_Rowsorted[total_no_of_cnts_midlane - 1, :] # will get (x,y)
        if (BottomPoint_Mid[1] < Image_bottom):
            MidLane = cv2.line(MidLane, tuple(BottomPoint_Mid), (BottomPoint_Mid[0], Image_bottom), 255)


        # >>>>>>>>>>>>>> step 3: Connect Outerlane to the image bottom by drawing a vertical line if it is not connected >>>>>>>>>>>>>>

        # 3a) Taking the last 20 points to estimate the slope
        BottomPoint_Outer = Outer_cnts_Rowsorted[total_no_of_cnts_outerlane - 1, :] # will get (x,y)
        if (BottomPoint_Outer[1] < Image_bottom):
            if(total_no_of_cnts_outerlane > 20):
                shift = 20 # shift is the number of points to look back from the last point.
            else:
                shift = 2
            # total_no_of_cnts_outerlane-shift >> This part defines the starting index of the slice, pointing to a position shift rows above the last row
            # total_no_of_cnts_outerlane-1     >> This is the ending index, pointing to the second-to-last row of the array
            # 2 >> This is the step argument, which selects every second point within the specified range. So if shift is set to 20, it will take every second point within the last 20 rows
            # : >>     This is used to indicate that all columns (both x and y coordinates) of the selected rows should be included
            RefLast10Points = Outer_cnts_Rowsorted[(total_no_of_cnts_outerlane-shift):(total_no_of_cnts_outerlane-1):2, :]

            # 3b) Calculating the slope
            if (len(RefLast10Points) > 1): # atleast 2 points is needed to calculate the slope
                Ref_x = RefLast10Points[:,0]#cols - [:, 0] means "take all rows (:) in RefLast10Points, but only the first column (0) from each row."
                Ref_y = RefLast10Points[:,1]#rows - [:, 1] means "take all rows in RefLast10Points, but only the second column (1) from each row."
                Ref_parameters = np.polyfit(Ref_x, Ref_y, 1) # 1 specifies that we want a 1st-degree polynomial (a straight line, or y = mx + c).
                Ref_slope = Ref_parameters[0]
                Ref_yIntercept = Ref_parameters[1]

                # 3c) Extending the outerlane in the direction of its slope
                if (Ref_slope < 0): # If Ref_slope is negative, the line extends upwards toward the left edge.
                    Ref_LineTouchPoint_col = 0 # set the x-coordinate (col) to the left edge of the image (column 0).
                    Ref_LineTouchPoint_row = Ref_yIntercept # Since we’re on the left edge, the y-coordinate (row) is simply the y-intercept, as it represents where the line intersects the y-axis.
                else: # If Ref_slope is positive, the line slopes downward (toward the right in OpenCV coordinates).
                    Ref_LineTouchPoint_col = OuterLane.shape[1] - 1 # Sets the x-coordinate to the far-right edge of the image 
                    Ref_LineTouchPoint_row = Ref_slope * Ref_LineTouchPoint_col +Ref_yIntercept # Calculates the y-coordinate at the right edge using the equation of the line, y=mx+c
                Ref_TouchPoint = (Ref_LineTouchPoint_col), int(Ref_LineTouchPoint_row) #  endpoint of the extended line
                Ref_BottomPoint_tup = tuple(BottomPoint_Outer) #  starting point of the line extension.
                OuterLane = cv2.line(OuterLane, Ref_TouchPoint, Ref_BottomPoint_tup, 255, 2)

                # 3d) if still cannot reach the bottom line, perfome below
                # [If required] : Connect outerlane to bottom bt drawing a vertical line
                if (Ref_LineTouchPoint_row < Image_bottom):
                    Ref_TouchPoint_Ref = (Ref_LineTouchPoint_col, Image_bottom)
                    # draws a line on OuterLane from Ref_TouchPoint (the current endpoint of the line) to Ref_TouchPoint_Ref (directly below on the bottom edge).
                    OuterLane = cv2.line(OuterLane, Ref_TouchPoint, Ref_TouchPoint_Ref, 255)
        
    return MidLane, OuterLane
