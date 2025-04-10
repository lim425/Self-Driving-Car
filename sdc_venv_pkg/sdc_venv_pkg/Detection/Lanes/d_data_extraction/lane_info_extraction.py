import cv2
import numpy as np
from sdc_venv_pkg.Detection.Lanes.utilities import Cord_Sort, findlaneCurvature


def LanePoints(midlane, outerlane, offset):
    mid_cnts = cv2.findContours(midlane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    outer_cnts = cv2.findContours(outerlane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    if mid_cnts and outer_cnts:
        mid_cnts_row_sorted = Cord_Sort(mid_cnts, "rows") # Sorted output(2D arrays), example: [[2, 1], [1, 2], [1, 3], [2, 3]].
        outer_cnts_row_sorted = Cord_Sort(outer_cnts, "rows")

        m_rows = mid_cnts_row_sorted.shape[0]
        o_rows = outer_cnts_row_sorted.shape[0]

        # m_rows-1 refer to last index in mid_cnts_row_sorted, representing the bottommost point of the mid-lane contour.
        # " : " Retrieves the [x, y] coordinates of the bottommost mid-lane point.
        m_rows_btm_pt = mid_cnts_row_sorted[m_rows-1, :]
        o_rows_btm_pt = outer_cnts_row_sorted[o_rows-1, :]
        m_rows_top_pt = mid_cnts_row_sorted[0, :]
        o_rows_top_pt = outer_cnts_row_sorted[0, :]

        traj_btm_pt = ( int((m_rows_btm_pt[0] + o_rows_btm_pt[0]) / 2) + offset,  int((m_rows_btm_pt[1] + o_rows_btm_pt[1]) / 2)) # The central point at the bottom of the lane.
        traj_top_pt = ( int((m_rows_top_pt[0] + o_rows_top_pt[0]) / 2) + offset,  int((m_rows_top_pt[1] + o_rows_top_pt[1]) / 2)) # The central point at the top of the lane.
        

        return traj_btm_pt, traj_top_pt
    
    else:
        return (0,0), (0,0)


def EstimateNonMidMask(MidEdgeROI):

    # a.  Create an Empty Mask
    Mid_Hull_Mask = np.zeros((MidEdgeROI.shape[0], MidEdgeROI.shape[1], 1), dtype=np.uint8)
    # b. Extracts contours from the binary image MidEdgeROI
    contours = cv2.findContours(MidEdgeROI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    if contours:
        hull_list = [] # The hull_list is used to store one or more convex hulls. This is helpful if there are multiple contours 
        # c. Combine Contours
        contours = np.concatenate(contours)
        # d. Computes the convex hull of the combined contours
        # If the input contours have jagged or irregular edges, the convex hull smooths those edges to form a clean convex boundary around the object
        hull = cv2.convexHull(contours) # hull refers to the outer boundary or shape that encloses a set of points or objects
        hull_list.append(hull)
        # e. Fills the convex hull area on the Mid_Hull_Mask
        Mid_Hull_Mask = cv2.drawContours(Mid_Hull_Mask, hull_list, 0, 255, -1)
    
    # f. Inverts the mask, turning the mid-lane region (white) into black and the outside region (black) into white.
    Non_Mid_Mask = cv2.bitwise_not(Mid_Hull_Mask) # All 255 (white) pixels become 0 (black), All 0 (black) pixels become 255 (white)
    return Non_Mid_Mask


def FetchInfoAndDisplay(Mid_lane_edge, Mid_lane, Outer_lane, frame, Offset_correction):

    # >>>>>>>>>>>>>>>> step1: Using both outer and middle information to create probable path >>>>>>>>>>>>>>>>
    Traj_lowP, Traj_upP = LanePoints(Mid_lane, Outer_lane, Offset_correction)

    # >>>>>>>>>>>>>>>> step2: Compute distance and curvature from trajectory points >>>>>>>>>>>>>>>>
    PerpDist_LaneCentralStart_CarNose = -1000
    if(Traj_lowP != (0,0)):
        PerpDist_LaneCentralStart_CarNose = Traj_lowP[0] - int(Mid_lane.shape[1] / 2)
    curvature = findlaneCurvature(Traj_lowP[0], Traj_lowP[1], Traj_upP[0], Traj_upP[1])

    # >>>>>>>>>>>>>>>> step3: Keep only those edge that are part of midlane >>>>>>>>>>>>>>>>
    Mid_lane_edge = cv2.bitwise_and(Mid_lane_edge, Mid_lane) # Any extraneous edges outside the midlane (from noise, other lanes, or unrelated features) are filtered out

    # >>>>>>>>>>>>>>>> step4: Combine mid and outerlane to get lane combined and extract its contours >>>>>>>>>>>>>>>>
    Lanes_combined = cv2.bitwise_or(Outer_lane, Mid_lane)
    ProjectedLane = np.zeros(Lanes_combined.shape, Lanes_combined.dtype)
    cnts = cv2.findContours(Lanes_combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    # >>>>>>>>>>>>>>>> step5: Fill ProjectedLane with fillConvexPoly >>>>>>>>>>>>>>>>
    if cnts:
        cnts = np.concatenate(cnts) # Combines all the contours (cnts) into a single array
        cnts = np.array(cnts) # Ensures that cnts is a properly formatted NumPy array
        cv2.fillConvexPoly(ProjectedLane, cnts, 255) # Fills the convex polygon defined by the points in cnts onto the ProjectedLane image
    
    # >>>>>>>>>>>>>>>> step6: Remove midlane region from projectedlane by extracting the midless mask >>>>>>>>>>>>>>>>
    # Seperate midlane after combining it earlier because we want to Focus on Outer Lane for Path Planning:
    # After generating the unified lane representation, the mid-lane region is removed because:
    # The outer lane boundaries are often more critical for defining the drivable path or the robot's trajectory.
    # Keeping the mid-lane in ProjectedLane could confuse or bias path planning algorithms that rely on this mask.
    Mid_less_Mask = EstimateNonMidMask(Mid_lane_edge)
    ProjectedLane = cv2.bitwise_and(Mid_less_Mask, ProjectedLane)

    # >>>>>>>>>>>>>>>> step7: Draw projected line >>>>>>>>>>>>>>>>
    Lane_drawn_frame = frame
    Lane_drawn_frame[ProjectedLane==255] = Lane_drawn_frame[ProjectedLane==255] + (0,100,0) # Moveable lane is set to Green
    Lane_drawn_frame[Outer_lane==255] = Lane_drawn_frame[Outer_lane==255] + (0,0,100)# Outer Lane is set to Red
    Lane_drawn_frame[Mid_lane==255] = Lane_drawn_frame[Mid_lane==255] + (100,0,0)# Mid Lane is set to Blue
    Out_image = Lane_drawn_frame
    
    # >>>>>>>>>>>>>>>> step8: Draw car direction and lane direction and distance between car and lane path >>>>>>>>>>>>>>>>
    cv2.line(Out_image,(int(Out_image.shape[1]/2),Out_image.shape[0]),(int(Out_image.shape[1]/2),Out_image.shape[0]-int (Out_image.shape[0]/5)),(0,0,255),2)
    cv2.line(Out_image,Traj_lowP,Traj_upP,(255,0,0),2)
    if(Traj_lowP!=(0,0)):
        cv2.line(Out_image,Traj_lowP,(int(Out_image.shape[1]/2),Traj_lowP[1]),(255,255,0),2)# distance of car center with lane path


    # >>>>>>>>>>>>>>>> step9: Draw extracted distance and curvature value >>>>>>>>>>>>>>>>
    curvature_str="Curvature = " + f"{curvature:.2f}" 
    PerpDist_ImgCen_CarNose_str="Distance = " + str(PerpDist_LaneCentralStart_CarNose)
    textSize_ratio = 0.5
    cv2.putText(Out_image,curvature_str,(10,30),cv2.FONT_HERSHEY_DUPLEX,textSize_ratio,(0,255,255),1)
    cv2.putText(Out_image,PerpDist_ImgCen_CarNose_str,(10,50),cv2.FONT_HERSHEY_DUPLEX,textSize_ratio,(0,255,255),1)

    return PerpDist_LaneCentralStart_CarNose, curvature
