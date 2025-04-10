import cv2
import numpy as np
from sdc_venv_pkg.Detection.Lanes.Morph_op import BwareaOpen, Ret_LowestEdgePoints, RetLargestContour_OuterLane
from sdc_venv_pkg.config import config


# White region range
Hue_Low = 0
Lit_Low = 225
Sat_Low = 0

Hue_Low_Y = 30
Hue_High_Y = 33
Lit_Low_Y = 160
Sat_Low_Y = 0


def OnHueLowChange(val):
    global Hue_Low
    Hue_Low = val

def OnLitLowChange(val):
    global Lit_Low
    Lit_Low = val

def OnSatLowChange(val):
    global Sat_Low
    Sat_Low = val

def OnHueLowChange_Y(val):
    global Hue_Low_Y
    Hue_Low_Y = val

def OnHueHighChange_Y(val):
    global Hue_High_Y
    Hue_High_Y = val

def OnLitLowChange_Y(val):
    global Lit_Low_Y
    Lit_Low_Y = val

def OnSatLowChange_Y(val):
    global Sat_Low_Y
    Sat_Low_Y = val



cv2.namedWindow("white_regions")
cv2.namedWindow("yellow_regions")

cv2.createTrackbar("Hue_L","white_regions",Hue_Low,255,OnHueLowChange)
cv2.createTrackbar("Lit_L","white_regions",Lit_Low,255,OnLitLowChange)
cv2.createTrackbar("Sat_L","white_regions",Sat_Low,255,OnSatLowChange)

cv2.createTrackbar("Hue_L","yellow_regions",Hue_Low_Y,255,OnHueLowChange_Y)
cv2.createTrackbar("Hue_H","yellow_regions",Hue_High_Y,255,OnHueHighChange_Y)
cv2.createTrackbar("Lit_L","yellow_regions",Lit_Low_Y,255,OnLitLowChange_Y)
cv2.createTrackbar("Sat_L","yellow_regions",Sat_Low_Y,255,OnSatLowChange_Y)



def segment_midlane(frame, white_regions, min_area):
    # >>>>>>>>>>>>>>>>> 4a. Keeping only Midlane ROI of frame >>>>>>>>>>>>>>>>>
    frame_roi = cv2.bitwise_and(frame, frame, mask=white_regions) # Extracting only RGB from a specific region
    # >>>>>>>>>>>>>>>>> 4b.  Converting frame to grayscale >>>>>>>>>>>>>>>>>
    frame_roi_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY) # Converting to grayscale
    # >>>>>>>>>>>>>>>>> 4c. Keeping only objects larger than min_area >>>>>>>>>>>>>>>>>
    mid_lane_mask = BwareaOpen(frame_roi_gray, min_area) # Getting mask of only objects larger then minArea
    frame_roi_gray = cv2.bitwise_and(frame_roi_gray, mid_lane_mask) # Getting the gray of that mask
    # >>>>>>>>>>>>>>>>> 4d. Extracting edges of those larger objects >>>>>>>>>>>>>>>>>
    frame_roi_smoothed = cv2.GaussianBlur(frame_roi_gray, (11,11), 1) # Smoothing out the edges for edge extraction later
    mid_lane_edge = cv2.Canny(frame_roi_smoothed, 50, 150, None, 3) # Extracting the Edge of Canny

    return mid_lane_mask, mid_lane_edge


def segment_outerlane(frame, yellow_regions, min_area):
    outer_points_list = []
    # Initialize edges to solve UnboundLocalError in segment_outerlane()
    # This way, edges always has a value before it is returned, even if largest_found is False
    edges = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)  # Initialize edges

    # >>>>>>>>>>>>>>>>> 5a . Extract OuterLanes Mask And Edge >>>>>>>>>>>>>>>>>
    frame_roi = cv2.bitwise_and(frame, frame, mask=yellow_regions) #Extracting only RGB from a specific region
    frame_roi_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY) # Converting to grayscale
    mask_of_larger_objects = BwareaOpen(frame_roi_gray, min_area) # Getting mask of only objects larger then minArea
    frame_roi_gray = cv2.bitwise_and(frame_roi_gray, mask_of_larger_objects) # Getting the gray of that mask
    frame_roi_smoothed = cv2.GaussianBlur(frame_roi_gray, (11,11), 1) # Smoothing out the edges for edge extraction later
    edges_of_larger_objects = cv2.Canny(frame_roi_smoothed, 50, 150, None, 3) # Extracting the Edge of Canny
    
    # >>>>>>>>>>>>>>>>> 5b . Kept Larger OuterLane >>>>>>>>>>>>>>>>>
    # choose only closest lane, eliminate another one, after call function below we have 1 lane(yellow)
    mask_largest, largest_found = RetLargestContour_OuterLane(mask_of_larger_objects, min_area)
    if largest_found:
        # >>>>>>>>>>>>>>>>> 5c. Kept Larger OuterLane [Edge] >>>>>>>>>>>>>>>>>
        edge_largest = cv2.bitwise_and(edges_of_larger_objects, mask_largest) # at this point we only have 1 lane, 2 edge
        # >>>>>>>>>>>>>>>>> # 5d. Returned Lowest Edge line & points,  >>>>>>>>>>>>>>>>>
        lanes_sides_sep, outer_points_list = Ret_LowestEdgePoints(edge_largest) # at this point we only have 1 edge of that lane
        edges = edge_largest
    else:
        lanes_sides_sep = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
    return edges, lanes_sides_sep, outer_points_list


def clr_segment(hls, lower_range, upper_range):
    # >>>>>>>>>>>>>>>>> 2. Performing Color Segmentation on Given Range >>>>>>>>>>>>>>>>>
    mask_in_range = cv2.inRange(hls, lower_range, upper_range)

    # >>>>>>>>>>>>>>>>> 3. Dilating Segmented ROI's >>>>>>>>>>>>>>>>>
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask_dilated = cv2.morphologyEx(mask_in_range, cv2.MORPH_DILATE, kernel)

    return mask_dilated


def segment_lanes(frame, min_area):

    """ Segment Lane-Lines (both outer and middle) from the road lane

    Args:
        frame (numpy nd array): Prius front-cam view
        minArea (int): minimum area of an object required to be considered as a valid object

    Returns:
        numpy 2d array: Mask of white mid-lane
        numpy 2d array: Edges  of white  mid-lane
        numpy 2d array: Edges of yellow outer-lane [2 lines: inner and outer line edge]
        numpy 2d array: Edge of yellow outer-lane [1 lines: inner line edge]
                  List: Two points taken one each from outer-Lane edge seperated
    """

    # >>>>>>>>>>>>>>>>> 1. Converting frame to HLS ColorSpace >>>>>>>>>>>>>>>>>
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    # Segmenting white & yellow region
    white_regions = clr_segment(hls, np.array([Hue_Low, Lit_Low, Sat_Low]), np.array([255, 255, 255]))
    yellow_regions = clr_segment(hls, np.array([Hue_Low_Y, Lit_Low_Y, Sat_Low_Y]), np.array([Hue_High_Y, 255, 255]))

    cv2.imshow("white_regions", white_regions)
    cv2.imshow("yellow_regions", yellow_regions)
    cv2.waitKey(1)

    # >>>>>>>>>>>>>>>>> # 6a. segmenting midlane from white regions >>>>>>>>>>>>>>>>>
    mid_lane_mask, mid_lane_edge = segment_midlane(frame, white_regions, min_area)

    # >>>>>>>>>>>>>>>>> # 6b. segmenting outerlane from yellow regions >>>>>>>>>>>>>>>>>
    outer_lane_edge, outerlane_side_sep, outerlane_points = segment_outerlane(frame, yellow_regions, min_area)

    return mid_lane_mask, mid_lane_edge, outer_lane_edge, outerlane_side_sep, outerlane_points
