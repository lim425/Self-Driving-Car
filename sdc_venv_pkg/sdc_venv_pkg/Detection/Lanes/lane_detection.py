import cv2

# venv path lib
from sdc_venv_pkg.config import config
from sdc_venv_pkg.Detection.Lanes.a_segmentation.colour_segmentation import segment_lanes
from sdc_venv_pkg.Detection.Lanes.b_estimation.midlane_estimation import estimate_midlane
from sdc_venv_pkg.Detection.Lanes.c_cleaning.cleaning import GetYellowLaneEdge, ExtendShortLane
from sdc_venv_pkg.Detection.Lanes.d_data_extraction.lane_info_extraction import FetchInfoAndDisplay



def detect_lanes(img):
    # cropping the ROI (eg. keeping only the below horizon)
    img_cropped = img[config.CropHeight_resized:,:]
    
    # [Lane Detection] Stage 1: Colour segmentation
    mid_lane_mask, mid_lane_edge, outer_lane_edge, outerlane_side_sep, outerlane_points = segment_lanes(img_cropped, config.minArea_resized)

    # [Lane Detection] Stage 2: Lane Estimation
    estimated_midlane = estimate_midlane(mid_lane_edge, config.MaxDist_resized)

    # [Lane Detection] Stage 3: Cleaning (Step1):
    OuterLane_oneside, Outer_cnts_oneside, Mid_cnts, Offset_correction = GetYellowLaneEdge(outerlane_side_sep, estimated_midlane, outerlane_points)
    # [Lane Detection] Stage 3: Cleaning (Step2):
    extended_midlane, extended_outerlane = ExtendShortLane(estimated_midlane, Mid_cnts, Outer_cnts_oneside, OuterLane_oneside.copy())

    # [Lane Detection] Stage 4: Data extraction:
    Distance, Curvature = FetchInfoAndDisplay(mid_lane_edge, extended_midlane, extended_outerlane, img_cropped, Offset_correction)


    # # Debugging
    cv2.imshow("mid_lane_mask", mid_lane_mask)
    cv2.imshow("mid_lane_edge", mid_lane_edge)
    cv2.imshow("outer_lane_edge", outer_lane_edge)
    cv2.imshow("outerlane_side_sep", outerlane_side_sep)
    cv2.imshow("estimated_midlane", estimated_midlane)

    cv2.imshow("OuterLane_oneside", OuterLane_oneside)
    cv2.imshow("extended_midlane", extended_midlane)
    cv2.imshow("extended_outerlane", extended_outerlane)

    cv2.imshow("img_cropped", img_cropped)

    cv2.waitKey(1)

    return Distance, Curvature
    



