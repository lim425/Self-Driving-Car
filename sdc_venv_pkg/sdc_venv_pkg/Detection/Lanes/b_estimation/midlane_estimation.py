import cv2
import math
import numpy as np

def Distance_(a,b):
    return math.sqrt( ( (a[1]-b[1])**2 ) + ( (a[0]-b[0])**2 ) )

def ApproxDistBWCntrs(cnt,cnt_cmp): # calculates the minimum distance between two contours
    # compute the center of the contour
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"]) #m00 is area, 
    cY = int(M["m01"] / M["m00"])
    # compute the center of the another contour
    M_cmp = cv2.moments(cnt_cmp)
    cX_cmp = int(M_cmp["m10"] / M_cmp["m00"])
    cY_cmp = int(M_cmp["m01"] / M_cmp["m00"])
    minDist=Distance_((cX,cY),(cX_cmp,cY_cmp))
    Centroid_a=(cX,cY)
    Centroid_b=(cX_cmp,cY_cmp)
    return minDist,Centroid_a,Centroid_b

def RetLargestContour(gray):
    LargestContour_Found = False
    thresh = np.zeros(gray.shape,dtype=gray.dtype)
    _, bin_img = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
    #Find the two Contours for which you want to find the min distance between them.
    cnts = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    Max_Cntr_area = 0
    Max_Cntr_idx= -1
    for index, cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        if area > Max_Cntr_area:
            Max_Cntr_area = area
            Max_Cntr_idx = index
            LargestContour_Found = True
    if (Max_Cntr_idx!=-1): # it means at least one contour was found
        thresh = cv2.drawContours(thresh, cnts, Max_Cntr_idx, (255,255,255), -1) # [ contour = less then minarea contour, contourIDx, Colour , Thickness ]
    return thresh, LargestContour_Found


def estimate_midlane(midlane_patches, max_dist):

    """Estimate the mid-lane trajectory based on the detected midlane (patches) mask

    Args:
        midlane_patches (numpy_1d_array): mid_lane_edge mask extracted from the segment_midlane()
        max_dist (int): max distance for a patch to be considered part of the midlane else it is noise

    Returns:
        numpy_1d_array: estimated midlane trajectory (mask)
    """

    # >>>>>>>>>> 1. Keep a midlane draw for displaying shortest connectivity later on >>>>>>>>>>
    midlane_connectivity_bgr = cv2.cvtColor(midlane_patches, cv2.COLOR_GRAY2RGB)

    # >>>>>>>>>> 2. Extract the contour for which we want to find the min distance between them >>>>>>>>>>
    cnts = cv2.findContours(midlane_patches, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    # >>>>>>>>>> 3. Keep only those contours that are not lines >>>>>>>>>>
    min_area = 1
    legit_cnts = []
    for _, cnt in enumerate(cnts):
        cnt_area = cv2.contourArea(cnt)
        if (cnt_area > min_area):
            legit_cnts.append(cnt)
    cnts = legit_cnts

   
    # Cycle through each point in the Two contours & find the distance between them.
    # Take the minimum Distance by comparing all other distances & Mark that Points.
    CntIdx_BstMatch = [] # [BstMatchwithCnt0, BstMatchwithCnt1, ......]

    # >>>>>>>>>> 4. Connect each contous with its closest >>>>>>>>>>
    for index, cnt in enumerate(cnts):
        prevmin_dist = 100000
        Bstindex_compare = 0 # Placeholder for the closest contour index
        BestCentroid_a = 0 # Placeholder for the centroid of the current contour
        BestCentroid_b = 0  # Placeholder for the centroid of the closest contour

        # The algorithm avoids redundant comparisons by only comparing each contour with those that come after it in the list
        # This structure allows the code to compute the closest contour for each contour without repeating or mirroring comparisons.
        for index_compare in range(len(cnts) - index):         
            index_compare = index_compare + index
            cnt_compare = cnts[index_compare]

            if (index != index_compare): # This condition ensures that cnt is not compared with itself
                min_dist, cent_a, cent_b = ApproxDistBWCntrs(cnt, cnt_compare) # calculate distance between contours
                if (min_dist < prevmin_dist):
                    if (len(CntIdx_BstMatch) == 0): # if len(CntIdx_BstMatch) == 0, meaning we havent found any closest contour
                        prevmin_dist = min_dist # Updates prevmin_dist with min_dist
                        Bstindex_compare = index_compare # Sets Bstindex_compare to index_compare (the index of the closest contour)
                        BestCentroid_a = cent_a # Updates BestCentroid_a to the centroids of cnt
                        BestCentroid_b = cent_b # Updates BestCentroid_b to the centroids of cnt_compare

                    else: # Check for Duplicate Contour Connections
                        already_presents = False
                        for i in range(len(CntIdx_BstMatch)):
                            if ((index_compare == 1) and (index == CntIdx_BstMatch[1])): # Check the particular combination is present before or not
                                already_presents = True
                            if not already_presents: # Update Closest Contour if No Duplicate is Found
                                prevmin_dist = min_dist
                                Bstindex_compare = index_compare
                                BestCentroid_a = cent_a
                                BestCentroid_b = cent_b

        # check if that closest contour is beyond min distance or not,
        # if greater than the min_dist that declare at the 2nd argument of function, then discard that contour
        if ((prevmin_dist != 100000) and (prevmin_dist > max_dist)):
            break

        if (type(BestCentroid_a) != int):
            CntIdx_BstMatch.append(Bstindex_compare)
            cv2.line(midlane_connectivity_bgr, (int(BestCentroid_a[0]), int(BestCentroid_a[1])), (int(BestCentroid_b[0]), int(BestCentroid_b[1])), (0, 255, 0), 2)


    midlane_connectivity = cv2.cvtColor(midlane_connectivity_bgr, cv2.COLOR_BGR2GRAY)

    # >>>>>>>>>> 5. Get estimated midlane by returning the largest contour >>>>>>>>>>
    estimated_midlane, largest_found = RetLargestContour(midlane_connectivity)

    # >>>>>>>>>> 6. Return estimated midlane if found, otherwise send original >>>>>>>>>>
    if largest_found:
        return estimated_midlane
    else:
        return midlane_patches


