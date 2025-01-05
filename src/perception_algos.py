"""
Date: 04.Dec.2024
Author: Harish Natarajan Ravi
Email: harrish.nr@gmail.com
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from m6bk import *

def xy_from_depth(depth, k):
    """
    Computes the x, and y coordinates of every pixel in the image using the depth map and the calibration matrix.

    Arguments:
    depth -- tensor of dimension (H, W), contains a depth value (in meters) for every pixel in the image.
    k -- tensor of dimension (3x3), the intrinsic camera matrix

    Returns:
    x -- tensor of dimension (H, W) containing the x coordinates of every pixel in the camera coordinate frame.
    y -- tensor of dimension (H, W) containing the y coordinates of every pixel in the camera coordinate frame.
    """
    # Get the shape of the depth tensor
    H,W = np.shape(depth)

    # Grab required parameters from the K matrix
    f, c_u, c_v = k[0,0], k[0,2], k[1,2]
    #     print(f,u,v)

    # Generate a grid of coordinates corresponding to the shape of the depth map
    x = np.zeros((H,W))
    y = np.zeros((H,W))

    # Compute x and y coordinates
    for i in range(H):
        for j in range(W):
            z = depth[i][j]
            x[i][j] = (j+1 - c_u) * z / f
            y[i][j] = (i+1 - c_v) * z / f
    
    return x, y

def ransac_plane_fit(xyz_data):
    """
    Computes plane coefficients a,b,c,d of the plane in the form ax+by+cz+d = 0
    using ransac for outlier rejection.

    Arguments:
    xyz_data -- tensor of dimension (3, N), contains all data points from which random sampling will proceed.
    num_itr -- 
    distance_threshold -- Distance threshold from plane for a point to be considered an inlier.

    Returns:
    p -- tensor of dimension (1, 4) containing the plane parameters a,b,c,d
    """
    
    # Set thresholds:
    num_itr = 100  # RANSAC maximum number of iterations
    min_num_inliers = xyz_data.shape[1] / 2  # RANSAC minimum number of inliers
    distance_threshold = 0.01  # Maximum distance from point to plane for point to be considered inlier
    largest_number_of_inliers = 0
    largest_inlier_set_indexes = 0
    
    for i in range(num_itr):

        # Step 1: Choose a minimum of 3 points from xyz_data at random.
        indexes = np.random.choice(xyz_data.shape[1], 3, replace = False)
        pts = xyz_data[:, indexes]

        # Step 2: Compute plane model
        p = compute_plane(pts)

        # Step 3: Find number of inliers
        distance = dist_to_plane(p, xyz_data[0, :].T, xyz_data[1, :].T, xyz_data[2, :].T)
        number_of_inliers = len(distance[distance > distance_threshold])
        
        # Step 4: Check if the current number of inliers is greater than all previous iterations and keep the inlier set with the largest number of points.
        if number_of_inliers > largest_number_of_inliers:
            largest_number_of_inliers = number_of_inliers
            largest_inlier_set_indexes = np.where(distance < distance_threshold)[0]

        # Step 5: Check if stopping criterion is satisfied and break.
        if (number_of_inliers > min_num_inliers):
            break         

        
    # Step 6: Recompute the model parameters using largest inlier set.
    output_plane = compute_plane(xyz_data[:, largest_inlier_set_indexes])
    
    return output_plane 

def estimate_lane_lines(segmentation_output):
    """
    Estimates lines belonging to lane boundaries. Multiple lines could correspond to a single lane.

    Arguments:
    segmentation_output -- tensor of dimension (H,W), containing semantic segmentation neural network output
    minLineLength -- Scalar, the minimum line length
    maxLineGap -- Scalar, dimension (Nx1), containing the z coordinates of the points

    Returns:
    lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2], where [x_1,y_1] and [x_2,y_2] are
    the coordinates of two points on the line in the (u,v) image coordinate frame.
    """
    # Step 1: Create an image with pixels belonging to lane boundary categories from the output of semantic segmentation
    lane_boundary_mask = np.zeros(segmentation_output.shape).astype(np.uint8)
    lane_boundary_mask[segmentation_output==6] = 255
    lane_boundary_mask[segmentation_output==8] = 255
    
    plt.imshow(lane_boundary_mask)

    # Step 2: Perform Edge Detection using cv2.Canny()
    edges = cv2.Canny(lane_boundary_mask, 100, 150)

    # Step 3: Perform Line estimation using cv2.HoughLinesP()
    lines = cv2.HoughLinesP(edges, rho=10, theta=np.pi/180, threshold=200, minLineLength=150, maxLineGap=50)
    lines = lines.reshape((-1, 4))
    
    # Note: Make sure dimensions of returned lines is (N x 4)
    return lines

def merge_lane_lines(
        lines):
    """
    Merges lane lines to output a single line per lane, using the slope and intercept as similarity measures.
    Also, filters horizontal lane lines based on a minimum slope threshold.

    Arguments:
    lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2],
    the coordinates of two points on the line.

    Returns:
    merged_lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2],
    the coordinates of two points on the line.
    """
    # Step 0: Define thresholds
    slope_similarity_threshold = 0.1
    intercept_similarity_threshold = 40
    min_slope_threshold = 0.3
    clusters = []
    current_inds = []
    itr = 0
    
    # Step 1: Get slope and intercept of lines
    slopes, intercepts = get_slope_intecept(lines)
    
    # Step 2: Determine lines with slope less than horizontal slope threshold.
    slopes_horizontal = np.abs(slopes) > min_slope_threshold

    # Step 3: Iterate over all remaining slopes and intercepts and cluster lines that are close to each other using a slope and intercept threshold.
    for slope, intercept in zip(slopes, intercepts):
        in_clusters = np.array([itr in current for current in current_inds])
        if not in_clusters.any():
            slope_cluster = np.logical_and(slopes < (slope+slope_similarity_threshold), slopes > (slope-slope_similarity_threshold))
            intercept_cluster = np.logical_and(intercepts < (intercept+intercept_similarity_threshold), intercepts > (intercept-intercept_similarity_threshold))
            inds = np.argwhere(slope_cluster & intercept_cluster & slopes_horizontal).T
            if inds.size:
                current_inds.append(inds.flatten())
                clusters.append(lines[inds])
        itr += 1
        
    # Step 4: Merge all lines in clusters using mean averaging
    merged_lines = [np.mean(cluster, axis=1) for cluster in clusters]
    merged_lines = np.array(merged_lines).reshape((-1, 4))
    
    # Note: Make sure dimensions of returned lines is (N x 4)
    return merged_lines

def filter_detections_by_segmentation(detections, segmentation_output):
    """
    Filter 2D detection output based on a semantic segmentation map.

    Arguments:
    detections -- tensor of dimension (N, 5) containing detections in the form of [Class, x_min, y_min, x_max, y_max, score].
    
    segmentation_output -- tensor of dimension (HxW) containing pixel category labels.
    
    Returns:
    filtered_detections -- tensor of dimension (N, 5) containing detections in the form of [Class, x_min, y_min, x_max, y_max, score].

    """
    # Set ratio threshold:
    ratio_threshold = 0.3  # If 1/3 of the total pixels belong to the target category, the detection is correct.
    filtered_detections = []
    
    for detection in detections:
        
        # Step 1: Compute number of pixels belonging to the category for every detection.
        class_name, x_min, y_min, x_max, y_max, score = detection
        x_min = int(float(x_min))
        y_min = int(float(y_min))
        x_max = int(float(x_max))
        y_max = int(float(y_max))
        box_area = (x_max-x_min) * (y_max-y_min)
        if class_name == 'Car':
            class_index = 10
        elif class_name == 'Pedestrian':
            class_index = 4
        correct_pixels = len(np.where(segmentation_output[y_min:y_max, x_min:x_max] == class_index)[0])
        
        # Step 2: Devide the computed number of pixels by the area of the bounding box (total number of pixels).
        ratio = correct_pixels / box_area
            
        # Step 3: If the ratio is greater than a threshold keep the detection. Else, remove the detection from the list of detections.
        if ratio > ratio_threshold:
            filtered_detections.append(detection)
    
    return filtered_detections

def find_min_distance_to_detection(detections, x, y, z):
    """
    Filter 2D detection output based on a semantic segmentation map.

    Arguments:
    detections -- tensor of dimension (N, 5) containing detections in the form of [Class, x_min, y_min, x_max, y_max, score].
    
    x -- tensor of dimension (H, W) containing the x coordinates of every pixel in the camera coordinate frame.
    y -- tensor of dimension (H, W) containing the y coordinates of every pixel in the camera coordinate frame.
    z -- tensor of dimensions (H,W) containing the z coordinates of every pixel in the camera coordinate frame.
    Returns:
    min_distances -- tensor of dimension (N, 1) containing distance to impact with every object in the scene.

    """
    min_distances = []

    for detection in detections:
        # Step 1: Compute distance of every pixel in the detection bounds
        class_name, x_min, y_min, x_max, y_max, score = detection
        x_min = int(float(x_min))
        y_min = int(float(y_min))
        x_max = int(float(x_max))
        y_max = int(float(y_max))
        box_x = x[y_min:y_max, x_min:x_max]
        box_y = y[y_min:y_max, x_min:x_max]
        box_z = z[y_min:y_max, x_min:x_max]
        box_distances = np.sqrt(box_x**2 + box_y**2 + box_z**2)
        
        # Step 2: Find minimum distance
        min_distances.append(np.min(box_distances))

    return min_distances