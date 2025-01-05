## Self-Driving Car Perception Pipeline in Python

This code implements a perception pipeline for a self-driving car using Python libraries like OpenCV, NumPy, and custom perception algorithms defined in a `separate` file.

The script utilizes a pre-recorded dataset containing various frames, including:

* RGB Image
* Depth Map
* Semantic Segmentation
* Object Detection

Here's a breakdown of the functionalities covered in the code:

**1. Data Access and Preprocessing**

* Establishes a connection to the dataset handler to access frames and their contents.
* Loads the RGB image, depth map, semantic segmentation, and object detection data for the current frame.

**2. Semantic Segmentation Analysis**

* Visualizes the semantic segmentation mask, where each pixel value corresponds to a specific object class (e.g., road, vehicle, pedestrian).
* Estimates the x, y, and z coordinates of every pixel in the image using the depth map and camera calibration matrix (function: `xy_from_depth` from `perception_algo.py`).

**3. Drivable Space Estimation**

* Extracts the road mask by selecting pixels with a specific value in the segmentation output corresponding to the road class.
* Performs RANSAC (Random Sample Consensus) to fit a plane to the ground points, effectively estimating the ground plane equation (function: `ransac_plane_fit` from `perception_algo.py`).
* Generates a ground mask to identify pixels belonging to the ground plane.
* Visualizes the drivable space in 3D, representing the areas the self-driving car can navigate.

**4. Lane Line Detection**

* Estimates lane boundaries using the semantic segmentation output (function: `estimate_lane_lines` from `perception_algo.py`).
* Merges and filters the detected lane lines to refine the lane markings (function: `merge_lane_lines` from `perception_algo.py`).
* Extrapolates the lane lines to the entire image height for better visualization.
* Identifies the closest lane lines to the center of the road (lane midpoint).
* Visualizes the final lane lines.

**5. Object Detection and Distance Estimation**

* Retrieves object detections from the dataset handler.
* Filters out unreliable detections based on the semantic segmentation mask to eliminate false positives on non-road surfaces (function: `filter_detections_by_segmentation` from `perception_algo.py`).
* Calculates the minimum distance to each detected object using their 2D bounding boxes and the 3D point cloud information (function: `find_min_distance_to_detection` from `perception_algo.py`).
* Visualizes the filtered detections along with their estimated distances to impact displayed on the image.

**Overall, this script demonstrates a perception pipeline for a self-driving car that leverages various computer vision techniques to extract critical information from sensor data, enabling the car to understand its surroundings, identify drivable areas, and detect potential obstacles.**

By integrating various perception algorithms, the main script - environment_perception can perform various tasks crucial for self-driving car navigation.


