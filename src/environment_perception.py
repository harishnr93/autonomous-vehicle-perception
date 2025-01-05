"""
Date: 04.Dec.2024
Author: Harish Natarajan Ravi
Email: harrish.nr@gmail.com
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from m6bk import *
import perception_algos

np.random.seed(1)

dataset_handler = DatasetHandler()

# current frame -> frame 0
current_frame = dataset_handler.current_frame
#print('Current Frame: {0}'.format(current_frame))

# frame contents - Image, Depth, Object Detection and Semantic Segmentation

image = dataset_handler.image
plt.figure(figsize=(8, 6), dpi=100)
plt.title("RGB Image")
plt.imshow(image)

# K matrix
k = dataset_handler.k
#print(k)

depth = dataset_handler.depth
plt.figure(figsize=(8, 6), dpi=100)
plt.title("Depth Map")
plt.imshow(depth, cmap='jet')

segmentation = dataset_handler.segmentation
plt.figure(figsize=(8, 6), dpi=100)
plt.title("Semantic Segmentation")
plt.imshow(segmentation)

"""
-----------------------------------------------------
Category      | Mapping Index | Visualization Colour
-----------------------------------------------------
Background    |      0        |       Black         |                     
Buildings     |      1        |       Red           |   
Pedestrains   |      4        |       Teal          |   
Poles         |      5        |       White         |      
Lane Markings |      6        |       Purple        |       
Roads         |      7        |       Blue          |
Side Walks    |      8        |       Yellow        |
Vehicles      |      10       |       Green         |
-----------------------------------------------------
"""
colored_segmentation = dataset_handler.vis_segmentation(segmentation)
plt.figure(figsize=(8, 6), dpi=100)
plt.title("Colored Segmentation")
plt.imshow(colored_segmentation)

dataset_handler.set_frame(2)
dataset_handler.current_frame

image = dataset_handler.image
#plt.imshow(image)

# Drivable Space Estimation Using Semantic Segmentation Output

dataset_handler.set_frame(0)

k = dataset_handler.k

z = dataset_handler.depth

# Estimating the x, y, and z coordinates of every pixel in the image
x, y = perception_algos.xy_from_depth(z, k)

print('x[800,800] = ' + str(x[800, 800]))
print('y[800,800] = ' + str(y[800, 800]))
print('z[800,800] = ' + str(z[800, 800]) + '\n')

print('x[500,500] = ' + str(x[500, 500]))
print('y[500,500] = ' + str(y[500, 500]))
print('z[500,500] = ' + str(z[500, 500]) + '\n')

# Get road mask by choosing pixels in segmentation output with value 7
road_mask = np.zeros(segmentation.shape)
road_mask[segmentation == 7] = 1

# Road mask
plt.figure(figsize=(8, 6), dpi=100)
plt.title("Road Mask")
plt.imshow(road_mask)

# Get x,y, and z coordinates of pixels in road mask
x_ground = x[road_mask == 1]
y_ground = y[road_mask == 1]
z_ground = dataset_handler.depth[road_mask == 1]
xyz_ground = np.stack((x_ground, y_ground, z_ground))

# RANSAC for plane estimation

p_final = perception_algos.ransac_plane_fit(xyz_ground)
print('Ground Plane: ' + str(p_final))

# visualise inlier set computed on the image

dist = np.abs(dist_to_plane(p_final, x, y, z))

ground_mask = np.zeros(dist.shape)

ground_mask[dist < 0.1] = 1
ground_mask[dist > 0.1] = 0

plt.figure(figsize=(8, 6), dpi=100)
plt.title("Ground Mask")
plt.imshow(ground_mask)

# Drivable space in 3D
"""
visualization only shows where the self-driving car can physically travel. 
The obstacles such as the SUV to the left of the image, can be seen as dark pixels in our visualization

"""

dataset_handler.plot_free_space(ground_mask)

# Lane Estimation Using The Semantic Segmentation Output

# Estimating Lane Boundary
lane_lines = perception_algos.estimate_lane_lines(segmentation)
print(lane_lines.shape)
plt.figure(figsize=(8, 6), dpi=100)
plt.title("Lane Visualization")
plt.imshow(dataset_handler.vis_lanes(lane_lines))

# Merging and Filtering Lane Lines
merged_lane_lines = perception_algos.merge_lane_lines(lane_lines)
plt.figure(figsize=(8, 6), dpi=100)
plt.title("Merged and Filtered Lane Visualization")
plt.imshow(dataset_handler.vis_lanes(merged_lane_lines))

max_y = dataset_handler.image.shape[0]
min_y = np.min(np.argwhere(road_mask == 1)[:, 0])

extrapolated_lanes = extrapolate_lines(merged_lane_lines, max_y, min_y)
final_lanes = find_closest_lines(extrapolated_lanes, dataset_handler.lane_midpoint)
plt.figure(figsize=(8, 6), dpi=100)
plt.title("Extrapolated Lane Visualization")
plt.imshow(dataset_handler.vis_lanes(final_lanes))

# Computing Minimum Distance To Impact Using The Output of 2D Object Detection.

detections = dataset_handler.object_detection
plt.figure(figsize=(8, 6), dpi=100)
plt.title("Objects detected")
plt.imshow(dataset_handler.vis_object_detection(detections))

print("Objects detected: \n")
print(detections)
print("\n")

# Filtering Out Unreliable Detections

filtered_detections = perception_algos.filter_detections_by_segmentation(detections, segmentation)
plt.figure(figsize=(8, 6), dpi=100)
ax = plt.subplot(111)
ax.imshow(dataset_handler.vis_object_detection(filtered_detections))
#ax.set_title("Filtered - Objects detected")
ax.set_title("Estimated distance to impact")

# Estimating Minimum Distance To Impact

min_distances = perception_algos.find_min_distance_to_detection(filtered_detections, x, y, z)
print('Minimum distance to impact is: ' + str(min_distances))

# Visualise estimated distance with 2D detected output

font = {'family': 'serif','color': 'red','weight': 'normal','size': 12}

im_out = dataset_handler.vis_object_detection(filtered_detections)

for detection, min_distance in zip(filtered_detections, min_distances):
    bounding_box = np.asarray(detection[1:5], dtype=np.float32) 
    plt.text(bounding_box[0], bounding_box[1] - 20, 'Distance to Impact:' + str(np.round(min_distance, 2)) + ' m', fontdict=font)


plt.figure(figsize=(8, 6), dpi=100)
ay = plt.subplot(111)
ay.imshow(im_out)
ay.set_title("Filtered - Objects detected")

print("Done!!")
plt.show()