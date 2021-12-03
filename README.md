# 3D-Vision 2021 Spring Homeworks

This repository contains the homeworks assigned in 3D Vision course in Istanbul Technical University in 2021 Spring.

## Homework 0

### Part 1: Numpy

### Part 2: Image Manipulation - I

### Part 3: Image Manipulation - II

### Part 4: Edge and Shape Detection

## Homework 1

### Task 1 - Plotting a 3D Object

### Task 2 - Exponential Coordinate Representation of 3D Rotations

### Task 3 - Tańcząca Polska Krowa

### Task 4 - Perturbed Rotations

### Task 5 - Quaternion Representation of 3D Rotations

### Task 6 - Questions

### Task 7 - Computing Angles

## Homework 2

### Task 1
#### The first time I’ve listened Myslovitz’s self-titled album, every song in it stuck in my head for at least a week. In the first task of the homework, you will create a video for your favourite album. A frame of the target video is given in Figure 1.

### Task 2
#### In this task, you will implement and experiment with estimation of a Homography matrix between images of a given scene (planar scene) from different views. You will use the Normalized Direct Linear Transform with SVD we studied in the class to estimate the 3x3 Homography matrix H. You will then use the estimated transformation to form a panoramic image, i.e. a combination of the three images by simple stitching.
#### In this task, you will select the corresponding points from the images manually, i.e. by clicking on the point with a mouse.
#### Image Blending: Rather than copying the second image over the first one as you did before, try simple image blending ideas, such as taking convex combinations of images etc.

### Task 3
#### In order to create a panoramic image, typically more than 2 images are used. In this task, you will stitch 3 images to create a panorama. To do that, estimate the homography matrices by taking the center image as your reference and estimate the homographies from the other two images towards the center image. Then repeat what you did in Task 2. Show your final panorama image constructed using 3 images.

### Task 4
#### In this task, you will use a feature extraction and matching technique based on a technique such as SIFT to automatically extract a set of corresponding point coordinates from the given images.

## Homework 3

### Task 1 - Camera Calibration

### Task 2 - Augmented Reality (AR)

## Homework 4

In this assignment, you will work on Epipolar Geometry. You will implement the well-known 8-point Algorithm in 3D vision to reconstruct camera poses, and 3D structure (depth) using point correspondences. Load the images (1.JPG and 2.JPG) provided in assignment zip file. In order to work within a calibrated epipolar geometry scenario, we will use the intrinsic camera parameters, which are given in the cube data.mat file. Load the mat file and see the 3x3 intrinsic camera calibration matrix “Calib”. This mat file also contains a set of 45 point correspondences selected where x1 and x2 contain homogeneous point pairs from image 1 and image 2 respectively. You will estimate the rotation and translation (R,T), i.e. the relative camera pose. In addition, you will obtain a point cloud of the 3D structure from a collection of corresponding points from two views of the given object.

### Task 1 - 8-pt Algorithm

### Task 2 - Decompose the Essential Matrix

### Task 3 - Impose Positive Depth Constraint

### Task 4 - Recover 3D structure as a Point Cloud

### Task 5 - Dense Correspondence with RANSAC and 3D Reconstruction (EXTRA)

## Homework 5

### Task 1 - Motion segmentation using optical flow

### Task 2 - Optical Flow via Deep Learning

### Task 3 - Stereo Rectification

### Task 4 - Dense Feature Matching and Disparity/Depth Estimation from Rectified Stereo Views
