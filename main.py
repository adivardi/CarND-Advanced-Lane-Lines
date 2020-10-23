#!/usr/bin/env python
from camera_calibration import run_calibration, undistort

VISUALIZE = True


def run_pipeline():
    # Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    camera_matrix, distortion_coeffs = run_calibration("./camera_cal", VISUALIZE)

    # open a video

    # Apply a distortion correction to raw images.
    for img in stream:
        img = undistort(img, camera_matrix, distortion_coeffs)


        # Use color transforms, gradients, etc., to create a thresholded binary image.
        # Apply a perspective transform to rectify binary image ("birds-eye view").
        # Detect lane pixels and fit to find the lane boundary.
        # Determine the curvature of the lane and vehicle position with respect to center.
        # Warp the detected lane boundaries back onto the original image.
        # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


if __name__ == '__main__':
    run_pipeline()
