#!/usr/bin/env python
import cv2
import numpy as np
from IPython import embed

from camera_calibration import run_calibration, undistort
from thresholding import threshold

VISUALIZE = True
CALIBRATE = False


def run_pipeline():
    # 1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    if CALIBRATE:
        camera_matrix, distortion_coeffs = run_calibration("./camera_cal", VISUALIZE)
    else:  # use cached
        camera_matrix = np.array([[1.15777942e+03, 0.00000000e+00, 6.67111050e+02],
                                 [0.00000000e+00, 1.15282305e+03, 3.86129068e+02],
                                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        distortion_coeffs = np.array([[-0.24688833, -0.02372816, -0.00109843, 0.00035105, -0.00259134]])

    print("camera martix: {}".format(camera_matrix))
    print("distortion_coeffs: {}".format(distortion_coeffs))

    # open a video
    stream = cv2.VideoCapture('./challenge_video.mp4')
    while (stream.isOpened()):
        ret, frame = stream.read()
        if not ret:
            break

        # 2. Use color transforms, gradients, etc., to create a thresholded binary image.
        img_binary = threshold(frame, VISUALIZE)

        # 3. Apply a perspective transform to rectify binary image ("birds-eye view").

        # cv2.imshow('frame', frame)

        # Press Q on keyboard to  exit
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break


    stream.release()
    cv2.destroyAllWindows()

#
        #
        # Detect lane pixels and fit to find the lane boundary.
        # Determine the curvature of the lane and vehicle position with respect to center.
        # Warp the detected lane boundaries back onto the original image.
        # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


if __name__ == '__main__':
    run_pipeline()
