import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython import embed


nx = 9  # number of inside corners in x
ny = 6  # number of inside corners in y

# we need compute the mapping from 2D image plane to 3D world coordinates
# for this, we need to associate 2D points in each image place (corners, imgpoints)
# with 3D points in the world (objpoints)
# we can choose the points in the world to fit our chessboard, and can set the chessboard origin to (0,0,0) and assume it is vertical (so z = 0)
# so the 3D points can be simply defined as a grid
# so we finally get (0,0,0), (1,0,0), (2,0,0) ....,(nx,ny,0)
objpoints_single = np.zeros((nx * ny, 3), np.float32)    # number of corners, each 3D
# mgrid gives 3D array where in the 1st layer the x increase each line and the 2nd the y increase each column
# we then reshape into a N x 2 array
objpoints_single[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)


def run_calibration(folder_path, visualize):
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is None:
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if not ret:
            continue

        if visualize:
            img_corners = img.copy()
            img_corners = cv2.drawChessboardCorners(img_corners, (nx, ny), corners, ret)

        imgpoints.append(corners)
        objpoints.append(objpoints_single)

    # use gray.shape[::-1] to flip the shape (function takes (width, height) but shape is usually (height, width))
    ret, camera_matrix, distortion_coeffs, cam_pos_rot, cam_pos_trans = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    if not ret:
        print "Caliubration failed"
        return

    print("camera martix: {}".format(camera_matrix))
    print("distortion_coeffs: {}".format(distortion_coeffs))

    if visualize:
        undistorted_img = cv2.undistort(img, camera_matrix, distortion_coeffs)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img_corners)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(undistorted_img, cmap="gray")
        ax2.set_title('Undistorted Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    return camera_matrix, distortion_coeffs


def undistort(img, camera_matrix, distortion_coeffs):
    return cv2.undistort(img, camera_matrix, distortion_coeffs)
