import numpy as np
import cv2
import matplotlib.pyplot as plt

# pipeline:
# only use sobel x, as it is better for vertical lane lines (see apply_sobel.py and apply_sobel_magnitude.py)


def threshold(img, visualize):
    s_color_thresh = (200, 255)
    sobel_x_thresh = (20, 100)

    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel x in lightness channel
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sobel_x_thresh[0]) & (scaled_sobel <= sobel_x_thresh[1])] = 1

    # Threshold saturation channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_color_thresh[0]) & (s_channel <= s_color_thresh[1])] = 1

    # Stack each channel
    # green is sobel_x on lightness
    # blue is saturation threshold
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # get binary image
    binary = np.zeros_like(sxbinary)
    binary[(sxbinary == 1) | (s_binary == 1)] = 1

    if visualize:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
        f.tight_layout()

        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax1.imshow(RGB_img)
        ax1.set_title('Original Image', fontsize=30)

        ax2.imshow(color_binary)
        ax2.set_title('Colored Binary', fontsize=30)

        ax3.imshow(binary, cmap='gray')
        ax3.set_title('Colored Binary', fontsize=30)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    return color_binary



