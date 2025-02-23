import os
import threading

import cv2
import numpy as np
import msgpack

# Note: Please update the world.intrinsics file. You can get the file from Pupil Capture's recording folder.
# file path is in the same folder as this file
dir_path = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(dir_path, "world.intrinsics"), "rb") as f:
    intrinsics = msgpack.unpack(f, raw=False)

# Extract the intrinsics for the specific resolution
resolution_key = '(1920, 1080)'
intrinsics = intrinsics[resolution_key]

# Get the camera matrix and distortion coefficients from intrinsics
camera_matrix = np.array(intrinsics['camera_matrix'])
dist_coeffs = np.array(intrinsics['dist_coefs']).reshape(-1)


def abs_diff(image1, image2):
    diff = cv2.absdiff(image1, image2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, diff_binary = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)

    # Calculate fraction of image that's different
    diff_fraction = np.sum(diff_binary > 0) / (diff_binary.shape[0] * diff_binary.shape[1])
    return diff_fraction


# Function to undistort image
def undistort_img(img):
    img_undist = cv2.undistort(img, camera_matrix, dist_coeffs)

    return img_undist


def store_img(path, img):
    threading.Thread(target=store_img_async, args=(path, img)).start()


def compare_img(image1, image2):
    return abs_diff(image1, image2)


def store_img_async(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(undistort_img(img), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    # Load an image
    path = "data/recordings/pilot_crz/image/23_47_05.png"
    img = cv2.imread(path)

    # Undistort the image
    undistorted_img = undistort_img(img)

    # Display the undistorted image
    cv2.imshow('undistorted', undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
