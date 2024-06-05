from matplotlib import pyplot as plt
import cv2
from typing import Tuple
import numpy as np
from .camera import CameraIntrinsics

def undistort_img(img: cv2.typing.MatLike, camera_intrinsics: CameraIntrinsics, crop_borders: bool) -> Tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    """
    Undistort the image using the camera intrinsics (eg. to remove lens distortion due to a fisheye lens).

    :param img: The distorted image to correct.
    :param camera_intrinsics: The camera intrinsics (camera matrix, distortion coefficients, ...).
    :param crop: Whether to crop the image (remove black borders).
    :return: The undistorted image and, new camera matrix (K) and distortion coefficients (D).
    """
    img_size = img.shape[:2][::-1]

    # Change balance to 0 to crop and remove black borders
    balance = int(not crop_borders)
    new_D = np.zeros((4, 1))

    K = camera_intrinsics.K.copy()
    D = camera_intrinsics.D.copy()

    # If the image size is different from the camera intrinsics image size, scale the camera matrix.
    if img_size != camera_intrinsics.image_size:
        aspect_ratio = img_size[0] / img_size[1]
        expected_aspect_ratio = camera_intrinsics.image_size[0] / camera_intrinsics.image_size[1]

        if aspect_ratio != expected_aspect_ratio:
            raise ValueError("Expected aspect ratio does not match the image aspect ratio.")
        
        scaling_factor = img_size[0] / camera_intrinsics.image_size[0]
        K = K * scaling_factor

    # TODO: test the difference between the fisheye and the regular model
    if camera_intrinsics.fisheye_model:
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, img_size, np.eye(3), balance=balance)
        mapx, mapy = cv2.fisheye.initUndistortRectifyMap(K, D, None, new_K, img_size, cv2.CV_32SC2)
        undistorted_image = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_image, new_K, new_D
    
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, img_size, balance, img_size)
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, new_K, img_size, cv2.CV_32SC2)
    undistorted_image = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h = roi
    undistorted_image = img[y:y + h, x:x + w]
    return undistorted_image, new_K, new_D


def interactive_image_crop(img: cv2.typing.MatLike) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Crop an image interactively by selecting a region of interest.

    :param img: The image to crop.
    :return: The x and y limits of the cropped image.
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    fig.canvas.mpl_connect('key_press_event', lambda _, fig=fig: plt.close(fig))
    plt.show()

    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    xlim_min = min(int(xlim[0]), int(xlim[1]))
    xlim_max = max(int(xlim[0]), int(xlim[1]))
    ylim_min = min(int(ylim[0]), int(ylim[1]))
    ylim_max = max(int(ylim[0]), int(ylim[1]))
    return (xlim_min, xlim_max), (ylim_min, ylim_max)
