import cv2
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from ..utils import interactive_image_crop


def detect_classic_board(
    image: cv2.typing.MatLike,
    pattern_size: cv2.typing.Size,
    square_length: float,
    interactive_crop=False,
) -> Tuple[bool, cv2.typing.MatLike, cv2.typing.MatLike]:

    board_corners = np.zeros((1, pattern_size[0] * pattern_size[1], 3), np.float32)
    board_corners[0, :, :2] = (
        np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
        * square_length
    )

    # flags = cv2.CALIB_CB_ACCURACY+cv2.CALIB_CB_NORMALIZE_IMAGE
    flags = (
        cv2.CALIB_CB_FAST_CHECK
        + cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    # Try to find board whithout cropping.
    ret, corners = cv2.findChessboardCorners(image, pattern_size, flags=flags)
    crop_x = None
    crop_y = None

    if not ret:
        # Try again with cropping.
        if interactive_crop:
            crop_x, crop_y = interactive_image_crop(image)
            cropped_image = image[crop_y[0] : crop_y[1], crop_x[0] : crop_x[1]]
            ret, corners = cv2.findChessboardCorners(
                cropped_image, pattern_size, flags=flags
            )

            if not ret:
                return False, None, None
        else:
            return False, None, None

    # If the image was cropped, bring the corners back to the original image coordinates.
    if crop_x is not None and crop_y is not None:
        corners += np.array([crop_x[0], crop_y[0]])

    return True, board_corners, corners


def detect_charuco_board(
    image: cv2.typing.MatLike, board: cv2.aruco.CharucoBoard, interactive_crop=False
) -> Tuple[bool, cv2.typing.MatLike, cv2.typing.MatLike]:
    """
    Detect a ChArUco board in an image and return the object points and image points.

    :param img: The image to detect the ChArUco board in. The image is expected to be free of distortion.
    :param board: The ChArUco board to detect.
    :param interactive_crop: Whether to interactively crop the image to the region of interest.
    :return: A tuple containing a boolean indicating whether the ChArUco board was found, the object points and the image points.
    """
    board_corners = board.getChessboardCorners().reshape(-1, 1, 3)

    # Try to find board whithout cropping.
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(image)
    crop_x = None
    crop_y = None

    if charuco_ids is None:
        # Try again with cropping.
        if interactive_crop:
            crop_x, crop_y = interactive_image_crop(image)
            cropped_image = image[crop_y[0] : crop_y[1], crop_x[0] : crop_x[1]]
            charuco_corners, charuco_ids, _, _ = detector.detectBoard(cropped_image)

            if charuco_ids is None:
                return False, None, None
        else:
            return False, None, None

    # If the image was cropped, bring the corners back to the original image coordinates.
    if crop_x is not None and crop_y is not None:
        charuco_corners += np.array([crop_x[0], crop_y[0]])

    return True, board_corners[charuco_ids.ravel()], charuco_corners
