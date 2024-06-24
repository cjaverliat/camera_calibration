import numpy as np
from camera_calibration.chessboard_detection import find_chessboard_corners
import cv2
from camera_calibration.camera import Camera
from camera_calibration.board import Board


def calibrate_camera_intrinsics(
    camera: Camera,
    imgs: np.ndarray,
    board: Board,
    allow_interactive_warp_crop=True,
    flags=cv2.CALIB_CB_FAST_CHECK
    + cv2.CALIB_CB_ADAPTIVE_THRESH
    + cv2.CALIB_CB_NORMALIZE_IMAGE,
    subpix_win_size=(3, 3),
    subpix_zero_zone=(-1, -1),
    subpix_criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
) -> tuple[Camera, float, int]:

    object_points: list[cv2.typing.MatLike] = []
    image_points: list[cv2.typing.MatLike] = []
    imgs_count = 0

    for img in imgs:
        ret, corners = find_chessboard_corners(
            img,
            board.pattern_size,
            allow_interactive_warp_crop,
            flags,
            subpix_win_size,
            subpix_zero_zone,
            subpix_criteria,
        )

        if ret and corners is not None:
            object_points.append(board.points)
            image_points.append(corners)
            imgs_count += 1

    rms, K, d, _, _ = cv2.calibrateCamera(
        objectPoints=object_points,
        imagePoints=image_points,
        imageSize=imgs[0].shape[:2][::-1],
        cameraMatrix=np.eye(3),
        distCoeffs=np.zeros((5, 1)),
    )

    camera.K = K
    camera.d = d
    return camera, rms, imgs_count
