import cv2
from typing import Tuple
import numpy as np

from .utils import interactive_image_crop

def detect_board(img: cv2.typing.MatLike, pattern_size: cv2.typing.Size, square_length: float, interactive_crop=False) -> Tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    """
    Detect a chessboard in an image and return the object points and image points.

    :param img: The image to detect the chessboard in. The image is expected to be free of distortion.
    :param pattern_size: The size of the chessboard pattern.
    :param square_length: The length of a square on the chessboard.
    :return: A tuple containing a boolean indicating whether the chessboard was found, the object points and the image points.
    """
    objp = np.zeros((pattern_size[0] * pattern_size[1], 1, 3), np.float32)
    objp[:,0,:2] = (np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) + np.array([1, 1], dtype=np.float32)) * square_length

    if interactive_crop:
        crop_x, crop_y = interactive_image_crop(img)
        img = img[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]

    ret, corners = cv2.findChessboardCornersSB(img, pattern_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY + cv2.CALIB_CB_LARGER)

    if not ret:
        return False, None, None

    if interactive_crop:
        corners += np.array([crop_x[0], crop_y[0]])

    return ret, objp, corners

def detect_charuco_board(img: cv2.typing.MatLike, board: cv2.aruco.CharucoBoard, interactive_crop=False) -> Tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    """
    Detect a ChArUco board in an image and return the object points and image points.

    :param img: The image to detect the ChArUco board in. The image is expected to be free of distortion.
    :param board: The ChArUco board to detect.
    :param interactive_crop: Whether to interactively crop the image to the region of interest.
    :return: A tuple containing a boolean indicating whether the ChArUco board was found, the object points and the image points.
    """
    square_length = board.getSquareLength()
    n_cols = board.getChessboardSize()[0]
    n_rows = board.getChessboardSize()[1]
    objp = np.zeros(((n_cols - 1) * (n_rows - 1), 1, 3), np.float32)
    objp[:,0,:2] = (np.mgrid[0:n_cols-1, 0:n_rows-1].T.reshape(-1, 2) + np.array([1, 1], dtype=np.float32)) * square_length
    
    if interactive_crop:
        crop_x, crop_y = interactive_image_crop(img)
        img = img[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]

    corners, ids, rejected_corners = cv2.aruco.detectMarkers(img, board.getDictionary())

    if ids is None:
        return False, None, None

    corners, ids, rejected_corners, _ = cv2.aruco.refineDetectedMarkers(img, board, corners, ids, rejected_corners)
    _, corners, ids = cv2.aruco.interpolateCornersCharuco(corners, ids, img, board)

    if interactive_crop:
        corners += np.array([crop_x[0], crop_y[0]])
    
    return True, objp[ids.ravel()], corners

# Example usage
# 
# charuco_board = cv2.aruco.CharucoBoard((11, 8), 0.035, 0.027, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100))
# charuco_board.setLegacyPattern(True)
#
# from undistort_image import undistort_image_fisheye
#
# gopro = "gopro5"
# im = cv2.imread(f"D:/Charles_JAVERLIAT/Calibration GoPro/extrinsics/{gopro}_frame244.png")
# K = np.load(f"D:/Charles_JAVERLIAT/Technique/camera_calibration/output/{gopro}_K.npy")
# D = np.load(f"D:/Charles_JAVERLIAT/Technique/camera_calibration/output/{gopro}_D.npy")
# im, _ = undistort_image_fisheye(im, K, D)
#
# ret, object_points, img_points = detect_charuco_board(im, charuco_board)
#
# if not ret:
#     print("ChArUco board not found. Trying again with interactive cropping.")
#     print("Zoom to the region containing the ChArUco board and press any key to continue.")
#     ret, object_points, img_points = detect_charuco_board(im, charuco_board, interactive_crop=True)
#
# if ret:
#     print("Charuco board found")
#     for img_point in img_points:
#         # convert to int
#         img_point = img_point.reshape(2).astype(int)
#         cv2.drawMarker(im, img_point, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
#     plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
#     plt.show()
# else:
#     print("ChArUco board not found")