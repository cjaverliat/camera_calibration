import numpy as np
import cv2
from typing import Tuple

from undistort_image import undistort_image_fisheye

def create_charuco_board():
    charuco_board = cv2.aruco.CharucoBoard((11, 8), 0.035, 0.027, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100))
    charuco_board.setLegacyPattern(True)
    return charuco_board

board = create_charuco_board()

dict = board.getDictionary()
n_cols = board.getChessboardSize()[0]
n_rows = board.getChessboardSize()[1]
n_corners = (n_cols - 1) * (n_rows - 1)

objp = np.zeros((n_corners, 1, 3), np.float32)
objp[:,0,:2] = (np.mgrid[0:10, 0:7].T.reshape(-1, 2) + np.array([1, 1], dtype=np.float32)) * 0.035

def compute_camera_position(frame: cv2.typing.MatLike, K: cv2.typing.MatLike, D: cv2.typing.MatLike, crop_x, crop_y, crop_width, crop_height) -> Tuple[np.ndarray, np.ndarray]:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    undistorted_frame, new_K = undistort_image_fisheye(frame, K, D)

    cropped_frame = undistorted_frame[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]

    # Detect markers in the undistorted image
    corners, ids, rejected_corners = cv2.aruco.detectMarkers(cropped_frame, dict)

    if ids is None:
        raise Exception("No markers detected")

    corners, ids, rejected_corners, _ = cv2.aruco.refineDetectedMarkers(cropped_frame, board, corners, ids, rejected_corners)
    charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, cropped_frame, board)

    # convert back to original image coordinates
    charuco_corners = charuco_corners + np.array([crop_x, crop_y])

    if not charuco_retval or len(charuco_corners) < 4:
        raise Exception("Not enough charuco corners detected")

    new_D = np.zeros((4, 1))
    _, rvec, tvec = cv2.solvePnP(objp[charuco_ids.ravel()], charuco_corners, cameraMatrix=new_K, distCoeffs=new_D)
    return rvec, tvec


frame5 = cv2.imread("/media/charles/HDD/Charles_JAVERLIAT/Calibration GoPro/extrinsics/gopro5_frame244.png")
K5 = np.load("./output/gopro5_K.npy")
D5 = np.load("./output/gopro5_D.npy")
crop5_x = 2550
crop5_y = 1920
crop5_width = 350
crop5_height = 190

frame4 = cv2.imread("/media/charles/HDD/Charles_JAVERLIAT/Calibration GoPro/extrinsics/gopro4_frame244.png")
K4 = np.load("./output/gopro4_K.npy")
D4 = np.load("./output/gopro4_D.npy")
crop4_x = 2280
crop4_y = 1940
crop4_width = 350
crop4_height = 250

rvec4, tvec4 = compute_camera_position(frame4, K4, D4, crop4_x, crop4_y, crop4_width, crop4_height)
rvec5, tvec5 = compute_camera_position(frame5, K5, D5, crop5_x, crop5_y, crop5_width, crop5_height)

def world_to_camera_mtx(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.hstack((R, tvec))
    T = np.vstack((T, np.array([0, 0, 0, 1])))
    return T

gopro4_w2c = world_to_camera_mtx(rvec4, tvec4)
gopro5_w2c = world_to_camera_mtx(rvec5, tvec5)

gopro4_c2w = np.linalg.inv(gopro4_w2c)
gopro5_c2w = np.linalg.inv(gopro5_w2c)

origin_world_pos = np.array([0, 0, 0, 1])
origin_cam1_pos = gopro4_w2c @ origin_world_pos
origin_cam2_pos = gopro5_w2c @ origin_world_pos

gopro4_world_pos = gopro4_c2w @ np.array([0, 0, 0, 1])
gopro5_world_pos = gopro5_c2w @ np.array([0, 0, 0, 1])
gopro5_gopro4_pos = gopro4_w2c @ gopro5_world_pos

print(np.linalg.norm(gopro5_gopro4_pos[:3]))

print("Chessboard (0,0,0) distance from gopro4", np.linalg.norm(origin_cam1_pos[:3]))
print("Chessboard (0,0,0) distance from gopro5", np.linalg.norm(origin_cam2_pos[:3]))
print("gopro5 position in gopro4 frame", gopro5_gopro4_pos)