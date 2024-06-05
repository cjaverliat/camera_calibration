from camera_calibration.board import create_charuco_board
import cv2

from glob import glob
from camera_calibration.intrinsics import compute_intrinsics
from camera_calibration.board import create_charuco_board
from tqdm import tqdm

from camera_calibration.board_detection import (
    detect_classic_board,
    detect_charuco_board,
)

from camera_calibration.board import create_charuco_board

board = create_charuco_board(
    size=(11, 8),
    square_length=0.035,
    marker_length=0.027,
    dict=cv2.aruco.DICT_4X4_100,
    legacy=True,
)

for i in range(6):
    gopro_name = f"gopro{i+1}"
    print(f"Computing intrinsics for {gopro_name}...")
    intrinsics_calib_imgs = glob(f"input/{gopro_name}_anonymized/*.png")
    intrinsics = compute_intrinsics(
        intrinsics_calib_imgs, board, interactive_crop=True, use_fisheye_model=True
    )

# frame5 = cv2.imread("/media/charles/HDD/Charles_JAVERLIAT/Calibration GoPro/extrinsics/gopro5_frame244.png")
# K5 = np.load("./output/gopro5_K.npy")
# D5 = np.load("./output/gopro5_D.npy")
# crop5_x = 2550
# crop5_y = 1920
# crop5_width = 350
# crop5_height = 190

# frame4 = cv2.imread("/media/charles/HDD/Charles_JAVERLIAT/Calibration GoPro/extrinsics/gopro4_frame244.png")
# K4 = np.load("./output/gopro4_K.npy")
# D4 = np.load("./output/gopro4_D.npy")
# crop4_x = 2280
# crop4_y = 1940
# crop4_width = 350
# crop4_height = 250

# rvec4, tvec4 = compute_camera_position(frame4, K4, D4, crop4_x, crop4_y, crop4_width, crop4_height)
# rvec5, tvec5 = compute_camera_position(frame5, K5, D5, crop5_x, crop5_y, crop5_width, crop5_height)

# def world_to_camera_mtx(rvec, tvec):
#     R, _ = cv2.Rodrigues(rvec)
#     T = np.hstack((R, tvec))
#     T = np.vstack((T, np.array([0, 0, 0, 1])))
#     return T

# gopro4_w2c = world_to_camera_mtx(rvec4, tvec4)
# gopro5_w2c = world_to_camera_mtx(rvec5, tvec5)

# gopro4_c2w = np.linalg.inv(gopro4_w2c)
# gopro5_c2w = np.linalg.inv(gopro5_w2c)

# origin_world_pos = np.array([0, 0, 0, 1])
# origin_cam1_pos = gopro4_w2c @ origin_world_pos
# origin_cam2_pos = gopro5_w2c @ origin_world_pos

# gopro4_world_pos = gopro4_c2w @ np.array([0, 0, 0, 1])
# gopro5_world_pos = gopro5_c2w @ np.array([0, 0, 0, 1])
# gopro5_gopro4_pos = gopro4_w2c @ gopro5_world_pos

# print(np.linalg.norm(gopro5_gopro4_pos[:3]))

# print("Chessboard (0,0,0) distance from gopro4", np.linalg.norm(origin_cam1_pos[:3]))
# print("Chessboard (0,0,0) distance from gopro5", np.linalg.norm(origin_cam2_pos[:3]))
# print("gopro5 position in gopro4 frame", gopro5_gopro4_pos)
