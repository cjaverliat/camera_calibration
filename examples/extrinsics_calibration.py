from camera_calibration.extrinsics import calibrate_cameras_extrinsics
from camera_calibration.camera import Camera
from camera_calibration.board import Board
from camera_calibration.vis.camera_visualizer import show_cameras
from os import path
import numpy as np
import cv2


def opencv_to_blender(position):
    return np.array([position[1], position[0], -position[2]])


if __name__ == "__main__":

    # Prepare the cameras
    filepath = path.dirname(__file__)
    camera_intrinsics_path = path.join(filepath, "data", "camera_intrinsics.txt")
    camera_distortions_path = path.join(filepath, "data", "camera_distortions.txt")

    camera_intrinsics = np.loadtxt(camera_intrinsics_path)
    camera_distortions = np.loadtxt(camera_distortions_path)

    camera1 = Camera("Camera 1", (5312, 2988), camera_intrinsics, camera_distortions)
    camera2 = Camera("Camera 2", (5312, 2988), camera_intrinsics, camera_distortions)
    camera3 = Camera("Camera 3", (5312, 2988), camera_intrinsics, camera_distortions)
    camera4 = Camera("Camera 4", (5312, 2988), camera_intrinsics, camera_distortions)

    # Prepare the board
    board = Board((7, 10), 0.015)

    # Prepare the pairs

    ref = (camera1, cv2.imread(path.join(filepath, "data", "img", "calib0_1.png")))

    pairs = [
        (
            camera1,
            cv2.imread(path.join(filepath, "data", "img", "calib1_1.png")),
            camera2,
            cv2.imread(path.join(filepath, "data", "img", "calib1_2.png")),
        ),
        (
            camera2,
            cv2.imread(path.join(filepath, "data", "img", "calib2_2.png")),
            camera3,
            cv2.imread(path.join(filepath, "data", "img", "calib2_3.png")),
        ),
        (
            camera3,
            cv2.imread(path.join(filepath, "data", "img", "calib3_3.png")),
            camera4,
            cv2.imread(path.join(filepath, "data", "img", "calib3_4.png")),
        ),
    ]

    # Calibrate the cameras

    cameras, graph = calibrate_cameras_extrinsics(ref, pairs, board)

    for camera in cameras:
        offset = np.array([0, -0.015 * 6, 0])
        camera_pos = np.linalg.inv(camera.w2c)[:3, 3]
        camera_pos = opencv_to_blender(camera_pos) + offset
        print(camera.name, camera_pos)

    show_cameras(cameras)
