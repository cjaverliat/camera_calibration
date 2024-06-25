from camera_calibration.extrinsics import calibrate_cameras_extrinsics
from camera_calibration.camera import Camera
from camera_calibration.board import Board
from camera_calibration.vis.camera_visualizer import show_cameras
from os import path
import numpy as np
import cv2


def opencv_to_blender(position):
    return np.array([position[1], position[0], -position[2]])


def _read_img(img_name):
    return cv2.imread(path.join(filepath, "data", "extrinsics", img_name))


def _create_pair(camera1, image1, camera2, image2):
    return (camera1, _read_img(image1), camera2, _read_img(image2))


if __name__ == "__main__":

    # Prepare the cameras
    filepath = path.dirname(__file__)
    camera_intrinsics_path = path.join(filepath, "data", "gt_camera_intrinsics.txt")
    camera_distortions_path = path.join(filepath, "data", "gt_camera_distortions.txt")
    camera_resolution_path = path.join(filepath, "data", "gt_camera_resolution.txt")

    camera_intrinsics = np.loadtxt(camera_intrinsics_path)
    camera_distortions = np.loadtxt(camera_distortions_path)
    camera_resolution = np.loadtxt(camera_resolution_path)

    # Normalize the camera intrinsics
    camera_intrinsics[0, 0] /= camera_resolution[0]
    camera_intrinsics[1, 1] /= camera_resolution[1]
    camera_intrinsics[0, 2] /= camera_resolution[0]
    camera_intrinsics[1, 2] /= camera_resolution[1]

    img_resolution = _read_img("0000_1.png").shape[:2][::-1]

    camera_intrinsics[0, 0] *= img_resolution[0]
    camera_intrinsics[1, 1] *= img_resolution[1]
    camera_intrinsics[0, 2] *= img_resolution[0]
    camera_intrinsics[1, 2] *= img_resolution[1]

    camera1 = Camera("Camera 1", img_resolution, camera_intrinsics, camera_distortions)
    camera2 = Camera("Camera 2", img_resolution, camera_intrinsics, camera_distortions)
    camera3 = Camera("Camera 3", img_resolution, camera_intrinsics, camera_distortions)
    camera4 = Camera("Camera 4", img_resolution, camera_intrinsics, camera_distortions)

    board = Board((7, 10), 0.015)

    ref = (camera2, _read_img("0000_2.png"))

    pairs = [
        _create_pair(camera1, "0001_1.png", camera2, "0001_2.png"),
        _create_pair(camera2, "0002_2.png", camera3, "0002_3.png"),
        _create_pair(camera3, "0003_3.png", camera4, "0003_4.png"),
    ]

    # Calibrate the cameras
    cameras, graph = calibrate_cameras_extrinsics(ref, pairs, board)

    for camera in cameras:
        # offset = np.array([0, -0.015 * 6, 0])
        camera_pos = np.linalg.inv(camera.w2c)[:3, 3]
        camera_pos = opencv_to_blender(camera_pos)  # + offset
        print(camera.name, camera_pos)

    show_cameras(cameras)
