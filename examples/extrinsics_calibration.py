from camera_calibration.extrinsics import calibrate_cameras_extrinsics
from camera_calibration.camera import Camera
from camera_calibration.board import Board
from camera_calibration.vis.visualizer import Visualizer
from os import path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


def opencv_to_blender(position):
    return np.array([position[1], position[0], -position[2]])


def _create_pair(camera1: Camera, image1_path: str, camera2: Camera, image2_path: str):
    return (
        camera1,
        cv2.imread(
            path.join(filepath, "data", camera1.name, "extrinsics", image1_path)
        ),
        camera2,
        cv2.imread(
            path.join(filepath, "data", camera2.name, "extrinsics", image2_path)
        ),
    )


if __name__ == "__main__":

    # Prepare the cameras
    filepath = path.dirname(__file__)

    # Identical camera intrinsics for all cameras
    camera_intrinsics_path = path.join(filepath, "data/camera_1/camera_intrinsics.txt")
    camera_distortions_path = path.join(
        filepath, "data/camera_1/camera_distortions.txt"
    )
    camera_resolution_path = path.join(filepath, "data/camera_1/camera_resolution.txt")

    camera_intrinsics = np.loadtxt(camera_intrinsics_path)
    camera_distortions = np.loadtxt(camera_distortions_path)
    camera_resolution = np.loadtxt(camera_resolution_path).astype(int)

    gt_cameras: list[Camera] = []

    for i in range(4):
        gt_camera_extrinsics_path = path.join(
            filepath, "data", f"camera_{i+1}", "camera_extrinsics.txt"
        )
        gt_camera_extrinsics = np.loadtxt(gt_camera_extrinsics_path)
        gt_camera = Camera(
            f"gt_camera_{i+1}",
            (camera_resolution[0], camera_resolution[1]),
            camera_intrinsics,
            camera_distortions,
            gt_camera_extrinsics,
        )
        gt_cameras.append(gt_camera)

    # Normalize the camera intrinsics
    camera_intrinsics[0, 0] /= camera_resolution[0]
    camera_intrinsics[1, 1] /= camera_resolution[1]
    camera_intrinsics[0, 2] /= camera_resolution[0]
    camera_intrinsics[1, 2] /= camera_resolution[1]

    img_resolution = cv2.imread(
        path.join(filepath, "data/camera_1/extrinsics/0000.png")
    ).shape[:2][::-1]

    camera_intrinsics[0, 0] *= img_resolution[0]
    camera_intrinsics[1, 1] *= img_resolution[1]
    camera_intrinsics[0, 2] *= img_resolution[0]
    camera_intrinsics[1, 2] *= img_resolution[1]

    camera1 = Camera("camera_1", img_resolution, camera_intrinsics, camera_distortions)
    camera2 = Camera("camera_2", img_resolution, camera_intrinsics, camera_distortions)
    camera3 = Camera("camera_3", img_resolution, camera_intrinsics, camera_distortions)
    camera4 = Camera("camera_4", img_resolution, camera_intrinsics, camera_distortions)

    board = Board((7, 10), 0.015)

    ref = (
        camera2,
        cv2.imread(path.join(filepath, "data/camera_2/extrinsics/0000.png")),
    )

    pairs = [
        _create_pair(camera1, "0001.png", camera2, "0001.png"),
        _create_pair(camera2, "0002.png", camera3, "0002.png"),
        _create_pair(camera3, "0003.png", camera4, "0003.png"),
    ]

    estimated_cameras, graph = calibrate_cameras_extrinsics(ref, pairs, board)

    for camera in estimated_cameras:
        gt_camera_extrinsics_path = path.join(
            filepath, "data", camera.name, "camera_extrinsics.txt"
        )
        expected_camera_extrinsics = np.loadtxt(gt_camera_extrinsics_path)
        expected_pos = np.linalg.inv(expected_camera_extrinsics)[:3, 3]
        expected_rot = R.from_matrix(expected_camera_extrinsics[:3, :3]).as_quat()

        estimated_pos = np.linalg.inv(camera.w2c)[:3, 3]
        estimated_rot = R.from_matrix(camera.w2c[:3, :3]).as_quat()

        position_error = np.linalg.norm(estimated_pos - expected_pos)
        angle_error = np.arccos(np.dot(expected_rot, estimated_rot)) * 2

        print(
            f"{camera.name}: position error={position_error * 1000:.2f}mm, angle error={np.rad2deg(angle_error):.2f}°"
        )

    vis = Visualizer()
    vis.add_cameras(estimated_cameras, color=(0, 0, 1.0))
    vis.add_cameras(gt_cameras, color=(1.0, 0, 0))
    vis.reset_view()
    vis.run()
