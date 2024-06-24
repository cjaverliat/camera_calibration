import numpy as np
import open3d as o3d
from camera_calibration.extrinsics import Camera

OPENCV_TO_OPEN3D = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])


def _add_origin_axes(vis: o3d.visualization.O3DVisualizer):
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry("Axes", axes)


def _add_camera_mesh(vis: o3d.visualization.O3DVisualizer, camera: Camera):
    cam_line_sets = o3d.geometry.LineSet.create_camera_visualization(
        camera.resolution[0],
        camera.resolution[1],
        camera.K,
        camera.w2c @ OPENCV_TO_OPEN3D,
        scale=0.1,
    )

    vis.add_geometry(camera.name, cam_line_sets)


def show_cameras(cameras: list[Camera]):

    app = o3d.visualization.gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer()
    vis.show_skybox(False)
    vis.show_ground = True

    _add_origin_axes(vis)

    for camera in cameras:
        _add_camera_mesh(vis, camera)

    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()
