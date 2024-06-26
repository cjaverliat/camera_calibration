import numpy as np
import open3d as o3d
from camera_calibration.extrinsics import Camera

OPENCV_TO_OPEN3D = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])


class Visualizer:
    def __init__(
        self,
        verbosity_level: o3d.utility.VerbosityLevel = o3d.utility.VerbosityLevel.Error,
    ):
        o3d.utility.set_verbosity_level(verbosity_level)
        self.app = o3d.visualization.gui.Application.instance
        self.app.initialize()

        self.vis = o3d.visualization.O3DVisualizer()
        self.vis.show_skybox(False)
        self.vis.show_ground = True

        self.app.add_window(self.vis)

        self._add_origin_axes()

    def _add_origin_axes(self):
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry("Axes", axes)

    def _add_camera_mesh(
        self, camera: Camera, color: tuple[float, float, float] = (0, 0, 0)
    ):
        camera_mesh = o3d.geometry.LineSet.create_camera_visualization(
            camera.resolution[0],
            camera.resolution[1],
            camera.K,
            camera.w2c @ OPENCV_TO_OPEN3D,
            scale=0.1,
        )
        camera_mesh.paint_uniform_color(color)
        self.vis.add_geometry(camera.name, camera_mesh)

    def run(self):
        self.app.run()

    def reset_view(self):
        self.vis.reset_camera_to_default()

    def add_cameras(
        self,
        cameras: list[Camera],
        color: tuple[float, float, float],
    ):
        for camera in cameras:
            self._add_camera_mesh(camera, color)
