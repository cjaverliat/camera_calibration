import bpy
import numpy as np
from os import path, makedirs


# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == "VERTICAL":
        return sensor_y
    return sensor_x


# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == "AUTO":
        if size_x >= size_y:
            return "HORIZONTAL"
        else:
            return "VERTICAL"
    return sensor_fit


def get_camera_intrinsics_matrix(obj):
    """
    Build intrinsic camera matrix from Blender camera data.

    See notes on this in # https://blender.stackexchange.com/a/38189
    """
    if obj.type != "CAMERA":
        raise ValueError("Object is not a camera.")

    camd = obj.data

    if camd.type != "PERSP":
        raise ValueError("Non-perspective cameras not supported")
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(
        camd.sensor_fit, camd.sensor_width, camd.sensor_height
    )
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px,
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == "HORIZONTAL":
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0  # only use rectangular pixels

    K = np.array([[s_u, skew, u_0], [0, s_v, v_0], [0, 0, 1]])
    return K


def get_camera_resolution(obj):
    if obj.type != "CAMERA":
        raise ValueError("Object is not a camera.")
    scene = bpy.context.scene
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    return np.array([resolution_x_in_px, resolution_y_in_px])


def get_camera_extrinsics_matrix(obj):
    """
    Return the world to camera matrix for the given camera.
    """
    if obj.type != "CAMERA":
        raise ValueError("Object is not a camera.")

    BLENDER_TO_OPENCV = np.diag([1, -1, -1])

    location, rotation = obj.matrix_world.decompose()[0:2]
    R = rotation.to_matrix().transposed()
    T = -1.0 * R @ location

    R = BLENDER_TO_OPENCV @ R
    T = BLENDER_TO_OPENCV @ T

    R[:, [0, 1, 2]] = R[:, [1, 0, 2]]
    R[:, 2] = -R[:, 2]
    return np.vstack([np.column_stack((R, T)), [0, 0, 0, 1]])


def get_camera_distortions(obj):
    if obj.type != "CAMERA":
        raise ValueError("Object is not a camera.")
    # TODO: support lens distortion nodes and fisheye lenses
    return np.zeros(5)


def export_camera_ground_truth(obj, output_dir):
    if obj.type != "CAMERA":
        raise ValueError("Object is not a camera.")

    if not path.exists(output_dir):
        makedirs(output_dir, exist_ok=True)

    resolution = get_camera_resolution(obj)
    distortions = get_camera_distortions(obj)
    intrinsics = get_camera_intrinsics_matrix(obj)
    extrinsics = get_camera_extrinsics_matrix(obj)

    np.savetxt(path.join(output_dir, "camera_resolution.txt"), resolution)
    np.savetxt(path.join(output_dir, "camera_distortions.txt"), distortions)
    np.savetxt(path.join(output_dir, "camera_intrinsics.txt"), intrinsics)
    np.savetxt(path.join(output_dir, "camera_extrinsics.txt"), extrinsics)


if __name__ == "__main__":
    for obj in bpy.data.objects:
        if obj.type == "CAMERA":
            camera_name = obj.name

            if camera_name != "Camera_1":
                continue

            output_dir = bpy.path.abspath("//")
            output_dir = path.join(output_dir, camera_name)
            export_camera_ground_truth(obj, output_dir)
