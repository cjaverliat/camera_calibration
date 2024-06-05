from dataclasses import dataclass
import numpy as np
import cv2

@dataclass(frozen=True)
class CameraExtrinsics:
    """
    Represents the extrinsic parameters of a camera.

    Attributes:
        rvec (np.ndarray): The rotation vector.
        tvec (np.ndarray): The translation vector.
    """
    rvec: np.ndarray
    tvec: np.ndarray

    def world_to_camera_mtx(self) -> np.ndarray:
        R, _ = cv2.Rodrigues(self.rvec)
        T = np.hstack((R, self.tvec))
        T = np.vstack((T, np.array([0, 0, 0, 1])))
        return T
    
    def camera_to_world_mtx(self) -> np.ndarray:
        return np.linalg.inv(self.world_to_camera_mtx())

@dataclass(frozen=True)
class CameraIntrinsics:
    """
    Represents the intrinsic parameters of a camera.

    Attributes:
        K (cv2.typing.MatLike): The camera matrix in OpenCV format ([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).
        D (cv2.typing.MatLike): The distortion coefficients in OpenCV format (k_1, k_2, p_1, p_2, k_3).
        fisheye_model (bool): Whether the fisheye model was used for calibration.
        image_size (cv2.typing.Size): The size of the camera image.
        rms (float): The root mean square error of the calibration in pixels.
    """
    K: cv2.typing.MatLike
    D: cv2.typing.MatLike
    fisheye_model: bool
    image_size: cv2.typing.Size
    rms: float

def save_intrinsics_to_npz(camera_intrinsics: CameraIntrinsics, output_filename: str):
    """
    Save the camera intrinsics to the file "{output_filename}.npz" containing:
    - The camera matrix K.
    - The distortion coefficients D.
    - Whether the fisheye model was used for calibration.
    - The size of images.
    - The RMS error of the calibration.
    """
    np.savez(output_filename,
             K=camera_intrinsics.K,
             D=camera_intrinsics.D,
             fisheye_model=camera_intrinsics.fisheye_model,
             image_size=camera_intrinsics.image_size,
             rms=camera_intrinsics.rms)

def save_extrinsics_to_npz(camera_extrinsics: CameraExtrinsics, output_filename: str):
    """
    Save the camera extrinsics to the file "{output_filename}.npz" containing:
    - The rotation vector rvec in Rodrigues format (3x1 numpy array)
    - The translation vector tvec (3x1 numpy array)
    """
    np.savez(output_filename,
             rvec=camera_extrinsics.rvec,
             tvec=camera_extrinsics.tvec)
    

def load_intrinsics_from_npz(input_filename: str) -> CameraIntrinsics:
    """
    Load the camera intrinsics from the file "{input_filename}.npz"

    :return: The camera intrinsics.
    """
    data = np.load(input_filename)
    return CameraIntrinsics(K=data["K"], D=data["D"],
                            fisheye_model=data["fisheye_model"],
                            image_size=tuple(data["image_size"]),
                            rms=data["rms"])

def load_extrinsics_from_npz(input_filename: str) -> CameraExtrinsics:
    """
    Load the camera extrinsics from the file "{input_filename}.npz"

    :return: The camera extrinsics.
    """
    data = np.load(input_filename)
    return CameraExtrinsics(rvec=data["rvec"], tvec=data["tvec"])
