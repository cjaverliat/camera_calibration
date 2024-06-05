"""
Set of utilities to compute the camera intrinsics from a set of images containing a ChArUco board.

We recommend using images where the chessboard is at different positions
and orientations in the camera's field of view for better calibration.

The code is based on the OpenCV tutorials:
- https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html
- https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
"""

import cv2
from warnings import warn
from tqdm import tqdm
from .camera import CameraIntrinsics, CameraExtrinsics
from .board_detection import detect_charuco_board
from .utils import undistort_img


def compute_intrinsics(
    images_path: list[str],
    board: cv2.aruco.CharucoBoard,
    use_fisheye_model: bool = False,
    interactive_crop=True,
) -> CameraIntrinsics:
    """
    Compute the camera intrinsics from a set of images of a ChArUco board.

    :param board: The ChArUco board used for calibration.
    :param images_path: The path to the images containing the ChArUco board.
    :param fisheye: Whether to use the fisheye model for calibration. Otherwise, the pinhole model is used.
    :return: Instance of CameraIntrinsics containing:
    - The camera matrix K.
    - The distortion coefficients D.
    - Whether the fisheye model was used for calibration.
    - The size of images.
    - The RMS error of the calibration.
    """
    obj_points = []
    img_points = []

    image_size = None
    included_images = []

    for i, image_path in enumerate(tqdm(images_path)):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image_size is None:
            image_size = image.shape[:2][::-1]

        if image.shape[:2][::-1] != image_size:
            warn(
                f"Image {image_path} has a different size than the others. Skipping..."
            )
            continue

        ret, corners_obj_points, corners_img_points = detect_charuco_board(
            image, board, interactive_crop
        )

        if ret:
            obj_points.append(corners_obj_points)
            img_points.append(corners_img_points)
            included_images.append(i)
        else:
            warn(f"ChArUco board not found in image {image_path}. Skipping...")

    images_path = [images_path[i] for i in included_images]

    while len(obj_points) > 0:
        try:
            if use_fisheye_model:
                rms, K, D, _, _ = cv2.fisheye.calibrate(
                    objectPoints=obj_points,
                    imagePoints=img_points,
                    image_size=image_size,
                    K=None,
                    D=None,
                    flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                    + cv2.fisheye.CALIB_CHECK_COND
                    + cv2.fisheye.CALIB_FIX_SKEW,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30,
                        1e-6,
                    ),
                )
            else:
                rms, K, D, _, _ = cv2.calibrateCamera(
                    objectPoints=obj_points,
                    imagePoints=img_points,
                    imageSize=image_size,
                    cameraMatrix=None,
                    distCoeffs=None,
                    # flags=cv2.CALIB_CB_EXHAUSTIVE,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30,
                        1e-6,
                    ),
                )
            print(
                f"Computed camera intrinsics using {len(images_path)} images. RMS error = {rms:.2f}px."
            )
            return CameraIntrinsics(K, D, use_fisheye_model, image_size, rms)
        except cv2.error as err:
            print(err)
            idx = int(
                str(err).split()[-4]
            )  # Parse index of invalid image from error message
            obj_points = obj_points[:idx] + obj_points[idx + 1 :]
            img_points = img_points[:idx] + img_points[idx + 1 :]
            img_path = images_path[idx]
            images_path = images_path[:idx] + images_path[idx + 1 :]
            print(f"Removed ill-conditioned image {img_path} from the data.")

    raise Exception("There are no valid images from which to calibrate.")


def compute_extrinsics(
    image_path: str, board: cv2.aruco.CharucoBoard, camera_intrinsics: CameraIntrinsics
) -> CameraExtrinsics:
    """
    Compute the camera extrinsics from an image containing a ChArUco board.

    :param board: The ChArUco board used for calibration.
    :param image_path: The path to the image containing the ChArUco board.
    :param camera_intrinsics: The camera intrinsics.
    :return: Instance of CameraExtrinsics containing:
    - The rotation vector rvec in Rodrigues format (3x1 numpy array)
    - The translation vector tvec (3x1 numpy array)
    """
    image = cv2.imread(image_path)
    undistorted_img, new_K, new_D = undistort_img(image, camera_intrinsics, True)
    ret, obj_points, img_points = detect_charuco_board(undistorted_img, board)

    if not ret:
        raise ValueError("The ChArUco board was not found in the image.")

    _, rvec, tvec = cv2.solvePnP(
        obj_points, img_points, cameraMatrix=new_K, distCoeffs=new_D
    )
    return CameraExtrinsics(rvec, tvec)
