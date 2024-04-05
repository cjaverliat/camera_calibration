import numpy as np
import cv2
from typing import Tuple

def undistort_image_fisheye(img: cv2.typing.MatLike, K: cv2.typing.MatLike, D: cv2.typing.MatLike, crop=True) -> Tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    img_size = img.shape[:2][::-1]

    # TODO: if img is not the same size as the images used for calibration, we need to rescale K

    # Change balance to 0 to crop and remove black borders
    balance = int(not crop)

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, img_size, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, img_size, cv2.CV_16SC2)
    undistorted_image = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_image, new_K

if __name__ == "__main__":
    image_path = "input/calibration_test.png"
    gopro_name = "gopro1"
    K = np.load(f"output/{gopro_name}_K.npy")
    D = np.load(f"output/{gopro_name}_D.npy")
    undistorted_image, new_K = undistort_image_fisheye(image_path, K, D)
    cv2.imshow("undistorted", undistorted_image)
    cv2.waitKey(0)