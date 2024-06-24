import cv2
import numpy as np
from camera_calibration.utils.image_warper import ImageWarper


def _interactive_warp_crop(
    img: cv2.typing.MatLike, dst_size: tuple[int, int]
) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    img_warper = ImageWarper(img)
    img_warper.show()

    dst_corners = np.array(
        [[0, 0], [dst_size[0], 0], [dst_size[0], dst_size[1]], [0, dst_size[1]]],
        dtype=np.float32,
    )

    H, _ = cv2.findHomography(img_warper.warp_corners, dst_corners)
    warped_img = cv2.warpPerspective(img, H, dst_size, flags=cv2.INTER_LINEAR)
    return H, warped_img


def _find_and_refine_chessboard_corners(
    img: cv2.typing.MatLike,
    pattern_size: tuple[int, int],
    flags=cv2.CALIB_CB_FAST_CHECK
    + cv2.CALIB_CB_ADAPTIVE_THRESH
    + cv2.CALIB_CB_NORMALIZE_IMAGE,
    subpix_win_size=(3, 3),
    subpix_zero_zone=(-1, -1),
    subpix_criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
) -> tuple[bool, cv2.typing.MatLike | None]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)

    if ret and corners is not None:
        corners = cv2.cornerSubPix(
            gray,
            corners,
            winSize=subpix_win_size,
            zeroZone=subpix_zero_zone,
            criteria=subpix_criteria,
        )
        return True, corners

    return False, None


def find_chessboard_corners(
    img: cv2.typing.MatLike,
    pattern_size: tuple[int, int],
    allow_interactive_warp_crop=True,
    flags=cv2.CALIB_CB_FAST_CHECK
    + cv2.CALIB_CB_ADAPTIVE_THRESH
    + cv2.CALIB_CB_NORMALIZE_IMAGE,
    subpix_win_size=(3, 3),
    subpix_zero_zone=(-1, -1),
    subpix_criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
) -> tuple[bool, cv2.typing.MatLike | None]:
    ret, corners = _find_and_refine_chessboard_corners(img, pattern_size)

    if ret and corners is not None:
        return ret, corners

    if not allow_interactive_warp_crop:
        return False, None

    aspect_ratio = pattern_size[0] / pattern_size[1]
    dst_size = (720, int(720 / aspect_ratio))
    H, warped_img = _interactive_warp_crop(img, dst_size)

    ret, corners = _find_and_refine_chessboard_corners(
        warped_img,
        pattern_size,
        flags=flags,
        subpix_win_size=subpix_win_size,
        subpix_zero_zone=subpix_zero_zone,
        subpix_criteria=subpix_criteria,
    )

    if ret and corners is not None:
        # Unwarp the corners back to the original image space
        corners = cv2.perspectiveTransform(corners, np.linalg.inv(H))
        return ret, corners

    return False, None
