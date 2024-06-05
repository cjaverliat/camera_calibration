from camera_calibration.board import create_charuco_board
import cv2

import numpy as np
from glob import glob
from camera_calibration.intrinsics import compute_intrinsics
from camera_calibration.board import create_charuco_board
import argparse
import os

from camera_calibration.board import create_charuco_board

board = create_charuco_board(
    size=(11, 8),
    square_length=0.035,
    marker_length=0.027,
    dict=cv2.aruco.DICT_4X4_100,
    legacy=True,
)


def main(args):

    if os.path.exists(args.output_file):
        if not args.overwrite:
            response = input(
                f"{args.output_file} already exists. Do you want to overwrite it? [y/N] "
            )
            if response.lower() != "y":
                print("Exiting...")
                return

    parent_dir = os.path.dirname(args.output_file)

    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    board = create_charuco_board(
        size=args.board_size,
        square_length=args.square_length,
        marker_length=args.marker_length,
        dict=args.markers_dictionary,
        legacy=args.legacy_board,
    )
    intrinsics = compute_intrinsics(
        args.images,
        board,
        interactive_crop=True,
        use_fisheye_model=args.fisheye,
    )

    output = {
        "camera_matrix": intrinsics.K,
        "distortion_coefficients": intrinsics.D,
        "use_fisheye_model": intrinsics.fisheye_model,
        "image_size": intrinsics.image_size,
        "rms": intrinsics.rms,
    }

    np.save(args.output_file, output)
    print(f"Camera intrinsics saved to {args.output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--board-size",
        type=str,
        default="11x8",
        help="Size of the ChArUco board formatted as WxH.",
    )
    parser.add_argument(
        "--square-length",
        type=float,
        default=0.035,
        help="The length of the squares in meters.",
    )
    parser.add_argument(
        "--marker-length",
        type=float,
        default=0.027,
        help="The length of the markers in meters.",
    )
    parser.add_argument(
        "--markers-dictionary",
        type=int,
        default=cv2.aruco.DICT_4X4_100,
        help="The dictionary to use for the markers.",
    )
    parser.add_argument(
        "--legacy-board",
        action="store_true",
        default=False,
        help="Whether to use the legacy ChArUco board constructor.",
    )
    parser.add_argument(
        "--fisheye",
        action="store_true",
        default=False,
        help="Whether to use the fisheye model for calibration.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Whether to overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="output/camera_intrinsics.npy",
        help="The output file path for the camera intrinsics.",
    )
    parser.add_argument(
        "images",
        type=str,
        nargs="+",
        help="The image or glob pattern to search for calibration images.",
    )

    parser.epilog = """
Example:
    python calibrate_intrinsics.py --legacy-board --fisheye --board-size 11x8 --square-length 0.035 --marker-length 0.027 --markers-dictionary 1 --output-file output/camera_intrinsics.npy input/images/*.jpg
    
    Note: Here the markers dictionary is set to 1, which corresponds to cv2.aruco.DICT_4X4_100.
    """

    args = parser.parse_args()

    try:
        args.board_size = tuple(map(int, args.board_size.split("x")))
    except ValueError:
        raise ValueError("Invalid format for board size.")

    args.images = [m for m in args.images for m in glob(m)]
    main(args)
