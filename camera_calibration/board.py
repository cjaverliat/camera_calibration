import cv2
import numpy as np


def create_charuco_board(size: cv2.typing.Size, square_length: float, marker_length: float, dict: int, legacy: bool) -> cv2.aruco.CharucoBoard:
    """
    Create a ChArUco board.

    :param size: The size of the board (large side then short side).
    :param square_length: The length of a square on the board in meters.
    :param marker_length: The length of a marker on the board in meters.
    :param dict: The dictionary to use for the markers.
    :param legacy: Whether to use the legacy pattern (for patterns created with opencv versions prior 4.6.0).
    """
    charuco_board = cv2.aruco.CharucoBoard(size, square_length, marker_length, cv2.aruco.getPredefinedDictionary(dict))
    charuco_board.setLegacyPattern(legacy)
    return charuco_board


def get_board_corners_count(board: cv2.aruco.CharucoBoard) -> int:
    """
    Get the number of corners on the board.

    :param board: The ChArUco board.
    """
    n_cols = board.getChessboardSize()[0]
    n_rows = board.getChessboardSize()[1]
    return (n_cols - 1) * (n_rows - 1)
