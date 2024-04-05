import numpy as np
import cv2
import datetime
import glob
import os
from tqdm import tqdm

def create_charuco_board():
    charuco_board = cv2.aruco.CharucoBoard((11, 8), 0.035, 0.027, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100))
    charuco_board.setLegacyPattern(True)
    return charuco_board

def calibrate(name: str, board: cv2.aruco.CharucoBoard, frames_dir: str):
    dict = board.getDictionary()
    n_cols = board.getChessboardSize()[0]
    n_rows = board.getChessboardSize()[1]
    n_corners = (n_cols - 1) * (n_rows - 1)

    objp = np.zeros((n_corners, 1, 3), np.float32)
    objp[:,0,:2] = (np.mgrid[0:10, 0:7].T.reshape(-1, 2) + np.array([1, 1], dtype=np.float32)) * 0.035

    objpoints = np.empty((0, n_corners, 1, 3), np.float32) # 3d point in real world space
    imgpoints = np.empty((0, n_corners, 1, 2), np.float32) # 2d points in image plane.

    frames_path = glob.glob(os.path.join(frames_dir, "*.png"))

    for frame_path in tqdm(frames_path):
        frame = cv2.imread(frame_path, flags=cv2.IMREAD_GRAYSCALE)

        # Detect markers in the undistorted image
        corners, ids, rejected_corners = cv2.aruco.detectMarkers(frame, dict)
        # If at least one marker is detected
        if ids is not None:
            corners, ids, rejected_corners, _ = cv2.aruco.refineDetectedMarkers(frame, board, corners, ids, rejected_corners)
            
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, frame, board)
            if charuco_retval and len(charuco_corners) == n_corners:
                objpoints = np.append(objpoints, [objp], axis=0)
                imgpoints = np.append(imgpoints, [charuco_corners], axis=0)

    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    
    while True:
        assert len(objpoints) > 0, "There are no valid images from which to calibrate."
        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                objectPoints=objpoints,
                imagePoints=imgpoints,
                image_size=frame.shape[::-1],
                K=None,
                D=None,
                flags=calibration_flags,
                criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
            return K, D, rms
        except cv2.error as err:
            try:
                idx = int(str(err).split()[-4])  # Parse index of invalid image from error message
                objpoints = np.delete(objpoints, idx, axis=0)
                imgpoints = np.delete(imgpoints, idx, axis=0)
                print("Removed ill-conditioned image {} from the data. Trying again...".format(idx))
            except IndexError:
                raise err

def save_calibration_results(name, K, D, rms):
    np.save(f"output/{name}_K.npy", K)
    np.save(f"output/{name}_D.npy", D)
    with open(f"output/{name}_rms.txt", "w") as f:
        f.write(str(rms))

if __name__ == "__main__":

    board = create_charuco_board()

    for gopro_name in ["gopro1", "gopro2", "gopro3", "gopro4", "gopro5", "gopro6"]:
        print(f"Calibrating {gopro_name}...")
        frames_dir = f"D:/Charles_JAVERLIAT/Calibration GoPro/{gopro_name}_anonymized/"
        K, D, rms = calibrate(gopro_name, board, frames_dir)
        save_calibration_results(gopro_name, K, D, rms)
        print(f"RMS error for {gopro_name}:", rms)