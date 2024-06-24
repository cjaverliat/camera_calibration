import numpy as np


class Camera:
    def __init__(
        self,
        name: str,
        resolution: tuple[int, int],
        K: np.ndarray = np.eye(4),
        d: np.ndarray = np.zeros((5, 1)),
        w2c: np.ndarray = np.eye(4),
    ):
        self.name = name
        self.resolution = resolution
        self.K = K
        self.d = d
        self.w2c = w2c
