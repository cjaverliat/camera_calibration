import numpy as np


class Board:
    def __init__(self, pattern_size: tuple[int, int], square_size: float):
        self.pattern_size = pattern_size
        self.square_size = square_size

        self.points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.points[:, :2] = np.mgrid[
            0 : pattern_size[0], 0 : pattern_size[1]
        ].T.reshape(-1, 2)
        self.points *= square_size
