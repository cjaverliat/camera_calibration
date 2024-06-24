from camera_calibration.camera import Camera
from camera_calibration.board import Board
from camera_calibration.chessboard_detection import find_chessboard_corners
import cv2
import numpy as np
from tqdm import tqdm


class CameraGraph:
    """
    A class representing a graph where each node is a camera and each edge is a transformation matrix between two cameras.

    The graph also has a camera node marked as the reference camera. It is used to compute the world to camera matrix for any other camera in the scene
    to go from camera space to world space.
    """

    def __init__(self):
        self.nodes = set[Camera]()
        self.edges = dict[Camera, dict[Camera, np.ndarray]]()
        self.ref_camera = None
        self.ref_w2c = None

    def add_node(self, camera: Camera):
        self.nodes.add(camera)

    def add_edge(self, c1: Camera, c2: Camera, c1_to_c2_mtx: np.ndarray):
        if c1 not in self.nodes:
            raise ValueError(f"Camera {c1} not in the scene")
        if c2 not in self.nodes:
            raise ValueError(f"Camera {c2} not in the scene")

        self.edges.setdefault(c1, dict())[c2] = c1_to_c2_mtx
        self.edges.setdefault(c2, dict())[c1] = np.linalg.inv(c1_to_c2_mtx)

    def set_reference_camera(self, camera: Camera, w2c: np.ndarray = np.eye(4)):
        if camera not in self.nodes:
            raise ValueError(f"Camera {camera} not in the scene")

        self.ref_camera = camera
        # If no world to camera matrix is provided, assume the reference camera is at the origin.
        self.ref_w2c = w2c

    def get_reference_camera(self):
        if len(self.nodes) == 0:
            return None

        if self.ref_camera is None:
            return self.nodes[0]

        return self.ref_camera

    def get_world_to_camera_mtx(self, camera: Camera):
        if camera not in self.nodes:
            raise ValueError(f"Camera {camera} not in the scene")

        if camera == self.ref_camera:
            return self.ref_w2c

        return self.get_camera_x_to_camera_y_mtx(camera, self.ref_camera) @ self.ref_w2c

    def get_camera_to_world_mtx(self, camera: Camera):
        return np.linalg.inv(self.get_world_to_camera_mtx(camera))

    def get_camera_x_to_camera_y_mtx(self, cx: Camera, cy: Camera):
        """
        Recursively compute the transformation matrix between two cameras.
        :param cx: The source camera.
        :param cy: The destination camera.
        :return: The transformation matrix from cx to cy.
        """
        if cx == cy:
            return np.eye(4)

        if cx not in self.nodes:
            raise ValueError(f"Camera {cx} not in the scene")
        if cy not in self.nodes:
            raise ValueError(f"Camera {cy} not in the scene")

        if cx not in self.edges:
            raise ValueError(f"{cx} is disconnected from the rest of the scene graph")
        if cy not in self.edges:
            raise ValueError(f"{cy} is disconnected from the rest of the scene graph")

        if cy in self.edges[cx]:
            return self.edges[cx][cy]

        path = self._get_path(cx, cy)

        if path is None:
            raise ValueError(f"No path found between {cx} and {cy}")

        transform = np.eye(4)

        for i in range(len(path) - 1):
            transform = transform @ self.edges[path[i]][path[i + 1]]

        # Cache the result for future calls.
        self.edges[cx][cy] = transform
        self.edges[cy][cx] = np.linalg.inv(transform)

        return transform

    def _get_path(self, c1: Camera, c2: Camera):
        """
        Find a path between two cameras using a depth-first search.
        :param c1: The source camera.
        :param c2: The destination camera.
        :return: A list of cameras representing one possible path between c1 and c2.
        """

        visited = set()

        def dfs(current, path):
            if current == c2:
                return path

            visited.add(current)
            for neighbor in self.edges[current]:
                if neighbor not in visited:
                    result = dfs(neighbor, path + [neighbor])
                    if result:
                        return result
            return None

        return dfs(c1, [c1])


def compute_world_to_camera_mtx_from_img(
    img: cv2.typing.MatLike, board: Board, K: cv2.typing.MatLike, d: cv2.typing.MatLike
):
    ret, corners = find_chessboard_corners(img, board.pattern_size)

    if ret and corners is not None:
        _, rvec, tvec = cv2.solvePnP(board.points, corners, K, d)
        R = cv2.Rodrigues(rvec)[0]
        w2c = np.hstack([R, tvec])
        w2c = np.vstack([w2c, np.array([0, 0, 0, 1])])
        return w2c

    raise ValueError("Chessboard not found in image.")


def calibrate_cameras_extrinsics(
    reference: tuple[Camera, cv2.typing.MatLike],
    pairs: list[tuple[Camera, cv2.typing.MatLike, Camera, cv2.typing.MatLike]],
    board: Board,
) -> tuple[list[Camera], CameraGraph]:

    graph = CameraGraph()

    print("Creating camera graph...")

    for c1, img1, c2, img2 in tqdm(pairs):

        w2c_cam1 = compute_world_to_camera_mtx_from_img(img1, board, c1.K, c2.d)
        w2c_cam2 = compute_world_to_camera_mtx_from_img(img2, board, c2.K, c2.d)
        c1_to_c2_mtx = w2c_cam1 @ np.linalg.inv(w2c_cam2)

        if c1 not in graph.nodes:
            graph.add_node(c1)
        if c2 not in graph.nodes:
            graph.add_node(c2)
        graph.add_edge(c1, c2, c1_to_c2_mtx)

    print("Computing reference camera transformation matrix...")

    ref_cam = reference[0]
    ref_img = reference[1]
    ref_w2c = compute_world_to_camera_mtx_from_img(ref_img, board, ref_cam.K, ref_cam.d)
    graph.set_reference_camera(ref_cam, ref_w2c)

    print("Computing camera matrices...")

    for c in tqdm(graph.nodes):
        c.w2c = graph.get_world_to_camera_mtx(c)

    return graph.nodes, graph
