import matplotlib.pyplot as plt
import numpy as np


class ImageWarper:
    def __init__(self, img):
        self.warp_corners = np.empty((0, 2))
        self.selected_point_idx = None
        self.img = img

    def show(self):
        self.fig, self.ax = plt.subplots()

        if self.img.shape[-1] == 3:
            self.ax.imshow(self.img)
        else:
            self.ax.imshow(self.img, cmap="gray")

        self._update()
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        plt.show()
        plt.close()

    def _update(self):

        # Save the current zoom state
        zoom_state = self.ax.get_xlim(), self.ax.get_ylim()

        self.ax.clear()

        if self.img.shape[-1] == 3:
            self.ax.imshow(self.img)
        else:
            self.ax.imshow(self.img, cmap="gray")

        self.ax.set_title("Click to add points, drag to move, right-click to delete")

        # Draw a line between each pair of points
        for idx in range(len(self.warp_corners)):
            point = self.warp_corners[idx]
            next_idx = (idx + 1) % len(self.warp_corners)

            self.ax.plot(
                [self.warp_corners[idx][0], self.warp_corners[next_idx][0]],
                [self.warp_corners[idx][1], self.warp_corners[next_idx][1]],
                "b-",
            )

            self.ax.scatter(point[0], point[1], c="r", marker="o", s=5)

        # Restore the zoom state
        self.ax.set_xlim(zoom_state[0])
        self.ax.set_ylim(zoom_state[1])
        self.fig.canvas.draw()

    def _on_click(self, event):

        # Check that no specific tool is selected
        if plt.get_current_fig_manager().toolbar.mode != "":
            return

        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Left click
            self.selected_point_idx = self.get_nearest_point_index(
                event.xdata, event.ydata
            )
            if self.selected_point_idx is None and len(self.warp_corners) < 4:
                self.warp_corners = np.append(
                    self.warp_corners, [[event.xdata, event.ydata]], axis=0
                )
            self._update()

        elif event.button == 3:  # Right click
            nearest_point_idx = self.get_nearest_point_index(event.xdata, event.ydata)
            if nearest_point_idx is not None:
                self.warp_corners = np.delete(
                    self.warp_corners, nearest_point_idx, axis=0
                )
            self._update()

    def _on_motion(self, event):
        if event.inaxes != self.ax or self.selected_point_idx is None:
            return
        if event.button == 1:  # Left click and drag
            self.warp_corners[self.selected_point_idx] = (
                event.xdata,
                event.ydata,
            )
            self._update()

    def _on_release(self, event):
        self.selected_point_idx = None

    def get_nearest_point_index(self, x, y):
        if len(self.warp_corners) == 0:
            return None
        points_array = np.array(self.warp_corners)
        distances = np.sqrt(
            (points_array[:, 0] - x) ** 2 + (points_array[:, 1] - y) ** 2
        )
        nearest_index = np.argmin(distances)
        if distances[nearest_index] < 10:
            return nearest_index
        return None
