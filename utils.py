from matplotlib import pyplot as plt
import cv2
import numpy as np
from typing import Tuple

def interactive_image_crop(img: cv2.typing.MatLike) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Crop an image interactively by selecting a region of interest.

    :param img: The image to crop.
    :return: The x and y limits of the cropped image.
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    fig.canvas.mpl_connect('key_press_event', lambda _, fig=fig: plt.close(fig))
    plt.show()

    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    xlim_min = min(int(xlim[0]), int(xlim[1]))
    xlim_max = max(int(xlim[0]), int(xlim[1]))
    ylim_min = min(int(ylim[0]), int(ylim[1]))
    ylim_max = max(int(ylim[0]), int(ylim[1]))
    return (xlim_min, xlim_max), (ylim_min, ylim_max)
