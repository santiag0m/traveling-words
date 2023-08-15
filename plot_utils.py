from typing import Union

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes


def find_closest_to_square(area: int) -> tuple[int, int]:
    square = area**0.5
    rows = int(square)
    while True:
        if area % rows == 0:
            cols = area // rows
            return (rows, cols)
        rows += 1


def plt_subplots_3d(
    nrows: int = 1,
    ncols: int = 1,
    sharex: bool = False,
    sharey: bool = False,
    sharez: bool = False,
    z_lim: tuple[int, int] = (0, 1),
    **kwargs,
) -> tuple[Figure, Union[np.ndarray, Axes]]:
    fig = plt.figure(**kwargs)

    axes = []
    shared_axis = None

    # If sharing axes, create an invisible primary axis
    if sharex or sharey or sharez:
        shared_axis = fig.add_subplot(111, projection="3d", frame_on=False)
        shared_axis.axis("off")
        if sharez:
            shared_axis.set_zlim(*z_lim)
    for i in range(nrows * ncols):
        shared_kwargs = {}
        if sharex:
            shared_kwargs["sharex"] = shared_axis
        if sharey:
            shared_kwargs["sharey"] = shared_axis
        if sharez:
            shared_kwargs["sharez"] = shared_axis

        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d", **shared_kwargs)
        axes.append(ax)

    # Reshape the axes to a 2D grid if multiple rows and columns
    if nrows > 1 or ncols > 1:
        axes = np.array(axes).reshape(nrows, ncols)

    plt.tight_layout()
    return fig, axes
