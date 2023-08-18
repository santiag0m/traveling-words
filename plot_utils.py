from math import ceil
from typing import Union

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes


def find_closest_to_square(area: int) -> tuple[int, int]:
    square = area**0.5
    cols = ceil(square)
    while True:
        if area % cols == 0:
            rows = area // cols
            return (rows, cols)
        cols += 1


def plt_subplots_3d(
    nrows: int = 1,
    ncols: int = 1,
    sharex: bool = False,
    sharey: bool = False,
    sharez: bool = False,
    **kwargs,
) -> tuple[Figure, Union[Axes, np.ndarray]]:
    fig = plt.figure(**kwargs)
    axes = []

    shared_axis = None
    if sharex or sharey or sharez:
        shared_axis = fig.add_subplot(111, projection="3d", frame_on=False)
        shared_axis.axis("off")

    for i in range(nrows * ncols):
        shared_kwargs = {}
        if sharex:
            shared_kwargs["sharex"] = shared_axis
        if sharey:
            shared_kwargs["sharey"] = shared_axis

        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d", **shared_kwargs)
        axes.append(ax)

    button_press = False

    def on_move(event):
        if event.inaxes is None:
            return

        for ax in np.ravel(axes):
            if event.inaxes == ax:
                continue

            # Synchronize rotation
            ax.view_init(elev=event.inaxes.elev, azim=event.inaxes.azim)

            # Synchronize zoom and pan
            ax.set_xlim(event.inaxes.get_xlim())
            ax.set_ylim(event.inaxes.get_ylim())
            if sharez and button_press:
                ax.set_zlim(event.inaxes.get_zlim())

        fig.canvas.draw_idle()

    def on_scroll(event):
        if event.inaxes is None:
            return

        for ax in np.ravel(axes):
            if event.inaxes == ax:
                continue

            # Synchronize rotation
            ax.view_init(elev=event.inaxes.elev, azim=event.inaxes.azim)

            # Synchronize zoom and pan
            ax.set_xlim(event.inaxes.get_xlim())
            ax.set_ylim(event.inaxes.get_ylim())
            if sharez:  # No check for button
                ax.set_zlim(event.inaxes.get_zlim())

        fig.canvas.draw_idle()

    def button_on(event):
        nonlocal button_press
        button_press = True

    def button_off(event):
        nonlocal button_press
        button_press = False

    # Connect the event handlers to the figure
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", button_on)
    fig.canvas.mpl_connect("button_release_event", button_off)

    if nrows > 1 or ncols > 1:
        axes = np.array(axes).reshape(nrows, ncols)

    plt.tight_layout()
    return fig, axes
