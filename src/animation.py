# -*- coding: UTF-8 -*-
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def saveGazeAnimation(x, y, file_name):
    """
    x, y: gaze posisions:
    file_name: e.g. "video.mp4"
    you may need to install ffmepg in your computer.
    """
    fig, ax = plt.subplots()
    ax.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
    m = ax.plot([], [], 'o', color='black', markersize=3)
    draw_points_n = 30

    def animate(i):
        m[0].set_data(x[i-draw_points_n:i],
                      y[i-draw_points_n:i])
        # m[0].set_color(c[i-draw_points_n:i])
    gaze_animation = animation.FuncAnimation(
        fig, animate, frames=len(x), interval=1)
    # plt.show()
    gaze_animation.save(file_name)
