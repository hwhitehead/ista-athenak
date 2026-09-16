import sys, os
import numpy
import matplotlib.pyplot as plt

vis_python = os.path.join(os.path.dirname(__file__), '../vis/python')
if vis_python not in sys.path: sys.path.append(vis_python)

import athena_read

def plot_hst(load_str, save_str):

    data = athena_read.hst(load_str)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect("equal")

    t = data["time"]
    for s in range(2):
        x = data["x" + str(s)]
        y = data["y" + str(s)]
        ax.plot(x, y)

    ax.set_xlim([-0.05, 0.05])
    ax.set_ylim([-0.05, 0.05])

    plt.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(save_str, dpi=300, bbox_inches="tight")
    plt.close("all")
    

if __name__ == "__main__":

    os.chdir("/Users/hwhitehead/wdir/athenak_wdir/")
    plot_hst("tde.user.hst","hst_plot.png")