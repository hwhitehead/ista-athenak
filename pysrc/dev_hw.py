import sys, os, re
import numpy as np
import matplotlib.pyplot as plt

vis_python = os.path.join(os.path.dirname(__file__), '../vis/python')
if vis_python not in sys.path: sys.path.append(vis_python)
import athena_read

from ista_classes import NBody

if __name__ == "__main__":

    os.chdir("/Users/hwhitehead/wdir/athenak_wdir")
    hist_str = "nbody_long_large/nbody.user.hst"
    nbody = NBody(hist_str)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect("equal")
    ax.plot(nbody.t, nbody.diff(0,1,"x"))
    fig.savefig("diff_test.png", dpi=300, bbox_inches="tight")
    plt.close("all")