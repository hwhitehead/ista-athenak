import sys, os, re
import numpy as np
import matplotlib.pyplot as plt

vis_python = os.path.join(os.path.dirname(__file__), '../vis/python')
if vis_python not in sys.path: sys.path.append(vis_python)
import athena_read

from ista_classes import NBody

if __name__ == "__main__":

    os.chdir("/Users/hwhitehead/wdir/athenak_wdir")
    hist_str = "nbody_acc/nbody.user.hst"
    nbody = NBody(hist_str)

    fig = plt.figure(figsize=(10.0 / 3, 5.0 /3))
    ax = fig.add_subplot()
    r = nbody.calc_diff("r")
    ax.plot(nbody.t, r)
    fig.savefig("diff_test.png", dpi=300, bbox_inches="tight")
    plt.close("all")