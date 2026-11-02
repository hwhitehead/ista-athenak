import sys, os, re
import numpy as np

pysrc = os.path.join(os.path.dirname(__file__), "/Users/hwhitehead/github/ista-athenak/pysrc")
if pysrc not in sys.path: sys.path.append(pysrc)
from ista_classes import NBody, InitBinary

def build_athinput(): 

if __name__ == "__main__":

    # define initial conditions for TDE athinput

    # construct dictionary of input values
    athinput_dict = {}

    # binary properties
    athinput_dict['a'] = 1.0             # semi-major axis of binary
    athinput_dict['q'] = 0.1             # mass ratio of binary
    athinput_dict['e'] = 0.99            # eccentricity of binary
    athinput_dict['theta_in_pi'] = 0.25  # initial true anomaly (from apoapsis, in units pi)
    athinput_dict['prograde'] = 1

    # define physics settings
    athinput_dict[""]