import sys, os, re
import numpy as np
from shutil import copyfile
from copy import deepcopy

pysrc = os.path.join(os.path.dirname(__file__), "/Users/hwhitehead/github/ista-athenak/pysrc")
if pysrc not in sys.path: sys.path.append(pysrc)
from ista_classes import NBody, InitBinary

def build_athinput(athinput_dict, output_file, template_file): 

    copyfile(template_file, output_file)
    # open as read for replacement
    with open(output_file, "r") as file:
        file_data = file.read()
        # write mesh format
        file_data = file_data.replace("RUN_LABEL_VAL", str(run_label))



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

    # define path to input template and output
    template_file = os.path.join(os.path.dirname(__file__), "../inputs/ista/tde_template.athinput"