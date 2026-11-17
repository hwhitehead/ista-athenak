import numpy as np
import matplotlib.pyplot as plt

vis_python = os.path.join(os.path.dirname(__file__), "../vis")
if vis_python not in sys.path: sys.path.append(vis_python)
import athena_read

from ista_classes import HydroDataAthenaK, NBody, InitBinary, InitParabola, InitOrbit