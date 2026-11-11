import sys, os, re
import numpy as np
from shutil import copyfile
from copy import deepcopy

pysrc = os.path.join(os.path.dirname(__file__), "/Users/hwhitehead/github/ista-athenak/pysrc")
if pysrc not in sys.path: sys.path.append(pysrc)
from ista_classes import InitBinary

def build_athinput(IC_dict, output_file, template_file): 

    # TODO: add 3D support
    if (IC_dict["i_in_pi"] != 0.0):
        raise ValueError("3D support not yet implemented")

    # read IC_dict value, inform athinput values
    athinput_dict = {}

    # build binary
    binary = InitBinary(a=IC_dict["a"], q=IC_dict["q"], e=IC_dict["e"])
    init_nbody_state = binary.init(theta=IC_dict["theta_init_in_pi"]* np.pi, i=IC_dict["i_in_pi"] * np.pi, print_values=False)
    athinput_dict["M0"]  = init_nbody_state[0]
    athinput_dict["X0"]  = init_nbody_state[1]
    athinput_dict["Y0"]  = init_nbody_state[2]
    athinput_dict["Z0"]  = init_nbody_state[3]
    athinput_dict["VX0"] = init_nbody_state[4]
    athinput_dict["VY0"] = init_nbody_state[5]
    athinput_dict["VZ0"] = init_nbody_state[6]
    athinput_dict["M1"]  = init_nbody_state[7]
    athinput_dict["X1"]  = init_nbody_state[8]
    athinput_dict["Y1"]  = init_nbody_state[9]
    athinput_dict["Z1"]  = init_nbody_state[10]
    athinput_dict["VX1"] = init_nbody_state[11]
    athinput_dict["VY1"] = init_nbody_state[12]
    athinput_dict["VZ1"] = init_nbody_state[13]

    # determine timings by angular sweep of orbit
    f_init = np.pi + IC_dict["theta_init_in_pi"] * np.pi # switch to true anomaly defined from periapsis
    E_init = np.arctan2(np.sqrt(1 - IC_dict["e"] ** 2) * np.sin(f_init), IC_dict["e"] + np.cos(f_init))
    if IC_dict["theta_exit_in_pi"] is not None:
        f_exit = np.pi + IC_dict["theta_exit_in_pi"] * np.pi
        E_exit = np.arctan2(np.sqrt(1 - IC_dict["e"] ** 2) * np.sin(f_exit), IC_dict["e"] + np.cos(f_exit))
    else:
        E_exit = - E_init # symmetric about pericenter
    m_bin = athinput_dict["M0"] + athinput_dict["M1"]
    mean_motion = np.sqrt(m_bin / IC_dict["a"] ** 3)
    t_init = (E_init - IC_dict["e"] * np.sin(E_init)) / mean_motion
    t_exit = (E_exit - IC_dict["e"] * np.sin(E_exit)) / mean_motion
    delta_t = t_exit - t_init
    if IC_dict["tlim"] is not None:
        athinput_dict["TLIM"]     = IC_dict["tlim"]
    else:
        athinput_dict["TLIM"]     = delta_t
    if IC_dict["hst_dt"] is not None:
        athinput_dict["HST_DT"]   = IC_dict["hst_dt"]
    else:
        athinput_dict["HST_DT"]   = delta_t / IC_dict["num_hst"]
    if IC_dict["hydro_dt"] is not None:
        athinput_dict["HYDRO_DT"] = IC_dict["hydro_dt"]
    else:
        athinput_dict["HYDRO_DT"] = delta_t / IC_dict["num_hydro"]

    # determine target resolution and domain size
    r_p = IC_dict["a"] * (1.0 - IC_dict["e"])
    dx_min = r_p / (IC_dict["cells_per_soft"] * IC_dict["soft_per_rp"])
    dx_root = dx_min * np.power(2, IC_dict["num_levels"] - 1)
    Lx = 2 * IC_dict["domain_scale"] * athinput_dict["X1"]
    Ly = 2 * IC_dict["domain_scale"] * athinput_dict["Y1"]
    xlims = [-0.5 * Lx, 0.5 * Lx]
    ylims = [-0.5 * Ly, 0.5 * Ly]

    # resize domain to fit exact multiple of meshblock size
    nx = int((xlims[1] - xlims[0]) / dx_root) 
    ny = int((ylims[1] - ylims[0]) / dx_root)
    nx = IC_dict["cells_per_mb"] * np.ceil(nx / IC_dict["cells_per_mb"]) # recast to fit exact count of meshblocks
    ny = IC_dict["cells_per_mb"] * np.ceil(ny / IC_dict["cells_per_mb"]) # recast to fit exact count of meshblocks
    Lx = nx * dx_root
    Ly = ny * dx_root
    xlims = [-0.5 * Lx, 0.5 * Lx]
    ylims = [-0.5 * Ly, 0.5 * Ly]

    # specify mesh properties
    athinput_dict["NX1"] = int(nx)
    athinput_dict["NX2"] = int(ny)
    athinput_dict["X1MIN"] = xlims[0]
    athinput_dict["X1MAX"] = xlims[1]
    athinput_dict["X2MIN"] = ylims[0]
    athinput_dict["X2MAX"] = ylims[1]

    # specify mesh refinement properties
    athinput_dict["NUM_LEVELS"]       = IC_dict["num_levels"]
    athinput_dict["MAX_NMB_PER_RANK"] = IC_dict["max_nmb_per_rank"]

    # specify disc properties and physics modules
    athinput_dict["MACH"]         = IC_dict["Mach"]
    athinput_dict["ALPHA"]        = IC_dict["alpha"]
    athinput_dict["NU_ISO"]       = IC_dict["nu_iso"]
    athinput_dict["GRAV_MODE"]    = IC_dict["grav_mode"]
    athinput_dict["COOLING_MODE"] = IC_dict["cooling_mode"]
    athinput_dict["SINK_MODE"]    = IC_dict["sink_mode"]
    athinput_dict["VISC_MODE"]    = IC_dict["visc_mode"]
    athinput_dict["SOFT_MODE"]    = IC_dict["soft_mode"]

    # specify non-binary nbody properties
    r_H_p = r_p * np.cbrt(IC_dict["q"] / (3 * (1 + IC_dict["q"]))) # Hill radius at periapsis
    for n in range(2):
        # determine disk size
        if IC_dict["r_disk{0}".format(n)] is not None:
            r_disk = IC_dict["r_disk{0}".format(n)]  
        else:
            r_disk = IC_dict["beta{0}".format(n)] * r_H_p
        # determine amr radius
        if IC_dict["r_amr{0}".format(n)] is not None:
            r_amr = IC_dict["r_amr{0}".format(n)]
        else:
            r_amr = 0.5 * r_disk
        athinput_dict["R_SOFT{0}".format(n)] = r_p / IC_dict["soft_per_rp"]
        athinput_dict["GRADE{0}".format(n)]  = IC_dict["grade{0}".format(n)]
        athinput_dict["R_CAV{0}".format(n)]  = IC_dict["r_cav{0}".format(n)]        
        athinput_dict["R_AMR{0}".format(n)]  = r_amr
        athinput_dict["R_DISK{0}".format(n)] = r_disk
        athinput_dict["T_AMR{0}".format(n)]  = IC_dict["t_amr{0}".format(n)]

    # reset amr value to ensure no-disk systems are resolve
    m = 1
    for n in range(2):
        if athinput_dict["R_AMR{0}".format(n)] == 0:    
            athinput_dict["R_AMR{0}".format(n)] = athinput_dict["R_AMR{0}".format(m)]
        m -= 1

    copyfile(template_file, output_file)
    # open as read for replacement
    with open(output_file, "r") as file:
        file_data = file.read()
        # write dictioniary values to template
        for key, value in athinput_dict.items():
            file_data = file_data.replace("@{0}@".format(key), str(value))

    # open as write to save
    with open(output_file, "w") as file:
        file.write(file_data)

    # open as write to append SMR states as unique blocks
    smr_region_label = 1
    with open(output_file, "a") as file:
        for n in range(2):
            smr_radius = 0.5 * athinput_dict["R_AMR{0}".format(n)] # match AMR region
            if smr_radius == 0: continue
            smr_block = "\n"
            smr_block += "<refined_region{0}>\n".format(smr_region_label)
            smr_block += "x1min = {0}\n".format(athinput_dict["X{0}".format(n)] - smr_radius)
            smr_block += "x1max = {0}\n".format(athinput_dict["X{0}".format(n)] + smr_radius)
            smr_block += "x2min = {0}\n".format(athinput_dict["Y{0}".format(n)] - smr_radius)
            smr_block += "x2max = {0}\n".format(athinput_dict["Y{0}".format(n)] + smr_radius)
            smr_block += "x3min = -0.05\n"
            smr_block += "x3max = 0.05\n"
            smr_block += "level = {0}\n".format(athinput_dict["NUM_LEVELS"] - 1)
            file.write(smr_block)
            smr_region_label += 1

    # add comment block at end with explicit contents of IC_dict for reproducibility
    with open(output_file, "a") as file:
        comment_block = "\n"
        comment_block += "<comment>\n"
        for key, value in IC_dict.items():
            comment_block += "{0} = {1}\n".format(key, str(value))
        file.write(comment_block)

    print("finished athinput construction, saved output at {0}".format(output_file))



if __name__ == "__main__":

    # define initial conditions for TDE athinput

    # construct dictionary of input values
    IC_dict = {}

    # binary properties
    IC_dict['a']                = 1.0     # semi-major axis of binary
    IC_dict['q']                = 0.1     # mass ratio of binary
    IC_dict['e']                = 0.9    # eccentricity of binary
    IC_dict["i_in_pi"]          = 0.0     # inclination of binary  (in units pi)
    IC_dict['theta_init_in_pi'] = 0.125   # initial true anomaly (from apoapsis, in units pi)
    IC_dict["theta_exit_in_pi"] = None    # if None, symmetrise about pericenter

    # set timing based on orbit
    IC_dict["num_hydro"] = 200
    IC_dict["num_hst"]   = 1e4
    IC_dict['tlim']      = None   
    IC_dict['hst_dt']    = None
    IC_dict['hydro_dt']  = None

    # define physics settings
    IC_dict["grav_mode"]    = 2 
    IC_dict["cooling_mode"] = 2
    IC_dict["sink_mode"]    = 2
    IC_dict["visc_mode"]    = 1
    IC_dict["grav_mode"]    = 2
    IC_dict["soft_mode"]    = 0

    # non-binary nbody properties
    IC_dict["grade0"]  = 1
    IC_dict["grade1"]  = 1
    IC_dict["r_cav0"]  = 0.0
    IC_dict["r_cav1"]  = 0.0
    IC_dict["r_disk0"] = None # set with scaling to periapsis?
    IC_dict["r_disk1"] = None
    IC_dict["beta0"]   = 0    # TDE-like impact parameter = r_t / r_p
    IC_dict["beta1"]   = 3
    IC_dict["r_amr0"]  = None # auto set as half-disc width
    IC_dict["r_amr1"]  = None # auto set as half-disc width
    
    IC_dict["t_amr0"]  = 0.0 # TODO: make adaptive for single disc flyby
    IC_dict["t_amr1"]  = 0.0

    # set disk properties
    IC_dict["Mach"]   = 50.0
    IC_dict["alpha"]  = 0.1
    IC_dict["nu_iso"] = 1e-8 # set to non-zero to allow for adaptive viscosity

    # set resolution and softening
    IC_dict["cells_per_soft"]   = 16 # 10
    IC_dict["soft_per_rp"]      = 16 # 20
    IC_dict["num_levels"]       = 7
    IC_dict["max_nmb_per_rank"] = 2048
    IC_dict["domain_scale"]     = 2.5           # target domain this scale larger than initial binary size
    IC_dict["three_d"]          = False              # currently inactive
    IC_dict["cells_per_mb"]     = 16

    # define path to input template and output
    template_file = os.path.join(os.path.dirname(__file__), "../inputs/ista/tde_template.athinput")
    output_file   = os.path.join(os.path.dirname(__file__), "../inputs/ista/tde_output.athinput")

    # invoke builder
    build_athinput(IC_dict, output_file, template_file)