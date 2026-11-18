import numpy as np
import matplotlib.pyplot as plt
import subprocess, os, sys
import shutil

vis_python = os.path.join(os.path.dirname(__file__), "../vis/python")
if vis_python not in sys.path: sys.path.append(vis_python)
import athena_read

from ista_classes import HydroDataAthenaK, NBody, InitBinary, InitParabola, InitOrbit

def mass_convert_athdf(master_dir, prod_params):

    grade_span    = prod_params["grade_span"]
    beta_span     = prod_params["beta_span"]
    ecc_span      = prod_params["ecc_span"]
    q             = prod_params["q_span"]

    for grade in grade_span:
        for beta in beta_span:
            for ecc in ecc_span:
                # define save_str
                run_name = "tde_{0}q{1}b{2}e{3}".format(grade, q, beta, ecc)
                run_dir = os.path.join(master_dir, run_name)
                bin_stem = os.path.join(run_dir, "bin/nbody.hydro_w")
                # invoke subproc to call binary converter
                subprocess.run(["python3", convert_path, bin_stem])
                print("finished .athdf creation for {0}".format(run_name))

def mass_track_mass(master_dir, prod_params, skip=1):

    grade_span    = prod_params["grade_span"]
    beta_span     = prod_params["beta_span"]
    ecc_span      = prod_params["ecc_span"]
    q             = prod_params["q_span"]

    convert_path = "/nfs/scistore24/haimagrp/hwhitehe/ista-athenak/vis/python/make_athdf.py"

    for grade in grade_span:
        for beta in beta_span:
            for ecc in ecc_span:
                # define save_str
                run_name = "tde_{0}q{1}b{2}e{3}".format(grade, q, beta, ecc)
                run_dir = os.path.join(master_dir, run_name)
                save_str = os.path.join(run_dir, run_name + "_mass.npy")
                single_track_mass(run_dir, save_str, skip = skip)
                print("finished mass_track for {0}".format(run_name))

def single_track_mass(load_dir, save_str, skip = 1):

    dz = 0.1
    axis = 2
    slice_pos = 0
    bounds_2d = np.zeros(shape=(2, 2))
    bounds_2d[0, 0] = -1  # whole domain
    bounds_2d[0, 1] = 3
    bounds_2d[1, 0] = -1
    bounds_2d[1, 1] = 1

    hist_str = os.path.join(load_dir, "nbody.user.hst")
    hist_data = athena_read.hst(hist_str)
    P_bin = 2 * np.pi / np.sqrt(hist_data["m0"][0] + hist_data["m1"][0])
    t_hist = hist_data["time"]
    r_12 = np.sqrt(np.power(hist_data["x0"] - hist_data["x1"], 2) + np.power(hist_data["y0"] - hist_data["y1"], 2))
    r_H_ar = r_12 * np.cbrt(hist_data["m1"] / (3 * hist_data["m0"]))
    r_H_sqr = np.power(r_H_ar, 2.0)

    r_prim_sqr = 0.05 ** 2
    m_H_ar = []
    m_prim_ar = []
    t_H_ar = []

    for m in range(0, 200, skip):
        # load hydro data
        load_str = os.path.join(load_dir, "bin/nbody.hydro_w." + str(m).zfill(5) + ".athdf")
        try:
            HData = HydroDataAthenaK(load_str)
        except:
            break
        t_index = np.where(t_hist >= HData.Time)[0][0]
        data, levels = HData.inhomo_slice(axis=axis, slice_pos=slice_pos, slice_vars=["rho"], bounds_2d=bounds_2d)
        dx0 = data["x"] - hist_data["x0"][t_index]
        dy0 = data["y"] - hist_data["y0"][t_index]
        dx1 = data["x"] - hist_data["x1"][t_index]
        dy1 = data["y"] - hist_data["y1"][t_index]
        dr0_sqr = dx0 ** 2 + dy0 ** 2
        dr1_sqr = dx1 ** 2 + dy1 ** 2
        num_in_slice = np.size(levels)
        m_H = 0
        m_prim = 0
        for n in range(0, num_in_slice):
            in_prim = (dr0_sqr[n,:,:] < r_prim_sqr)
            in_Hill = (dr1_sqr[n,:,:] < r_H_sqr[t_index])
            in_prim = in_prim & ~in_Hill
            cell_volume = (data["x"][n,0,1] - data["x"][n,0,0]) * (data["y"][n,1,0] - data["y"][n,0,0]) * dz
            m_H += np.sum(data["rho"][n,:,:][in_Hill]) * cell_volume
            m_prim += np.sum(data["rho"][n,:,:][in_prim]) * cell_volume
        m_prim_ar.append(m_prim)
        m_H_ar.append(m_H)
        t_H_ar.append(HData.Time)

    data_to_save = np.vstack((np.array(t_H_ar), np.array(m_prim_ar), np.array(m_H_ar)))
    np.save(save_str, data_to_save)

def collect_files(master_dir, prod_params, mass_dir, hst_dir):

    grade_span    = prod_params["grade_span"]
    beta_span     = prod_params["beta_span"]
    ecc_span      = prod_params["ecc_span"]
    q             = prod_params["q_span"]

    for grade in grade_span:
        for beta in beta_span:
            for ecc in ecc_span:
                # define save_str
                run_name = "tde_{0}q{1}b{2}e{3}".format(grade, q, beta, ecc)
                run_dir = os.path.join(master_dir, run_name)
                mass_in = os.path.join(run_dir, run_name + "_mass.npy")
                hst_in = os.path.join(run_dir, "nbody.user.hst")

                mass_out = os.path.join(mass_dir, run_name + "_mass.npy")
                hst_out = os.path.join(hst_dir, num_name + ".hst")

                shutil.copyfile(mass_in, mass_out)
                shutil.copyfile(hst_in, hist_out)

                print("finished collection for {0}".format(run_name))

if __name__ == "__main__":

    master_dir = "/nfs/scistore24/haimagrp/hwhitehe/athenak_wdir/tde_prod"
    mass_dir = os.path.join(master_dir, "mass_store")
    hst_dir = os.path.join(master_dir, "hst_store")
    
    prod_params = {}
    prod_params["grade_span"] = ["P", "R"]
    prod_params["beta_span"]  = [0.5, 1, 2]
    prod_params["ecc_span"]   = [1.0, 2.0, 4.0]
    prod_params["q"]          = 0.1

    # convert binary files to atdf    
    #mass_convert_athdf(master_dir, prod_params)
    #print("finished binary conversion")
    mass_track_mass(master_dir, prod_params, skip=10)
    print("finished mass track")
    collect_files(master_dir, prod_params, mass_dir, hst_dir)