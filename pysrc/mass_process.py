import numpy as np
import matplotlib.pyplot as plt
import subprocess, os, sys
import shutil

vis_python = os.path.join(os.path.dirname(__file__), "../vis/python")
if vis_python not in sys.path: sys.path.append(vis_python)
import athena_read

from ista_classes import HydroDataAthenaK, NBody, InitBinary, InitParabola, InitOrbit

def count_files(path, file_ending):

    if os.path.isdir(path):
        return len([f for f in os.listdir(path) if f.endswith(file_ending) and os.path.isfile(os.path.join(path, f))])
    
    return None

def mass_convert_athdf(master_dir, prod_params):

    grade_span    = prod_params["grade_span"]
    beta_span     = prod_params["beta_span"]
    ecc_span      = prod_params["ecc_span"]
    q             = prod_params["q"]

    convert_path = os.path.join(os.path.dirname(__file__), "../vis/python/make_athdf.py")

    for grade in grade_span:
        for beta in beta_span:
            for ecc in ecc_span:
                # define save_str
                run_name = "tde_{0}q{1}b{2}e{3}".format(grade, q, beta, ecc)
                run_dir  = os.path.join(master_dir, run_name)
                bin_dir  = os.path.join(run_dir, "bin")
                bin_stem = os.path.join(bin_dir, "nbody.hydro_w")
                # check if last entry already converted
                num_bin = count_files(bin_dir, ".bin")
                num_athdf = count_files(bin_dir, ".athdf")
                if (num_bin == num_athdf):
                    print(".athdf creation already complete for {0}, skipping".format(run_name))
                    continue
                # invoke subproc to call binary converter
                subprocess.run(["python3", convert_path, bin_stem])
                print("finished .athdf creation for {0}".format(run_name))

def mass_track_mass(master_dir, prod_params, skip=1):

    grade_span    = prod_params["grade_span"]
    beta_span     = prod_params["beta_span"]
    ecc_span      = prod_params["ecc_span"]
    q             = prod_params["q"]

    for grade in grade_span:
        for beta in beta_span:
            for ecc in ecc_span:
                # define save_str
                run_name = "tde_{0}q{1}b{2}e{3}".format(grade, q, beta, ecc)
                run_dir = os.path.join(master_dir, run_name)
                save_str = os.path.join(run_dir, run_name + "_mass.npy")
                if os.path.exists(save_str):
                    print("mass_track already exists for {0}, skipping.".format(run_name))
                    continue
                single_track_mass(run_dir, save_str, skip = skip)
                print("finished mass_track for {0}".format(run_name))

def single_track_mass(load_dir, save_str, skip = 1):

    dz = 0.1
    axis = 2
    slice_pos = 0

    hist_str = os.path.join(load_dir, "nbody.user.hst")
    hist_data = athena_read.hst(hist_str)
    P_bin = 2 * np.pi / np.sqrt(hist_data["m0"][0] + hist_data["m1"][0])
    t_hist = hist_data["time"]
    r_12 = np.sqrt(np.power(hist_data["x0"] - hist_data["x1"], 2) + np.power(hist_data["y0"] - hist_data["y1"], 2))
    r_H_ar = r_12 * np.cbrt(hist_data["m1"] / (3 * hist_data["m0"]))
    r_H_sqr = np.power(r_H_ar, 2.0)

    r_prim_sqr = 0.05 ** 2
    m_prim_ar = []
    m_sec_ar = []
    t_ar = []

    for m in range(0, 201, skip):
        # load hydro data
        load_str = os.path.join(load_dir, "bin/nbody.hydro_w." + str(m).zfill(5) + ".athdf")
        try:
            HData = HydroDataAthenaK(load_str)
        except:
            break
        try:
            t_index = np.where(t_hist >= HData.Time)[0][0]
        except:
            break
        m_sec = 0
        m_prim = 0
        for n in range(HData.NumMeshBlocks):
            xx, yy = np.meshgrid(HData.x[n, :], HData.y[n, :])
            dx0 = xx - hist_data["x0"][t_index]
            dy0 = yy - hist_data["y0"][t_index]
            dx1 = xx - hist_data["x1"][t_index]
            dy1 = yy - hist_data["y1"][t_index]        
            dr0_sqr = dx0 ** 2 + dy0 ** 2
            dr1_sqr = dx1 ** 2 + dy1 ** 2
            in_prim = (dr0_sqr < r_prim_sqr)
            in_sec  = (dr1_sqr < r_H_sqr[t_index])
            in_prim = in_prim & ~in_sec
            cell_volume = (HData.x[n, 1] - HData.x[n, 0]) * (HData.y[n, 1] - HData.y[n, 0]) * dz
            m_sec += np.sum(HData.rho[n,0,:,:][in_sec]) * cell_volume
            m_prim += np.sum(HData.rho[n,0,:,:][in_prim]) * cell_volume
        m_prim_ar.append(m_prim)
        m_sec_ar.append(m_sec)
        t_ar.append(HData.Time)

    data_to_save = np.vstack((np.array(t_ar), np.array(m_prim_ar), np.array(m_sec_ar)))
    np.save(save_str, data_to_save)

def collect_files(master_dir, prod_params, mass_dir, hst_dir):

    grade_span    = prod_params["grade_span"]
    beta_span     = prod_params["beta_span"]
    ecc_span      = prod_params["ecc_span"]
    q             = prod_params["q"]

    for grade in grade_span:
        for beta in beta_span:
            for ecc in ecc_span:
                # define save_str
                run_name = "tde_{0}q{1}b{2}e{3}".format(grade, q, beta, ecc)
                run_dir = os.path.join(master_dir, run_name)
                mass_in = os.path.join(run_dir, run_name + "_mass.npy")
                hst_in = os.path.join(run_dir, "nbody.user.hst")

                mass_out = os.path.join(mass_dir, run_name + "_mass.npy")
                hst_out = os.path.join(hst_dir, run_name + ".hst")

                shutil.copyfile(mass_in, mass_out)
                shutil.copyfile(hst_in, hst_out)

                print("finished collection for {0}".format(run_name))

if __name__ == "__main__":

    master_dir = "/nfs/scistore24/haimagrp/hwhitehe/athenak_wdir/tde_prod"
    mass_dir = os.path.join(master_dir, "mass_dir")
    hst_dir = os.path.join(master_dir, "hst_dir")
    
    prod_params = {}
    prod_params["grade_span"] = ["P", "R"]
    prod_params["beta_span"]  = [0.5, 1, 2]
    prod_params["ecc_span"]   = [0.9, 1.0, 2.0, 4.0]
    prod_params["q"]          = 0.1

    # convert binary files to atdf    
    mass_convert_athdf(master_dir, prod_params)
    print("finished binary conversion")
    mass_track_mass(master_dir, prod_params, skip=1)
    print("finished mass track")
    collect_files(master_dir, prod_params, mass_dir, hst_dir)
    print("finished file collection")