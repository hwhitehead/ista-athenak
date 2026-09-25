import sys, os, re
import numpy as np
import matplotlib.pyplot as plt

vis_python = os.path.join(os.path.dirname(__file__), '../vis/python')
if vis_python not in sys.path: sys.path.append(vis_python)

import athena_read

# TODO: deprecated BlackHole usage from the HydroData class

class HydroDataAthenaK:
    """
    this class loads single HDF5 snapshots into python memory retaining labels
    it also features regularisation routines to compile MeshBlocks into homogenous meshes
    """

    def __init__(self, h_str, user_vars="T"):
        print("Loading {0} as HydroData".format(h_str))
        self.coord_str = ["x", "y", "z"]
        # rename variables for user ease
        self.variable_dict = {
            # primitive
            "rho": "rho",
            "press": "P",
            "vel1": "vx",
            "vel2": "vy",
            "vel3": "vz",
            "r0": "C_J",
            # conservative
            "dens": "rho",  # degenerate for non-relativistic simulations
            "Etot": "E",
            "mom1": "Mx",
            "mom2": "My",
            "mom3": "Mz",
            "s0": "DC_J",
        }
        # accept user specified labels for user_out_var
        if user_vars is not None:
            for i, user_var in enumerate(user_vars):
                self.variable_dict.update({"user_out_var" + str(i): user_var})
        # load data from HDF5 file
        with h5py.File(h_str, 'r') as f:
            # copy topline attributes
            for attr in list(f.attrs):
                setattr(self, attr, f.attrs.get(attr))
            # copy coordinates
            for attr in ("x1f", "x2f", "x3f", "x1v", "x2v", "x3v", "Levels", "LogicalLocations"):
                setattr(self, attr, np.array(f[attr]))
            # dupe coordinates for ease of access
            self.x = self.x1v
            self.y = self.x2v
            self.z = self.x3v
            # copy hydro data
            variable_names = np.array([x.decode("ascii", "replace") for x in f.attrs["VariableNames"][:]])
            dataset_sizes = f.attrs['NumVariables'][:]
            dataset_names = np.array([x.decode('ascii', 'replace') for x in f.attrs['DatasetNames'][:]])
            for dataset_index, dataset_name in enumerate(dataset_names):
                variable_begin = sum(dataset_sizes[:dataset_index])
                variable_end = variable_begin + dataset_sizes[dataset_index]
                variable_names_local = variable_names[variable_begin:variable_end]
                for variable_index, variable_name in enumerate(variable_names_local):
                    if variable_name in self.variable_dict: # if new label exists, relabel
                        attr_name = self.variable_dict[variable_name]
                    else:
                        attr_name = variable_name
                    setattr(self, attr_name, np.array(f[dataset_name][variable_index, ...]))

    def homogenize(self, level=None, homo_vars=None, verbose=False, bounds=None):
        mb_size = self.MeshBlockSize
        max_level = np.max(self.Levels)
        # test restriction limits
        if level is None:
            level = max_level
        if level > max_level:
            warnings.warn("target level {0} exceeds maximum mesh level {1}".format(level, max_level))
        else:
            max_restrict = 2 ** (max_level - level)
            for d in range(0, 3):
                if mb_size[d] != 1 and mb_size[d] < max_restrict:
                    limit = max_level - int(np.log2(mb_size[d]))
                    warnings.warn(
                        "target level " + str(level) + " too low for restriction routine, must be >= " + str(limit))

        # define dimensions of output array
        nx_vals = []
        for d in range(3): # handle slicing here?
            if mb_size[d] == 1: # do not expand along unexpanded dimension
                nx_vals.append(self.RootGridSize[d])
            else:
                nx_vals.append(self.RootGridSize[d] * 2 ** level)
        nx1 = nx_vals[0]
        nx2 = nx_vals[1]
        nx3 = nx_vals[2]

        if verbose:
            print("starting homogenization routine...")
            print("root grid                            [nk, nj, ni] = [{0}, {1}, {2}]".format(*self.RootGridSize[-1::-1]))
            print("max mesh level                                    = {0}".format(max_level))
            print("homogenous level                                  = {0}".format(level))
            print("master grid                          [nk, nj, ni] = [{0}, {1}, {2}]".format(nx3, nx2, nx1))

        # populate coordinate arrays
        data = {}
        for d, (nx, c) in enumerate(zip(nx_vals, self.coord_str)):
            xmin = getattr(self, "RootGridX" + str(d+1))[0]
            xmax = getattr(self, "RootGridX" + str(d+1))[1]
            data[c] = np.linspace(xmin, xmax, nx+1)

        # account for domain selection
        index_lim = np.zeros(shape=(3,2), dtype=np.int32)
        index_lim[0, 1] = nx1
        index_lim[1, 1] = nx2
        index_lim[2, 1] = nx3
        trims = np.array([False, False, False])
        slices = np.array([False, False, False])
        err_string = "{0} must be {1} than {2} in order to overlap domain"
        if np.any(bounds is not None): # trim domain to spec
            # test user input
            if np.shape(bounds) != (3, 2):
                raise Exception("Invalid pass to bounds, require input shape (3,2)")
            # test if bounds in domain
            for d, c in enumerate(self.coord_str):
                bound = bounds[d, :]
                if bound[0] is not None and bound[0] >= data[c][0]:
                    if bound[0] >= data[c][-1]:
                        raise Exception(err_string.format(c + "_min", "less", data[c][-1]))
                    index_lim[d, 0] = np.where(data[c] <= bound[0])[0][-1]
                    trims[d] = True
                if bound[1] is not None and bound[1] <= data[c][-1]:
                    if bound[1] <= data[c][0]:
                        raise Exception(err_string.format(c, "_max", "greater", data[c][0]))
                    index_lim[d, 1] = np.where(data[c] >= bound[1])[0][0]
                    trims[d] = True
                if nx_vals[d] != 1: # if extended dimension, check for slice
                    if bound[0] == bound[1] and (bound[0] is not None) and (bound[1] is not None): # select slice
                        slices[d] = True
                        index_lim[d, 1] += 1 # bump to allow for single value

        # trim data arrays
        for d, c in enumerate(self.coord_str):
            if trims[d]:
                data[c] = data[c][index_lim[d, 0]:index_lim[d, 1] + 1]

        # unpack indices
        i_min = index_lim[0, 0]
        i_max = index_lim[0, 1]
        j_min = index_lim[1, 0]
        j_max = index_lim[1, 1]
        k_min = index_lim[2, 0]
        k_max = index_lim[2, 1]

        # identify output variables to merge
        if homo_vars is None:  # merge all variables
            homo_vars = []
            for q, (x, variable_name) in enumerate(self.variable_dict.items()):
                if hasattr(self, variable_name):
                    homo_vars.append(variable_name)
        elif isinstance(homo_vars, str):  # accept single variable specified
            homo_vars = [homo_vars]
        elif not (isinstance(homo_vars, list) or isinstance(homo_vars, np.ndarray)): # accept list of str
            raise Exception("invalid pass to homo_vars")

        # purge improper variables
        for merge_var in homo_vars:
            if not hasattr(self, merge_var):
                homo_vars.remove(merge_var)
                print("removing {0}" + str(merge_var) + " from list")

        # build output array
        for merge_var in homo_vars:
            data.update({merge_var: np.zeros((k_max - k_min, j_max - j_min, i_max - i_min))})

        if verbose:
            print("apply bounds [xmin, xmax, ymin, ymax, zmin, zmax] = [{0}, {1}, {2}, {3}, {4}, {5}]".format(*np.ravel(bounds)))
            print("homogenous grid                      [nk, nj, ni] = [{0}, {1}, {2}]".format(k_max - k_min, j_max - j_min, i_max - i_min))
            print("homogenizing hydro variables                       ", homo_vars)

        # iterate over mb
        for mb_num in range(self.NumMeshBlocks):
            mb_level = self.Levels[mb_num]
            mb_location = self.LogicalLocations[mb_num, :]

            # apply prolongation to coarse, copy same-level
            if mb_level <= level:
                # scale multiplier
                s = 2 ** (level - mb_level)
                # destination indices in merged
                il_d = (mb_location[0] * mb_size[0] * s
                        if nx1 > 1 else 0)
                jl_d = (mb_location[1] * mb_size[1] * s
                        if nx2 > 1 else 0)
                kl_d = (mb_location[2] * mb_size[2]* s
                        if nx3 > 1 else 0)
                iu_d = il_d + mb_size[0] * s if nx1 > 1 else 1
                ju_d = jl_d + mb_size[1] * s if nx2 > 1 else 1
                ku_d = kl_d + mb_size[2] * s if nx3 > 1 else 1

                # Calculate (prolongated) source indices, with selection
                il_s = max(il_d, i_min) - il_d
                jl_s = max(jl_d, j_min) - jl_d
                kl_s = max(kl_d, k_min) - kl_d
                iu_s = min(iu_d, i_max) - il_d
                ju_s = min(ju_d, j_max) - jl_d
                ku_s = min(ku_d, k_max) - kl_d
                if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
                    continue

                # Account for selection in destination indices
                il_d = max(il_d, i_min) - i_min
                jl_d = max(jl_d, j_min) - j_min
                kl_d = max(kl_d, k_min) - k_min
                iu_d = min(iu_d, i_max) - i_min
                ju_d = min(ju_d, j_max) - j_min
                ku_d = min(ku_d, k_max) - k_min

                # insert values
                for merge_var in homo_vars:
                    mb_data = getattr(self, merge_var)[mb_num, ...]
                    if s > 1: # level != mb_level, prolongate data and insert
                        if nx1 > 1:
                            mb_data = np.repeat(mb_data, s, axis=2)[:, :, il_s:iu_s]
                        if nx2 > 1:
                            mb_data = np.repeat(mb_data, s, axis=1)[:, jl_s:ju_s, :]
                        if nx3 > 1:
                            mb_data = np.repeat(mb_data, s, axis=0)[kl_s:ku_s, :, :]
                        data[merge_var][kl_d:ku_d, jl_d:ju_d, il_d:iu_d] = mb_data
                    else: # level match, insert directly
                        data[merge_var][kl_d:ku_d, jl_d:ju_d, il_d:iu_d] = mb_data[kl_s:ku_s,
                                                                   jl_s:ju_s,
                                                                   il_s:iu_s]
            else: # restrict fine data
                # Calculate scale
                s = 2 ** (mb_level - level)

                # Calculate destination indices, without selection
                il_d = mb_location[0] * mb_size[0] // s if nx1 > 1 else 0
                jl_d = mb_location[1] * mb_size[1] // s if nx2 > 1 else 0
                kl_d = mb_location[2] * mb_size[2] // s if nx3 > 1 else 0
                iu_d = il_d + mb_size[0] // s if nx1 > 1 else 1
                ju_d = jl_d + mb_size[1] // s if nx2 > 1 else 1
                ku_d = kl_d + mb_size[2] // s if nx3 > 1 else 1

                # Calculate (restricted) source indices, with selection
                il_s = max(il_d, i_min) - il_d
                jl_s = max(jl_d, j_min) - jl_d
                kl_s = max(kl_d, k_min) - kl_d
                iu_s = min(iu_d, i_max) - il_d
                ju_s = min(ju_d, j_max) - jl_d
                ku_s = min(ku_d, k_max) - kl_d
                if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
                    continue

                # Account for selection in destination indices
                il_d = max(il_d, i_min) - i_min
                jl_d = max(jl_d, j_min) - j_min
                kl_d = max(kl_d, k_min) - k_min
                iu_d = min(iu_d, i_max) - i_min
                ju_d = min(ju_d, j_max) - j_min
                ku_d = min(ku_d, k_max) - k_min

                # Account for restriction in source indices
                if nx1 > 1:
                    il_s *= s
                    iu_s *= s
                if nx2 > 1:
                    jl_s *= s
                    ju_s *= s
                if nx3 > 1:
                    kl_s *= s
                    ku_s *= s

                # Apply subsampling
                # Calculate fine-level offsets (nearest cell at or below center)
                o1 = s // 2 - 1 if nx1 > 1 else 0
                o2 = s // 2 - 1 if nx2 > 1 else 0
                o3 = s // 2 - 1 if nx3 > 1 else 0

                # Assign values
                for merge_var in homo_vars:
                    data[merge_var][kl_d:ku_d,
                    jl_d:ju_d,
                    il_d:iu_d] = getattr(self, merge_var)[mb_num, kl_s + o3:ku_s:s,
                                 jl_s + o2:ju_s:s, il_s + o1:iu_s:s]

        if verbose:
            print("finished homogenizing.")

        return data

    def homo_slice(self, axis=2, slice_pos=0, level=None, slice_vars=None, verbose=False, bounds_2d=None):

        new_row = np.array([slice_pos, slice_pos])
        bounds_3d = np.insert(bounds_2d, axis, new_row, axis=0)
        data = self.homogenize(level=level, homo_vars=slice_vars, verbose=verbose, bounds=bounds_3d)
        if axis == 0:
            slice_index = np.s_[:, :, 0]
        elif axis == 1:
            slice_index = np.s_[:, 0, :]
        else:
            slice_index = np.s_[0, :, :]
        for slice_var in self.variable_dict.keys():
            if slice_var in ["x", "y", "z"]: continue
            if slice_var in data.keys():
                data[slice_var] = data[slice_var][slice_index]
        return data

    def homo_sigma(self, level=None, verbose=False, bounds_2d=None):

        new_row = np.array([None, None])
        bounds_3d = np.insert(bounds_2d, 2, new_row, axis=0) # only allows for z view
        data_3d = self.homogenize(level=level, homo_vars="rho", verbose=verbose, bounds=bounds_3d)
        data_2d = {}
        for d, c in enumerate(self.coord_str): # copy 2d coordinates from 3d array
            if d != 2:
                data_2d.update({c: data_3d[c]})
        # compute surface density
        delta_z = data_3d["z"][-1] - data_3d["z"][0]
        n_z = np.size(data_3d["z"])
        Sigma = np.sum(data_3d["rho"], axis=0) * delta_z / n_z
        data_2d.update({"Sigma": Sigma})
        del data_3d
        return data_2d

    def inhomo_slice(self, axis=2, slice_pos=0, slice_vars=None, verbose=False, bounds_2d=None):

        # order coordinate arrays
        if axis == 0:
            coords = [self.x2v, self.x3v, self.x1v]
            slice_size = [self.MeshBlockSize[1], self.MeshBlockSize[2]]
            coords_c = ["y", "z", "x"]
            fcoords = [self.x2f, self.x3f, self.x1f]
        elif axis == 1:
            coords = [self.x1v, self.x3v, self.x2v]
            slice_size = [self.MeshBlockSize[0], self.MeshBlockSize[2]]
            coords_c = ["x", "z", "y"]
            fcoords = [self.x1f, self.x3f, self.x2f]
        else:
            coords = [self.x1v, self.x2v, self.x3v]
            slice_size = [self.MeshBlockSize[0], self.MeshBlockSize[1]]
            coords_c = ["x", "y", "z"]
            fcoords = [self.x1f, self.x2f, self.x3f]

        if isinstance(bounds_2d, np.ndarray) & (np.shape(bounds_2d) == (2,2)): # accept array, potentially containing None
            for i, c in enumerate(coords_c[:-1]):
                if bounds_2d[i, 0] is None: bounds_2d[i, 0] = np.min(getattr(self, c))
                if bounds_2d[i, 1] is None: bounds_2d[i, 1] = np.max(getattr(self, c))
        elif bounds_2d is None:
            bounds_2d = np.zeros(shape=(2,2))
            for i, c in enumerate(coords_c[:-1]):
                bounds_2d[i, 0] = np.min(getattr(self, c))
                bounds_2d[i, 1] = np.max(getattr(self, c))
        else:
            raise Exception("Invalid pass to bounds_2d, require input shape (2,2) or None")

        # identify MeshBlocks straddling slice within bounds_2d
        on_slice = (fcoords[2][:, 0] < slice_pos) & (fcoords[2][:, -1] >= slice_pos)
        miss_lr = (fcoords[0][:, -1] < bounds_2d[0, 0]) | (fcoords[0][:, 0] > bounds_2d[0, 1])
        miss_ud = (fcoords[1][:, -1] < bounds_2d[1, 0]) | (fcoords[1][:, 0] > bounds_2d[1, 1])
        in_slice = on_slice & ~(miss_lr | miss_ud)
        mb_ids = np.where(in_slice)[0]
        num_in_slice = np.size(mb_ids)

        # find slice index for intersecting MeshBlocks
        slice_dist = np.abs(coords[2] - slice_pos)
        min_pos = np.argmin(slice_dist, axis=1)

        # generate output
        data = {}
        if slice_vars is None:
            slice_vars = []
            for variable_name in self.variable_dict.items():
                if hasattr(self, variable_name):
                    slice_vars.append(variable_name)
        elif isinstance(slice_vars, str):
            slice_vars = [slice_vars]
        elif not (isinstance(slice_vars, list) or isinstance(slice_vars, np.ndarray)):  # accept list of str
            raise Exception("invalid pass to slice_vars")

        # add empty arrays for coordinates
        data.update({coords_c[0]: np.zeros(shape=(num_in_slice, slice_size[0], slice_size[1]))})
        data.update({coords_c[1]: np.zeros(shape=(num_in_slice, slice_size[0], slice_size[1]))})
        data.update({coords_c[2]: slice_pos}) # single value for sliced axis

        # add empty arrays for hydro variables
        for slice_var in slice_vars:
            data.update({slice_var: np.zeros(shape=(num_in_slice, slice_size[0], slice_size[1]))})

        # extract in_slice data
        levels = np.zeros_like(mb_ids)
        for i, n in enumerate(mb_ids):
            if axis == 0:
                mb_slice = (n, np.s_[:], np.s_[:], min_pos[n])
            elif axis == 1:
                mb_slice = (n, np.s_[:], min_pos[n], np.s_[:])
            else:
                mb_slice = (n, min_pos[n], np.s_[:], np.s_[:])
            XX, YY = np.meshgrid(coords[0][n, :], coords[1][n, :])
            data[coords_c[0]][i, :, :] = XX
            data[coords_c[1]][i, :, :] = YY
            for slice_var in slice_vars:
                data[slice_var][i, :, :] = getattr(self, slice_var)[mb_slice]
            levels[i] = self.Levels[n]

        return data, levels

class QuickPlot:
    """
    This class is used to generate simple plots from data packaged in the HydroData class
    """
    def __init__(self, h_str):

        self.Hydro = HydroData(h_str)

    def inhomo_plot(self, save_str, npy_str=None, target=None, cvar="rho", axis=2, slice_pos=0, bounds_2d=None, vdata=None):

        if axis == 0:
            coords_c = ["y", "z", "x"]
            fcoords = ["x2f", "x3f", "x1f"]
        elif axis == 1:
            coords_c = ["x", "z", "y"]
            fcoords = ["x1f", "x3f", "x2f"]
        else:
            coords_c = ["x", "y", "z"]
            fcoords = ["x1f", "x2f", "x3f"]

        if isinstance(bounds_2d, np.ndarray) & (np.shape(bounds_2d) == (2,2)): # accept array, potentially containing None
            for i, c in enumerate(coords_c[:-1]):
                if bounds_2d[i, 0] is None: bounds_2d[i, 0] = np.min(getattr(self.Hydro, c))
                if bounds_2d[i, 1] is None: bounds_2d[i, 1] = np.max(getattr(self.Hydro, c))
        elif bounds_2d is None:
            bounds_2d = np.zeros(shape=(2,2))
            for i, c in enumerate(coords_c[:-1]):
                bounds_2d[i, 0] = np.min(getattr(self.Hydro, c))
                bounds_2d[i, 1] = np.max(getattr(self.Hydro, c))
        else:
            raise Exception("Invalid pass to bounds_2d, require input shape (2,2) or None")

        # build plotting space
        set_plot_defaults()
        fig = plt.figure()
        ax = fig.add_subplot()
        data = self.Hydro.inhomo_slice(axis=axis, slice_pos=slice_pos, slice_vars=cvar, bounds_2d=bounds_2d)

        if cvar not in data:
            raise Exception("could not find " + cvar + " in HData")
        CC = data[cvar]

        if "vmin" in vdata:
            vmin = vdata["vmin"]
        else:
            vmin = np.min(CC)

        if "vmax" in vdata:
            vmax = vdata["vmax"]
        else:
            vmax = np.max(CC)

        if "v0" in vdata:
            v0 = vdata["v0"]
        else:
            v0 = 1

        if "cmap" in vdata:
            cmap = vdata["cmap"]
        else:
            cmap = "plasma"

        CC = CC / v0
        if cvar in ["rho", "P", "T"]:
            CC = np.log10(CC)

        XX = data[coords_c[0]]
        YY = data[coords_c[1]]
        num_in_slice = np.shape(CC)[0]

        for n in range(0, num_in_slice):
            ax.pcolormesh(XX[n, :, :], YY[n, :, :], CC[n, :, :], vmin=vmin, vmax=vmax, cmap=cmap)

        ax.set_xlim(bounds_2d[0, :])
        ax.set_ylim(bounds_2d[1, :])
        ax.set_aspect("equal")
        fig.savefig(save_str, dpi=300, bbox_inches="tight")
        plt.close("all")

class NBody:

    def __init__(self, hist_str):

        self.data = athena_read.hst(hist_str)
        self.t = self.data["time"]
        self.num_chkpts = np.size(self.t)
        self.num_nbody = 0
        for key in self.data.keys():
            if key.startswith("m"):
                self.num_nbody += 1
        self.G = 1
        self.build_var_names()

    def build_var_names(self):

        keys = self.data.keys()
        self.var_names = []
        for key in keys:
            if key.endswith("0"): # scan first body for variable names
                self.var_names.append(key.rstrip("0"))

    def check_body_index(self, i):

        if i < 0:
            raise Exception("Body index must be postive (recieved n = {0})".format(i))
        elif i > self.num_nbody - 1:
            raise Exception("Body index {0} exceeds bounds (num_nbody = {1})".format(i, self.num_nbody))

    def check_time_index(self, i):

        if i < 0:
            raise Exception("Time index must be postive (recieved t_index = {0})".format(i))
        elif i > self.num_chkpts:
            raise Exception("Time index {0} exceeds bounds (num_nbody = {1})".format(i, self.num_chkpts))

    def calc_diff(self, var="r", i = 0, j = 1, t_index=None):

        # parse user input
        self.check_body_index(i)
        self.check_body_index(j)
        if t_index is not None: self.check_time_index(t_index)

        if t_index is not None:
            slicer = np.s_[t_index]
        else:
            slicer = np.s_[:]

        # check if generic property or compound
        if var in self.var_names:
            # generic, direct access without computation
            return self.data["{0}{1}".format(var,i)][slicer] - self.data["{0}{1}".format(var,j)][slicer]
        elif var == "r": # compound: radial seperation
            dx = self.calc_diff("x", i, j, t_index)
            dy = self.calc_diff("y", i, j, t_index)
            dz = self.calc_diff("z", i, j, t_index)
            return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        elif var == "v": # compound, velocity differential
            dvx = self.calc_diff("vx", i, j, t_index)
            dvy = self.calc_diff("vy", i, j, t_index)
            dvz = self.calc_diff("vz", i, j, t_index)
            return np.sqrt(dvx ** 2 + dvy ** 2 + dvz ** 2)
        else:
            except_msg = "Unable to recognise var passed to NBody.diff\n"
            except_msg += "Please select from {0}, r or v".format(self.var_names)
            raise Exception(except_msg)
    
    def calc_Ebin(self, i = 0, j = 1, t_index=None):

        # parse user input
        self.check_body_index(i)
        self.check_body_index(j)
        if t_index is not None: self.check_time_index(t_index)

        if t_index is not None:
            slicer = np.s_[t_index]
        else:
            slicer = np.s_[:]

        E_grav = self.G * (self.data["m{0}".format(i)][slicer] + self.data["m{0}".format(j)][slicer]) / self.calc_diff(var="r", i=i, j=j, t_index=t_index)
        E_kin = 0
        for n in [i, j]:
            v_sqr = 0
            for axis in ["x","y","z"]:
                v_sqr += self.data["v{0}{1}".format(axis, n)][slicer] ** 2
            E_kin += 0.5 * self.data["m{0}".format(n)][slicer] * v_sqr

        return E_grav + E_kin

    def calc_disp(self, i = 0, j = 1, t_index=None):

        # parse user input
        self.check_body_index(i)
        self.check_body_index(j)
        if t_index is not None: self.check_time_index(t_index)

        dvx = self.calc_diff("vx", i, j, t_index)
        dvy = self.calc_diff("vy", i, j,  t_index)
        dvz = self.calc_diff("vz", i, j, t_index)

        dax_grav = self.calc_diff("ax_grav", i, j, t_index)
        day_grav = self.calc_diff("ay_grav", i, j, t_index)
        daz_grav = self.calc_diff("az_grav", i, j, t_index)

        dax_acc = self.calc_diff("ax_acc", i, j, t_index)
        day_acc = self.calc_diff("ay_acc", i, j, t_index)
        daz_acc = self.calc_diff("az_acc", i, j, t_index)

        # compute specific work done
        Edot_grav = dvx * dax_grav + dvy * day_grav + dvz * daz_grav
        Edot_acc = dvx * dax_acc + dvy * day_acc + dvz * daz_acc

        return Edot_grav, Edot_acc