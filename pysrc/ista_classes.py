import sys, os
import numpy
import matplotlib.pyplot as plt

vis_python = os.path.join(os.path.dirname(__file__), '../vis/python')
if vis_python not in sys.path: sys.path.append(vis_python)

import athena_read

# TODO: deprecated BlackHole usage from the HydroData class

class HydroData:
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

        return data

    def homo_plot(self, save_str, npy_str=None, target=0, cvar="rho", axis=2, slice_pos=0, bounds_2d=None, level=None, vdata=None, verbose=False):

        if axis == 0:
            coords_c = ["y", "z", "x"]
        elif axis == 1:
            coords_c = ["x", "z", "y"]
        else:
            coords_c = ["x", "y", "z"]

        if cvar == "flow":
            slice_vars = ["v" + c for c in coords_c[:-1]]

        # build plotting space
        set_plot_defaults()
        fig = plt.figure()
        ax = fig.add_subplot()

        dU = 0
        dV = 0
        if npy_str is not None:
            bh = BlackHole(target, npy_str)
            t_idx = np.where(bh.t >= HData.Time)[0][0]
            shifts = [getattr(bh, coords_c[i])[t_idx] for i in range(2)]
            bump_axes = [range(3)].pop(axis)
            bounds_2d = bump_bounds(bounds_2d, shifts, axes=bump_axes)
            slice_pos = getattr(bh, coords_c[2])[t_idx]
            if cvar == "flow":
                dU = getattr(bh, slice_vars[0])[t_idx]
                dV = getattr(bh, slice_vars[1])[t_idx]

        # handle dimensionality
        is_2D = (self.RootGridSize[2] == 1)
        if is_2D and slice_pos != 0:
            raise Exception("off-centre slicing only supported for 3D simulations")

        if cvar == "flow":
            data = self.homo_slice(axis=axis, bounds_2d=bounds_2d, slice_pos=slice_pos, slice_vars=slice_vars, verbose=verbose, level=level)
            X, Y = np.meshgrid(data[coords_c[0]][:-1], data[coords_c[1]][:-1])
            U = data[slice_vars[0]][..., 0] - dU
            V = data[slice_vars[1]][..., 0] - dV

            ax.streamplot(X, Y, U, V, color='k', linewidth=1, arrowsize=0.5, density=5)
        else:
            if cvar == "Sigma":
                data = self.homo_sigma(bounds_2d=bounds_2d, level=level, verbose=verbose)
                xx, yy = np.meshgrid(data["x"], data["y"])
                cdata = data["Sigma"]
            elif cvar in ["rho", "vx", "vy", "vz"]:
                data = self.homo_slice(axis=axis, bounds_2d=bounds_2d, slice_pos=slice_pos, slice_vars=cvar, verbose=verbose, level=level)
                if axis == 0:
                    xx, yy = np.meshgrid(data["y"], data["z"])
                elif axis == 1:
                    xx, yy = np.meshgrid(data["x"], data["z"])
                else:
                    xx, yy = np.meshgrid(data["x"], data["y"])
                cdata = data[cvar]
            else:
                raise Exception("unable to pass cvar")

            vmin, vmax, v0, cmap = pass_vdata(vdata, cdata)
            cdata /= v0
            if cvar in ["rho", "P", "T", "Sigma"]:
                cdata = np.log10(cdata)
            ax.pcolormesh(xx, yy, cdata, vmin=vmin, vmax=vmax, cmap=cmap, zorder=-100)

        ax.set_xlim(bounds_2d[0, :])
        ax.set_ylim(bounds_2d[1, :])
        ax.set_aspect("equal")
        fig.savefig(save_str, dpi=300, bbox_inches="tight")
        plt.close("all")

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
                if bounds_2d[i, 0] is None: bounds_2d[i, 0] = np.min(getattr(self, c))
                if bounds_2d[i, 1] is None: bounds_2d[i, 1] = np.max(getattr(self, c))
        elif bounds_2d is None:
            bounds_2d = np.zeros(shape=(2,2))
            for i, c in enumerate(coords_c[:-1]):
                bounds_2d[i, 0] = np.min(getattr(self, c))
                bounds_2d[i, 1] = np.max(getattr(self, c))
        else:
            raise Exception("Invalid pass to bounds_2d, require input shape (2,2) or None")

        # build plotting space
        set_plot_defaults()
        fig = plt.figure()
        ax = fig.add_subplot()
        data = self.inhomo_slice(axis=axis, slice_pos=slice_pos, slice_vars=cvar, bounds_2d=bounds_2d)

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

    def calc_hill_mass(self, npy_str, target=0, r_scale=1):

        if np.size(target) == 2:
            bh1 = BlackHole(0, npy_str)
            bh2 = BlackHole(1, npy_str)
            t_idx = np.where(bh1.t >= self.Time)[0][0]
            p = calc_com(bh1, bh2, t_idx)
            l = calc_rh(bh1, bh2) * r_scale
        else:
            bh = BlackHole(target, npy_str)
            t_idx = np.where(bh.t >= self.Time)[0][0]
            p = [bh.x[t_idx], bh.y[t_idx], bh.z[t_idx]]
            l = calc_rh(bh) * r_scale

        bounds = [[-l, l]]
        m_h = 0
        l_sqr = l * l
        for n in range(0, self.NumMeshBlocks):
            dx = self.x[n, :] - p[0]
            dy = self.y[n, :] - p[1]
            dz = self.z[n, :] - p[2]
            if not in_bounds([dx, dy, dz], bounds): continue
            dv = (dx[1] - dx[0]) * (dy[1] - dy[0]) * (dz[1] - dz[0]) # cell volume uniform across MeshBlock
            xx, yy, zz = np.meshgrid(dx, dy, dz)
            r_sqr = xx * xx + yy * yy + zz * zz
            enc_mask = (r_sqr < l_sqr)
            rho = self.rho[n, :, :, :]
            m_h += np.sum(rho[enc_mask]) * dv

        return m_h

    def triple_plot(self, save_str, npy_str, cvar="rho", axis=0, bounds_2d=None, vdata=None, show_bh=True):

        if axis == 0:
            coords_c = ["y", "z", "x"]
        elif axis == 1:
            coords_c = ["x", "z", "y"]
        else:
            coords_c = ["x", "y", "z"]

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

        # build plotting space
        set_plot_defaults()
        h_over_w_minor = (bounds_2d[1, 1] - bounds_2d[1, 0]) / (bounds_2d[0, 1] - bounds_2d[0, 0])
        width_ratios = np.array([1, 0.05])
        height_ratios = np.array([h_over_w_minor] * 3)
        h_over_w = np.sum(height_ratios) / np.sum(width_ratios)
        L = 20.0 / 3
        fig = plt.figure(figsize = (L, L * h_over_w))
        gs = fig.add_gridspec(3, 2, height_ratios=height_ratios, width_ratios=width_ratios)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[2, 0])
        cax = fig.add_subplot(gs[:, 1])

        bh1 = BlackHole(0, npy_str)
        bh2 = BlackHole(1, npy_str)
        r_H = calc_rh(bh1, bh2)
        r_Hs_1 = calc_rh(bh1)
        r_Hs_2 = calc_rh(bh2)
        t_idx = np.where(bh1.t >= self.Time)[0][0]
        slice_pos_1 = getattr(bh1, coords_c[2])[t_idx]
        slice_pos_2 = getattr(bh2, coords_c[2])[t_idx]
        slice_pos_com = (bh1.m[t_idx] * slice_pos_1 + bh2.m[t_idx] * slice_pos_2) / (bh1.m[t_idx] + bh2.m[t_idx])

        dX_1 = getattr(bh1, coords_c[0])[t_idx]
        dY_1 = getattr(bh1, coords_c[1])[t_idx]
        dX_2 = getattr(bh2, coords_c[0])[t_idx]
        dY_2 = getattr(bh2, coords_c[1])[t_idx]
        dX_com = (bh1.m[t_idx] * dX_1 + bh2.m[t_idx] * dX_2) / (bh1.m[t_idx] + bh2.m[t_idx])
        dY_com = (bh1.m[t_idx] * dY_1 + bh2.m[t_idx] * dY_2) / (bh1.m[t_idx] + bh2.m[t_idx])
        bounds_1 = bump_bounds(bounds_2d, [dX_1, dY_1], axes=[0, 1])
        bounds_com = bump_bounds(bounds_2d, [dX_com, dY_com], axes=[0, 1])
        bounds_2 = bump_bounds(bounds_2d, [dX_2, dY_2], axes=[0, 1])

        data_1 = self.inhomo_slice(axis=axis, slice_pos=slice_pos_1, slice_vars=cvar, bounds_2d=None)
        data_com = self.inhomo_slice(axis=axis, slice_pos=slice_pos_com, slice_vars=cvar, bounds_2d=None)
        data_2 = self.inhomo_slice(axis=axis, slice_pos=slice_pos_2, slice_vars=cvar, bounds_2d=None)

        if (cvar not in data_1) or (cvar not in data_com) or (cvar not in data_2):
            raise Exception("could not find " + cvar + " in HData")

        if "vmin" in vdata:
            vmin = vdata["vmin"]
        else:
            vmin = np.min(np.array([np.min(data_1[cvar]), np.min(data_com[cvar]), np.min(data_2[cvar])]))

        if "vmax" in vdata:
            vmax = vdata["vmax"]
        else:
            vmax = np.max(np.array([np.max(data_1[cvar]), np.max(data_com[cvar]), np.max(data_2[cvar])]))

        if "v0" in vdata:
            v0 = vdata["v0"]
        else:
            v0 = 1

        if "cmap" in vdata:
            cmap = vdata["cmap"]
        else:
            cmap = "plasma"

        is_log = False
        axes = [ax0, ax1, ax2]
        datas = [data_1, data_com, data_2]
        slices = [slice_pos_1, slice_pos_com, slice_pos_2]
        all_bounds = [bounds_1, bounds_com, bounds_2]
        bh_labels = ["$\mathrm{BH1}$\n", "$\mathrm{COM}$\n", "$\mathrm{BH2}$\n"]
        for i, (ax, data, spos, bounds, bh_label) in enumerate(zip(axes, datas, slices, all_bounds, bh_labels)):
            CC = data[cvar] / v0
            if cvar in ["rho", "P", "T"]:
                CC = np.log10(CC)
                is_log = True

            XX = data[coords_c[0]]
            YY = data[coords_c[1]]
            num_in_slice = np.shape(CC)[0]

            for n in range(0, num_in_slice):
                ax.pcolormesh(XX[n, :, :], YY[n, :, :], CC[n, :, :], vmin=vmin, vmax=vmax, cmap=cmap)

            ax.set_xlim(bounds[0, :])
            ax.set_ylim(bounds[1, :])
            ax.set_aspect("equal")
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

            sb = AnchoredSizeBar(ax.transData, r_H, r"$r_\mathrm{H}$", "lower right", zorder=100, pad=0.5,
                                  size_vertical=r_H/ 40, frameon=False, color='white', label_top=True)
            ax.add_artist(sb)

            pos_text = "$" + coords_c[2] + " = " + str(round(spos / r_H, 2)) + "r_\mathrm{H}$"
            label = bh_label + pos_text
            ax.text(0.025, 0.05, s=label, va="bottom", ha="left", transform=ax.transAxes, color='w')

            if show_bh:
                Hill_1 = plt.Circle((dX_1, dY_1), radius=r_Hs_1, fill=False, edgecolor="deepskyblue", lw=1.5, alpha=0.5)
                Hill_2 = plt.Circle((dX_2, dY_2), radius=r_Hs_2, fill=False, edgecolor="greenyellow", lw=1.5, alpha=0.5)
                ax.add_artist(Hill_1)
                ax.add_artist(Hill_2)
                if i == 1:
                    c = "w" if cvar == "T" else "k"
                    Hill_com = plt.Circle((dX_com, dY_com), radius=r_H, fill=False, edgecolor=c, lw=1.5, alpha=0.5)
                    ax.add_artist(Hill_com)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        fig.colorbar(sm, cax=cax, orientation="vertical")
        if cvar == "rho":
            cvar_tex = r"\rho"
        else:
            cvar_tex = cvar
        c0_tex = cvar + r"_0"
        if is_log:
            clabel = r"$\mathrm{log}_{10}(" + cvar_tex + "/" + c0_tex + ")$"
        else:
            clabel = r"$" + cvar_tex + "/" + c0_tex + ")$"
        cax.set_ylabel(clabel)

        plt.subplots_adjust(hspace=0, wspace=0)
        ax.set_aspect("equal")
        fig.savefig(save_str, dpi=1000, bbox_inches="tight")
        plt.close("all")

    def show_mb(self, save_str, npy_str):

        bh = BlackHole(0, npy_str)
        n_max = np.max(self.Levels)
        r_h = calc_rh(bh)
        t_idx = np.where(bh.t >= self.Time)[0][0]

        l = 1.5
        bounds_z = np.zeros(shape=(2, 2))
        bounds_z[0, 0] = bh.x[t_idx] - l * r_h
        bounds_z[0, 1] = bh.x[t_idx] + l * r_h
        bounds_z[1, 0] = bh.y[t_idx] - l * r_h
        bounds_z[1, 1] = bh.y[t_idx] + l * r_h
        z_data = self.inhomo_slice(axis=2, slice_pos=bh.z[t_idx], slice_vars=[], bounds_2d=bounds_z)

        bounds_x = np.zeros(shape=(2, 2))
        bounds_x[0, 0] = bh.y[t_idx] - l * r_h
        bounds_x[0, 1] = bh.y[t_idx] + l * r_h
        bounds_x[1, 0] = bh.z[t_idx] - l * r_h
        bounds_x[1, 1] = bh.z[t_idx] + l * r_h
        x_data = self.inhomo_slice(axis=0, slice_pos=bh.x[t_idx], slice_vars=[], bounds_2d=bounds_x)

        set_plot_defaults()
        L = 20.0 / 3
        height_ratios = np.array([1])
        width_ratios = np.array([1,1,0.05])
        h_over_w = np.sum(height_ratios) / np.sum(width_ratios)
        fig = plt.figure(figsize=(L, L * h_over_w))
        gs = fig.add_gridspec(1, 3, width_ratios=width_ratios, height_ratios=height_ratios)
        ax_z = fig.add_subplot(gs[0, 0])
        ax_x = fig.add_subplot(gs[0, 1])
        cax = fig.add_subplot(gs[0, 2])

        ax_z.set_title(r"$x-y$ $\mathrm{plane}$")
        xl = np.min(z_data["x"],axis=(1,2))
        yl = np.min(z_data["y"],axis=(1,2))
        xr = np.max(z_data["x"],axis=(1,2))
        mb_l = xr - xl
        min_mb_l = np.min(mb_l)
        R = np.round(np.log2(mb_l / min_mb_l))
        cmap = plt.get_cmap("viridis")
        for x0, y0, s, r in zip(xl, yl, mb_l, R):
            level = n_max - r
            c = cmap(level / n_max)
            mb = patches.Rectangle((x0, y0), s, s, edgecolor='k', fill=True, facecolor=c, lw=0.4)
            ax_z.add_patch(mb)
        inner = plt.Circle((bh.x[t_idx], bh.y[t_idx]), radius=0.2 * r_h, fill=False, edgecolor="k", lw=1.5, linestyle="solid")
        outer = plt.Circle((bh.x[t_idx], bh.y[t_idx]), radius=0.5 * r_h, fill=False, edgecolor="k", lw=1.5, linestyle="dashed")
        Hill = plt.Circle((bh.x[t_idx], bh.y[t_idx]), radius=r_h, fill=False, edgecolor="k", lw=1.5, linestyle="dotted")
        ax_z.add_patch(inner)
        ax_z.add_patch(outer)
        ax_z.add_patch(Hill)

        ax_x.set_title(r"$y-z$ $\mathrm{plane}$")
        yl = np.min(x_data["y"], axis=(1, 2))
        zl = np.min(x_data["z"], axis=(1, 2))
        yr = np.max(x_data["y"], axis=(1, 2))
        mb_l = yr - yl
        R = np.round(np.log2(mb_l / min_mb_l))
        for y0, z0, s, r in zip(yl, zl, mb_l, R):
            level = n_max - r
            c = cmap(level / n_max)
            mb = patches.Rectangle((y0, z0), s, s, edgecolor='k', fill=True, facecolor=c, lw=0.4)
            ax_x.add_patch(mb)
        inner = patches.Ellipse((bh.y[t_idx], bh.z[t_idx]), width=0.4 * r_h, height=0.2 * r_h, fill=False, edgecolor="k", lw=1.5, linestyle="solid")
        outer = patches.Ellipse((bh.y[t_idx], bh.z[t_idx]), width=r_h, height=0.5 * r_h, fill=False, edgecolor="k", lw=1.5, linestyle="dashed")
        Hill = plt.Circle((bh.y[t_idx], bh.z[t_idx]), radius=r_h, fill=False, edgecolor="k", lw=1.5, linestyle="dotted")
        ax_x.add_patch(inner)
        ax_x.add_patch(outer)
        ax_x.add_patch(Hill)

        for ax, b in zip([ax_z, ax_x], [bounds_z, bounds_x]):
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_xlim(b[0, :])
            ax.set_ylim(b[1, :])
            ax.set_aspect("equal")

        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
        fig.colorbar(sm, cax=cax, orientation="vertical")
        cticks = np.linspace(0, 1, n_max + 1)
        clabels = ["$n = " + str(n) + "$" for n in range(n_max + 1)]
        cax.yaxis.set_ticks(cticks)
        cax.yaxis.set_ticklabels(clabels)

        plt.subplots_adjust(hspace=0, wspace=0)
        fig.savefig(save_str, dpi=300, bbox_inches="tight")
        plt.close("all")


# TODO: migrate plotting tools from HydroData to QuickPlot

class QuickPlot:
    """"
    This class is used to generate simple plots from data packaged in the HydroData class
    """"
    def __init__(self, h_str):

        self.Hydro = HydroData(h_str)
