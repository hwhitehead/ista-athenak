import argparse
import struct
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import os


text_width   = 5.0 #6.8
column_width = 2.4 #3.3
def configure_matplotlib():
    plt.rc('xtick' , labelsize=8)
    plt.rc('ytick' , labelsize=8)
    plt.rc('axes'  , labelsize=8)
    plt.rc('legend', fontsize=8)
    plt.rc('font', family='DejaVu Sans', size=8)
    plt.rc('text', usetex=True)
    plt.rcParams.update({
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 2,
        "ytick.major.width": 2,
        "xtick.color": "black",
        "ytick.color": "black",
    })
    plt.rcParams.update({
        "axes.linewidth": 2, 
    })
configure_matplotlib()



def binary_positions(t, a, e, q, m_total):
    mean_motion = np.sqrt(m_total / a ** 3)
    period      = 2.0 * np.pi * np.sqrt(a ** 3 / m_total)
    t_periapse  = t - np.floor(t / period) * period

    E = mean_motion * t_periapse
    for _ in range(10):
        f = E - e * np.sin(E) - mean_motion * t_periapse
        if abs(f) <= 1e-15:
            break
        E -= f / (1.0 - e * np.cos(E))

    primary = np.array([a * q / (1 + q) * (np.cos(E) - e),
                        a * q / (1 + q) * np.sqrt(1 - e ** 2) * np.sin(E), 0.0])
    secondary = np.array([-a / (1 + q) * (np.cos(E) - e),
                          -a / (1 + q) * np.sqrt(1 - e ** 2) * np.sin(E), 0.0])
    return primary, secondary


def binary_orbit_track(a, e, q, n_points=200):
    E = np.linspace(0.0, 2.0 * np.pi, n_points)
    primary = np.stack([a * q / (1 + q) * (np.cos(E) - e),
                        a * q / (1 + q) * np.sqrt(1 - e ** 2) * np.sin(E),
                        np.zeros_like(E)], axis=-1)
    secondary = np.stack([-a / (1 + q) * (np.cos(E) - e),
                          -a / (1 + q) * np.sqrt(1 - e ** 2) * np.sin(E),
                          np.zeros_like(E)], axis=-1)
    return primary, secondary


# Derived quantities
DERIVED = {
    'speed': lambda q, p: np.sqrt(q['velx'] ** 2 + q['vely'] ** 2 + q['velz'] ** 2),
    'pgas': lambda q, p: (float(p['hydro']['gamma']) - 1.0) * q['eint'],
}

# Axis labels 
LABELS = {
    'dens': r'$\rho/\rho_0$',
    'velx': r'$v_x$',
    'vely': r'$v_y$',
    'velz': r'$v_z$',
    'eint': r'$e$',
    'derived:speed': r'$|v|$',
    'derived:pgas': r'$P$',
}


def read_header(f):
    """Parse the fixed-format preamble (time, variable names/sizes, etc) and
    the embedded input-file text block, leaving f positioned at the start of
    the first meshblock's data."""
    line = f.readline().decode('ascii')
    if line != 'Athena binary output version=1.1\n':
        raise RuntimeError('Unrecognized data file format.')
    next(f)  # size of preheader
    line = f.readline().decode('ascii')
    if line[:7] != '  time=':
        raise RuntimeError('Could not read simulation time.')
    time = float(line[7:])
    next(f)  # cycle
    line = f.readline().decode('ascii')
    location_size = int(line[19:])
    line = f.readline().decode('ascii')
    variable_size = int(line[19:])
    next(f)  # number of variables
    line = f.readline().decode('ascii')
    variable_names = line[12:].split()
    line = f.readline().decode('ascii')
    header_offset = int(line[16:])

    location_format = 'f' if location_size == 4 else 'd'
    variable_format = 'f' if variable_size == 4 else 'd'

    start_of_data = f.tell() + header_offset
    input_data = {}
    section = None
    while f.tell() < start_of_data:
        line = f.readline().decode('ascii')
        if line[0] == '#':
            continue
        if line[0] == '<':
            section = line[1:-2]
            input_data[section] = {}
            continue
        key, val = line.split('=', 1)
        input_data[section][key.strip()] = val.split('#', 1)[0].strip()

    return dict(time=time, location_size=location_size, variable_size=variable_size,
                location_format=location_format, variable_format=variable_format,
                variable_names=variable_names, input_data=input_data)


def read_slice(data_file, variable, dimension, location):
    """Read one 2D slice (list of per-meshblock 2D arrays + their (x1min,
    x1max, x2min, x2max) extents) out of an AthenaK .bin dump, picking
    whichever block covers `location` at each refinement level present."""
    with open(data_file, 'rb') as f:
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0, 0)

        h = read_header(f)
        input_data = h['input_data']
        num_ghost = int(input_data['mesh']['nghost'])
        num_vars = len(h['variable_names'])

        if variable == 'level':
            var_ind = -1
        elif variable.startswith('derived:'):
            name = variable[len('derived:'):]
            if name not in DERIVED:
                raise RuntimeError(f'Unknown derived variable "{name}"; options are '
                                   f'{{{", ".join(DERIVED)}}}.')
            deps = [v for v in h['variable_names']] 
            var_ind = None
        else:
            if variable not in h['variable_names']:
                raise RuntimeError(f'Variable "{variable}" not found; options are '
                                   f'{{{", ".join(h["variable_names"])}}} or '
                                   f'derived:{{{", ".join(DERIVED)}}}.')
            var_ind = h['variable_names'].index(variable)

        max_level_calculated = -1
        block_loc_for_level, block_ind_for_level = [], []
        num_blocks_used = 0
        extents = []
        raw = {name: [] for name in h['variable_names']} if var_ind is None else None
        single = [] if var_ind is not None else None

        first_time = True
        while f.tell() < file_size:
            block_indices = np.array(struct.unpack('@6i', f.read(24))) - num_ghost
            block_i, block_j, block_k, block_level = struct.unpack('@4i', f.read(16))

            if first_time:
                block_nx = block_indices[1] - block_indices[0] + 1
                block_ny = block_indices[3] - block_indices[2] + 1
                block_nz = block_indices[5] - block_indices[4] + 1
                cells_per_block = block_nz * block_ny * block_nx
                block_cell_format = '=' + str(cells_per_block) + h['variable_format']
                variable_data_size = cells_per_block * h['variable_size']
                if dimension == 'x':
                    block_nx1, block_nx2, slice_block_n = block_ny, block_nz, block_nx
                    loc_min = float(input_data['mesh']['x1min'])
                    loc_max = float(input_data['mesh']['x1max'])
                    root_blocks = (int(input_data['mesh']['nx1'])
                                   // int(input_data['meshblock']['nx1']))
                elif dimension == 'y':
                    block_nx1, block_nx2, slice_block_n = block_nx, block_nz, block_ny
                    loc_min = float(input_data['mesh']['x2min'])
                    loc_max = float(input_data['mesh']['x2max'])
                    root_blocks = (int(input_data['mesh']['nx2'])
                                   // int(input_data['meshblock']['nx2']))
                else:
                    block_nx1, block_nx2, slice_block_n = block_nx, block_ny, block_nz
                    loc_min = float(input_data['mesh']['x3min'])
                    loc_max = float(input_data['mesh']['x3max'])
                    root_blocks = (int(input_data['mesh']['nx3'])
                                   // int(input_data['meshblock']['nx3']))
                slice_norm_coord = (location - loc_min) / (loc_max - loc_min)
                first_time = False

            if block_level > max_level_calculated:
                for level in range(max_level_calculated + 1, block_level + 1):
                    if location <= loc_min:
                        block_loc_for_level.append(0)
                        block_ind_for_level.append(0)
                    elif location >= loc_max:
                        block_loc_for_level.append(root_blocks - 1)
                        block_ind_for_level.append(slice_block_n - 1)
                    else:
                        slice_mesh_n = slice_block_n * root_blocks * 2 ** level
                        mesh_ind = int(slice_norm_coord * slice_mesh_n)
                        block_loc_for_level.append(mesh_ind // slice_block_n)
                        block_ind_for_level.append(mesh_ind
                                                   - slice_block_n * block_loc_for_level[-1])
                max_level_calculated = block_level

            block_loc = {'x': block_i, 'y': block_j, 'z': block_k}[dimension]
            if block_loc != block_loc_for_level[block_level]:
                f.seek(6 * h['location_size'] + num_vars * variable_data_size, 1)
                continue
            num_blocks_used += 1

            block_lims = struct.unpack('=6' + h['location_format'],
                                       f.read(6 * h['location_size']))
            if dimension == 'x':
                extents.append((block_lims[2], block_lims[3], block_lims[4], block_lims[5]))
            elif dimension == 'y':
                extents.append((block_lims[0], block_lims[1], block_lims[4], block_lims[5]))
            else:
                extents.append((block_lims[0], block_lims[1], block_lims[2], block_lims[3]))

            block_ind = block_ind_for_level[block_level]

            def take(arr3d):
                if dimension == 'x':
                    return arr3d[:, :, block_ind]
                elif dimension == 'y':
                    return arr3d[:, block_ind, :]
                return arr3d[block_ind, :, :]

            cell_data_start = f.tell()
            if var_ind == -1:
                shape = (block_nz, block_ny) if dimension == 'x' else \
                        (block_nz, block_nx) if dimension == 'y' else (block_ny, block_nx)
                single.append(np.full(shape, block_level))
            elif var_ind is not None:
                f.seek(cell_data_start + var_ind * variable_data_size, 0)
                cell_data = np.array(struct.unpack(block_cell_format,
                                                    f.read(variable_data_size))
                                     ).reshape(block_nz, block_ny, block_nx)
                single.append(take(cell_data))
            else:
                for ind, name in enumerate(h['variable_names']):
                    f.seek(cell_data_start + ind * variable_data_size, 0)
                    cell_data = np.array(struct.unpack(block_cell_format,
                                                        f.read(variable_data_size))
                                         ).reshape(block_nz, block_ny, block_nx)
                    raw[name].append(take(cell_data))
            f.seek(cell_data_start + num_vars * variable_data_size, 0)

    if var_ind is not None:
        quantity = np.array(single)
    else:
        quantities = {name: np.array(arrs) for name, arrs in raw.items()}
        quantity = DERIVED[variable[len('derived:'):]](quantities, input_data)

    return quantity, extents, h['time'], input_data


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data_file', help='AthenaK .bin dump')
    parser.add_argument('variable', nargs='?', default='dens', help='raw variable name (dens, velx, ...), "level", or derived:{{{", ".join(DERIVED)}}} (default: dens)')
    parser.add_argument('output_dir', nargs='?', default='.', help='output image path, or "show" (default: show)')
    parser.add_argument('-l', '--location', type=float, default=0.0, help='coordinate along the slice normal (default: 0)')
    parser.add_argument('--r_max', type=float, help='half-width of plot, centered at 0')
    parser.add_argument('--x1_min', type=float)
    parser.add_argument('--x1_max', type=float)
    parser.add_argument('--x2_min', type=float)
    parser.add_argument('--x2_max', type=float)
    parser.add_argument('-c', '--cmap', default='magma', help='Matplotlib colormap')
    parser.add_argument('-n', '--norm', choices=('linear', 'log'), default='log')
    parser.add_argument('--vmin', type=float, default=-3, help='colorbar min; log10 exponent if -n log (default -3 = 1e-3), literal value if -n linear')
    parser.add_argument('--vmax', type=float, default=0,
                        help='colorbar max; log10 exponent if -n log (default 1 '
                             '= 10), literal value if -n linear')
    parser.add_argument('--title')
    parser.add_argument('--colorbar-title')
    parser.add_argument('--no-colorbar', action='store_true')
    parser.add_argument('--cbar-width', type=float, default=0.025,
                        help='colorbar column width, as a fraction of the combined '
                             'width of the 3 panels (default: 0.025)')
    parser.add_argument('--hide-yaxis', action='store_true')
    parser.add_argument('--grid', action='store_true', help='outline meshblock boundaries')
    parser.add_argument('--binary', action='store_true',
                        help='mark the cbdiso_3d.cpp binary component positions')
    parser.add_argument('--binary_color', default='white')
    parser.add_argument('--binary_edgecolor', default='black')
    parser.add_argument('--binary_size', type=float, default=10)
    parser.add_argument('--dpi', type=float, default=400)
    args = parser.parse_args()

    matplotlib.use('agg')


    quantity_x, extents_x, time_x, input_data_x = read_slice(args.data_file, args.variable, 'x', args.location)
    quantity_y, extents_y, time_y, input_data_y = read_slice(args.data_file, args.variable, 'y', args.location)
    quantity_z, extents_z, time_z, input_data_z = read_slice(args.data_file, args.variable, 'z', args.location)

    if args.norm == 'log':
        norm = matplotlib.colors.LogNorm(vmin=10 ** args.vmin, vmax=10 ** args.vmax)
        vmin = vmax = None
    else:
        norm = None
        vmin, vmax = args.vmin, args.vmax

    # Each panel's true physical extent from the mesh 
    mesh = input_data_x['mesh']
    x1_min, x1_max = float(mesh['x1min']), float(mesh['x1max'])
    x2_min, x2_max = float(mesh['x2min']), float(mesh['x2max'])
    x3_min, x3_max = float(mesh['x3min']), float(mesh['x3max'])
    panel_xlim = {'x': (x2_min, x2_max), 'y': (x1_min, x1_max), 'z': (x1_min, x1_max)}
    panel_ylim = {'x': (x3_min, x3_max), 'y': (x3_min, x3_max), 'z': (x2_min, x2_max)}

    
    if args.r_max is not None:
        panel_xlim = {d: (-args.r_max, args.r_max) for d in panel_xlim}
        panel_ylim = {d: (-args.r_max, args.r_max) for d in panel_ylim}
    else:
        if args.x1_min is not None:
            panel_xlim = {d: (args.x1_min, panel_xlim[d][1]) for d in panel_xlim}
        if args.x1_max is not None:
            panel_xlim = {d: (panel_xlim[d][0], args.x1_max) for d in panel_xlim}
        if args.x2_min is not None:
            panel_ylim = {d: (args.x2_min, panel_ylim[d][1]) for d in panel_ylim}
        if args.x2_max is not None:
            panel_ylim = {d: (panel_ylim[d][0], args.x2_max) for d in panel_ylim}

    width_ratios = [panel_xlim[d][1] - panel_xlim[d][0] for d in ('x', 'y', 'z')]

    
    if not args.no_colorbar:
        width_ratios = width_ratios + [args.cbar_width * sum(width_ratios)]
        ncols = 4
    else:
        ncols = 3

    
    gs_left, gs_right, gs_top, gs_bottom = 0.11, 0.86, 0.95, 0.12
    total_height_units = panel_ylim['x'][1] - panel_ylim['x'][0]
    total_width_units = sum(width_ratios)
    fig_width = text_width
    plot_area_width = fig_width * (gs_right - gs_left)
    plot_area_height = plot_area_width * total_height_units / total_width_units
    fig_height = plot_area_height / (gs_top - gs_bottom)

    fig, axes = plt.subplots(1, ncols, figsize=(fig_width, fig_height), dpi=args.dpi,
                              gridspec_kw={'width_ratios': width_ratios,
                                          'hspace': 0.0, 'wspace': 0.0,
                                          'left': gs_left, 'right': gs_right,
                                          'top': gs_top, 'bottom': gs_bottom})
    if not args.no_colorbar:
        ax1, ax2, ax3, cax = axes
    else:
        ax1, ax2, ax3 = axes

    
    for block in range(len(extents_x)):
        ax1.imshow(quantity_x[block], cmap=args.cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation='none', origin='lower', extent=extents_x[block], aspect='auto')

    for block in range(len(extents_y)):
        ax2.imshow(quantity_y[block], cmap=args.cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation='none', origin='lower', extent=extents_y[block], aspect='auto')

    im3 = None
    for block in range(len(extents_z)):
        im3 = ax3.imshow(quantity_z[block], cmap=args.cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation='none', origin='lower', extent=extents_z[block], aspect='auto')


    if not args.no_colorbar:
        label = args.colorbar_title or LABELS.get(args.variable, args.variable)
        cbar = fig.colorbar(im3, cax=cax)
        cbar.set_label(label)

    if args.grid:
        for x0, x1, y0, y1 in extents_x:
            ax1.add_patch(matplotlib.patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, facecolor='none', edgecolor='gray',
                linewidth=0.5, alpha=0.5))

        for x0, x1, y0, y1 in extents_y:
            ax2.add_patch(matplotlib.patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, facecolor='none', edgecolor='gray',
                linewidth=0.5, alpha=0.5))

        for x0, x1, y0, y1 in extents_z:
            ax3.add_patch(matplotlib.patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, facecolor='none', edgecolor='gray',
                linewidth=0.5, alpha=0.5))

    if args.binary:
        try:
            p            = input_data_x['problem']
            a_binary     = float(p['a_binary'])
            eccentricity = float(p['eccentricity'])
            q_mass       = float(p['q_mass'])
            m_total      = float(p['m_total'])
            primary, secondary = binary_positions(time_x, a_binary, eccentricity,q_mass, m_total)
        except KeyError:
            raise RuntimeError('--binary needs a_binary/eccentricity/q_mass/m_total in '
                               'the input file (cbdiso_3d-style problems only).')
        primary_track, secondary_track = binary_orbit_track(a_binary, eccentricity, q_mass)
        m1        = m_total / (1.0 + q_mass)
        m2        = m_total * q_mass / (1.0 + q_mass)
        m_max     = max(m1, m2)
        sizes     = [args.binary_size * m1 / m_max, args.binary_size * m2 / m_max]
        axes_inds = {'x': (1, 2), 'y': (0, 2), 'z': (0, 1)}

        for ax, d in zip((ax1, ax2, ax3), ('x', 'y', 'z')):
            i0, i1 = axes_inds[d]
            ax.plot(primary_track[:, i0], primary_track[:, i1], '--',
                   color=args.binary_color, linewidth=0.6, alpha=0.4, zorder=4)
            ax.plot(secondary_track[:, i0], secondary_track[:, i1], '--',
                   color=args.binary_color, linewidth=0.6, alpha=0.4, zorder=4)
            xs = [primary[i0], secondary[i0]]
            ys = [primary[i1], secondary[i1]]
            ax.scatter(xs, ys, s=sizes, c=args.binary_color, edgecolors=args.binary_edgecolor,linewidths=0.35, zorder=5)

    ax2.set_yticks([]); ax3.set_yticks([])
    ax1.text(-1.0,3.25, r'YZ plane',bbox = dict(boxstyle='round', facecolor='white', alpha=0.75))
    ax2.text(-1.0,3.25, r'XZ plane',bbox = dict(boxstyle='round', facecolor='white', alpha=0.75))
    ax3.text(-1.0,3.25, r'XY plane',bbox = dict(boxstyle='round', facecolor='white', alpha=0.75))
    
    for ax, d in zip((ax1, ax2, ax3), ('x', 'y', 'z')):
        ax.set_xlim(panel_xlim[d])
        ax.set_ylim(panel_ylim[d])

    
    for ax in (ax1, ax2):
        xmin, xmax = ax.get_xlim()
        tol = 0.01 * (xmax - xmin)
        ax.set_xticks([t for t in ax.get_xticks() if abs(t - xmax) > tol])
        ax.set_xlim(xmin, xmax)

    if args.hide_yaxis:
        for ax in (ax2, ax3):
            ax.tick_params(labelleft=False)

    if args.output_dir == '.':
        directory = os.getcwd()
    else:
        directory = args.output_dir
        
    filename = directory + '/panels.png'
    print('saving at...', filename)
    plt.savefig(filename, dpi=args.dpi)


if __name__ == '__main__':
    main()
