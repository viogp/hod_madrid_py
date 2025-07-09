import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import gridspec
import re
from glob import glob
from typing import Union, List, Optional, Tuple, Dict
import warnings

import src.hod_io as io

def get_distinct_colors(n_colors):
    """Generate distinct colors for plotting"""
    import matplotlib.cm as cm
    colors = cm.tab10(np.linspace(0, 1, min(n_colors, 10)))
    if n_colors > 10:
        colors2 = cm.Set3(np.linspace(0, 1, n_colors - 10))
        colors = np.vstack([colors, colors2])
    return colors


def plot_satellite_radial_distribution(
    mock_file: str,
    output_dir: str = "output/plots",
    rmin: float = -3.0,
    rmax: float = 2.0,
    dr: float = 0.05,
    xmin: float = -2.0,
    xmax: float = 0.7,
    ymin: float = -5.0,
    ymax: float = -0.4,
    testing: bool = False,
    verbose: bool = False,
    plot_title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> Dict:
    """
    Plot satellite radial distribution as function of distance from halo center.
    
    This function processes mock galaxy catalogs and plots the normalized number
    of satellite galaxies as a function of their radial distance from halo centers.
    
    Parameters:
    -----------
    mock_file : str  
        Path to mock file. If contains '_subvol' followed by a number,
        will automatically find and process all subvolumes. Expected format:
        mock000_subvolN.dat or similar
    output_dir : str, default="."
        Directory to save output plots
    rmin, rmax : float, default=-3.0, 2.0
        Log10 range for radial distance bins (Mpc/h)
    dr : float, default=0.05
        Bin width for radial distance
    xmin, xmax, ymin, ymax : float
        Plot axis limits
    testing : bool, default=False
        If True, only processes a subset of boxes for testing
    plot_title : str, optional
        Custom title for the plot
    figsize : tuple, default=(10, 8)
        Figure size for the plot
        
    Returns: path to plot
    --------
    """   
    # Setup radial bins
    rbins = np.arange(rmin, rmax, dr)
    rhist = rbins + dr * 0.5

    # Check if mock_file contains subvolume pattern and find all files
    subvol_pattern = r'_subvol(\d+)'
    subvols = re.search(subvol_pattern, mock_file)
    base_name = os.path.basename(mock_file)

    mock_files = []
    if subvols is not None:
        # Extract base pattern and find all subvolume files
        base_pattern = re.sub(subvol_pattern, '_subvol*', mock_file)
        # Replace the specific number with a wildcard pattern
        base_dir = os.path.dirname(mock_file)
        pattern = re.sub(subvol_pattern, '_subvol[0-9]*', base_name)
        
        if base_dir:
            search_pattern = os.path.join(base_dir, pattern)
        else:
            search_pattern = pattern
            
        # Find all matching files
        potential_files = []
        if base_dir:
            for f in os.listdir(base_dir):
                if re.search(re.sub(r'\[0-9\]\*', r'\\d+', pattern), f):
                    potential_files.append(os.path.join(base_dir, f))
        else:
            for f in os.listdir('.'):
                if re.search(re.sub(r'\[0-9\]\*', r'\\d+', pattern), f):
                    potential_files.append(f)
        
        mock_files = sorted(potential_files)
        if not mock_files or testing:
            # Fallback to single file
            mock_files = [mock_file]
            warnings.warn(f"No subvolume files found, using single file: {mock_file}")
    else:
        mock_files = [mock_file]

    if verbose:
        print(f"Radial profile plot with {len(mock_files)} file(s)")

    # Initialize figure
    fig = plt.figure(figsize=figsize)
    xtit = "${\\rm log}_{10}(r\\,h^{-1}{\\rm Mpc})$"
    ytit = "${\\rm log}_{10}(N_{\\rm sat,dex}/N_{\\rm sat, tot})$"
    
    ax = fig.add_subplot(111)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(xtit)
    ax.set_ylabel(ytit)
    
    if plot_title:
        ax.set_title(plot_title)
    
    # Get colors for plotting
    colors = get_distinct_colors(len(mock_files))
        
    # Track global min/max for logging
    global_minr = 999.0
    global_maxr = -999.0
    
    # Process each mock file
    for im, mock in enumerate(mock_files):
        if verbose:
            print(f"Processing file: {mock}")
        io.check_input_file(mock)

        # Extract K value if present in filename
        kval = "unknown"
        if 'K' in mock:
            try:
                kval1 = mock.split('K')[1]
                kval = kval1.split('_')[0]
            except:
                pass

        avgr = 0.0
        ntot = 0
        nsatr = np.zeros(len(rbins))

        # Read mock catalogue
        ldx, ldy, ldz = [], [], []
        try:
            # Load only the columns we need: dx, dy, dz, sat
            data = np.loadtxt(mock, usecols=[11, 12, 13, 7])

            # Filter for satellite galaxies (cen != 1)
            sat_mask = data[:, 3] > 0
            sat_data = data[sat_mask]

            # Extract coordinates
            ldx.extend(sat_data[:, 0])
            ldy.extend(sat_data[:, 1]) 
            ldz.extend(sat_data[:, 2])
    
        except Exception as e:
            print(f"    Error reading {mock}: {e}")
            continue
        
        if not ldx:  # No satellites found
            continue
                        
        dx = np.array(ldx, dtype=float)
        dy = np.array(ldy, dtype=float)
        dz = np.array(ldz, dtype=float)
                    
        # Calculate radial distance for satellite galaxies (in Mpc/h)
        rsat = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        # Average and total number
        avgr += np.sum(rsat)
        ntot += len(rsat)

        # Track global limits
        ind = np.where(rsat > 0.)[0]
        if len(ind) > 0:
            lrsat = np.log10(rsat[ind])
            global_minr = min(global_minr, np.min(lrsat))
            global_maxr = max(global_maxr, np.max(lrsat))
            
            # Create histogram
            H, _ = np.histogram(lrsat, bins=np.append(rbins, rmax))
            nsatr += H
                        
            print(f'    Satellites found: {len(rsat)}')
        
    if ntot > 0:
        ytot = np.sum(nsatr)
        avgr = avgr / ntot
            
        # Plot
        ind = np.where(nsatr > 0)[0]
        if len(ind) > 0:
            label = f'K={kval}'
            ax.plot(rhist[ind], np.log10(nsatr[ind]/ytot), 
                    label=label, color=colors[im])
                
            # Add arrow to show median
            dy = 0.05 * (ymax - ymin)
            ax.arrow(np.log10(avgr), ymax - dy, 0, dy, 
                     color=colors[im], length_includes_head=True,
                     head_width=dy*0.1, head_length=dy*0.3)

            if verbose:
                print(f'  {ntot} satellites, avg_r = {avgr:.3f} Mpc/h')
    
    # Add legend
    if len(mock_files) > 0:
        leg = ax.legend(loc='best')
        leg.draw_frame(False)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plotfile = os.path.join(output_dir, f'rp_sat_{base_name}.png')
    fig.savefig(plotfile, dpi=150, bbox_inches='tight')

    if verbose:
        print(f'Radial profile plot: {plotfile}')
        
    return plotfile



def make_test_plots(params,outroot,hmf_file=None,verbose=False):
    """
    Make test plots on mock galaxy catalogues
    
    Parameters:
    -----------
    params : Tuple
        Parameters used to ran the HOD model
    hmf_file : str
        Path to the Halo Mass Function file (for reference/metadata)
    verbose : bool
        True for progress messages
        
    Returns: 0 or -1 (errors)
    --------
    """
    mock_file = params.outfile
    #mock_file = '/home/violeta/buds/hods/galaxies_1000Mpc_V1.4more_NFW_mu12.000_Ac1.0000_As0.50000_vfact1.00_beta0.000_K1.00_vt0pm0_BVG_product_nosubhalos_trunc_binomialextended.dat'
    
    output_dir = outroot / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    sep = io.line_separator()
    print(sep+"\n Plots: "+str(output_dir)+"\n"+sep)
    
    plotfile = plot_satellite_radial_distribution(
        mock_file=mock_file,output_dir=output_dir,verbose=verbose)

    return 0

