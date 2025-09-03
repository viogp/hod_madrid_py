import numpy as np
from Corrfunc.theory import xi

def extract_positions_from_galaxy_catalog(
    input_catalog_file, # Output file from HOD code
    output_positions_file # New file to store only positions
):
    """ Extracts the x, y, z columns from the galaxy catalog file and saves them as a new file for use with correlation function codes. """
    data = np.loadtxt(input_catalog_file, usecols=(0, 1, 2))
    np.savetxt(output_positions_file, data, fmt="%.6f")
    print(f"Positions saved to: {output_positions_file}")
    return output_positions_file

import numpy as np

def extract_positions_from_galaxy_catalog_rs(
    input_catalog_file,        # Output file from HOD code
    output_positions_file,     # New file to store only positions (in RS)
    pos_key=(0, 1, 2),
    vel_key=(3, 4, 5),
    z_snap=1.321,
    Omega_m=0.3089,
    Omega_L=0.6911,
    h=0.6774,
    los_axis="z",
    verbose=True,
):
    """
    Extrae posiciones y aplica el desplazamiento a espacio de redshift (RSD)
    a lo largo del eje 'los_axis'. Guarda un TXT con 3 columnas (x, y, s_los).

    Nota de unidades: si las velocidades están en km/s y las posiciones en Mpc,
    el término añadido es v_los / H(z). Si quieres el factor 1/(a H) clásico,
    cambia la línea 'shift = v_los / Hz' por 'shift = v_los / (a * Hz)' con
    a = 1/(1+z).
    """
    # Leer de una vez posiciones+velocidades
    usecols = tuple(pos_key) + tuple(vel_key)
    data = np.loadtxt(input_catalog_file, usecols=usecols)

    # Separar bloques
    npos = len(pos_key)
    pos = data[:, :npos]              # (N, 3)
    vel = data[:, npos:npos+len(vel_key)]  # (N, 3)

    x, y, z  = pos[:, 0], pos[:, 1], pos[:, 2]
    vx, vy, vz = vel[:, 0], vel[:, 1], vel[:, 2]

    # H(z)
    H0 = 100.0 * h
    Hz = H0 * np.sqrt(Omega_m * (1.0 + z_snap)**3 + Omega_L)
    if verbose:
        print(f"[RS] z={z_snap:.3f} -> H(z)={Hz:.3f} km/s/Mpc, LOS='{los_axis}'")

    # Desplazamiento LOS (usa 1/Hz; si quieres 1/(a*Hz), descomenta las dos líneas siguientes)
    # a = 1.0 / (1.0 + z_snap)
    # factor = 1.0 / (a * Hz)
    factor = 1.0 / Hz

    if   los_axis.lower() == 'x':
        s = x + vx * factor
        positions = np.column_stack([s, y, z])
    elif los_axis.lower() == 'y':
        s = y + vy * factor
        positions = np.column_stack([x, s, z])
    elif los_axis.lower() == 'z':
        s = z + vz * factor
        positions = np.column_stack([x, y, s])
    else:
        raise ValueError("los_axis must be 'x', 'y' o 'z'.")

    np.savetxt(output_positions_file, positions, fmt="%.6e", delimiter=" ")
    if verbose:
        print(f"Positions saved to: {output_positions_file} "
              f"({positions.shape[0]} rows x {positions.shape[1]} columns)")
    return output_positions_file


def compute_correlation_corrfunc(
    positions_file,
    output_file,
    boxsize,
    rmin,
    rmax,
    n_bins,
    n_threads=4,
    verbose=True
):
    """
    Computes the two-point correlation function ξ(r) using Corrfunc,
    and returns analytical Poisson errors per bin.

    Parameters:
    -----------
    positions_file : str
        Path to a .txt file with three columns (x, y, z) of galaxy positions.
    output_file : str
        Path where the output .txt (three columns: r_center, xi, error) will be saved.
    boxsize : float
        Size of the simulation box (Mpc/h).
    rmin : float
        Minimum separation (Mpc/h) to consider.
    rmax : float
        Maximum separation (Mpc/h) to consider.
    n_bins : int
        Number of logarithmic bins between rmin and rmax.
    n_threads : int, optional
        Number of threads to use in Corrfunc (default: 4).
    verbose : bool, optional
        If True, prints progress messages (default: True).

    Returns:
    --------
    output_file : str
        Path to the saved .txt file with columns [r_center, xi(r), error(r)].
    r_centers, xi_vals, errors : arrays
        Arrays with the bin centers, xi(r), and error per bin.
    """
    if verbose:
        print(f"Loading positions from: {positions_file}")

    # Load positions (assumed to be a plain text file with 3 columns)
    data = np.loadtxt(positions_file)
    x_data, y_data, z_data = data[:, 0], data[:, 1], data[:, 2]

    if verbose:
        print("Generating log-spaced bins...")
    rbins = np.logspace(np.log10(rmin), np.log10(rmax), n_bins + 1)
    r_centers = 0.5 * (rbins[:-1] + rbins[1:])

    if verbose:
        print("Computing ξ(r) with Corrfunc...")
    results = xi(
        boxsize=boxsize,
        nthreads=n_threads,
        binfile=rbins,
        X=x_data, Y=y_data, Z=z_data
    )

    # Extract xi values and npairs (pair counts)
    xi_vals = np.array([b['xi'] for b in results])
    npairs = np.array([b['npairs'] for b in results])

    # Analytical Poisson errors: sigma_xi = (1 + xi) / sqrt(Npairs)
    # Avoid division by zero:
    errors = np.zeros_like(xi_vals)
    mask = npairs > 0
    errors[mask] = (1.0 + xi_vals[mask]) / np.sqrt(npairs[mask])
    errors[~mask] = 0.0

    # Stack r_centers, xi values, and errors into three columns
    output_data = np.column_stack((r_centers, xi_vals, errors))

    if verbose:
        print(f"Saving correlation to: {output_file}")
    header = "#r_center[Mpc/h], #xi(r), #err_analytical"
    np.savetxt(output_file, output_data, delimiter=",", header=header, comments='')

    if verbose:
        print("Correlation computation complete.")
    return output_file, r_centers, xi_vals, errors