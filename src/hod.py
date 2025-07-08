"""
Run all the parts of an HOD model 
"""
import numpy as np
from numba import jit
import time

import src.hod_io as io
import src.hod_const as c
import src.hod_r_profile as rp
import src.hod_v_profile as vp
from src.hod_shape import calculate_hod_occupation

@jit(nopython=True)
def process_halos_chunk(x, y, z, vx, vy, vz, logM, halo_ids, random_seeds,
                        hodshape, mu, Ac, As,alpha, sig, gamma,vfact, beta,
                        K, vt, vtdisp, M0, M1, omega_M, Lbox, zsnap):
    """
    Numba-optimized function to process a chunk of halos with dynamic memory allocation
    
    Parameters:
    -----------
    outfile : string
        Path to output file
    x, y, z : array_like
        Halo positions
    vx, vy, vz : array_like
        Halo velocities  
    logM : array_like
        Log10 halo masses
    halo_ids : array_like
        Halo IDs
    random_seeds : array_like
        Random seeds for each halo
    mu, Ac, As, alpha, sig, gamma, vfact, beta, K, vt, vtdisp, M0, M1, Lbox : float
        HOD parameters
    hodshape : int
        Average HOD shape
        
    Returns:
    --------
    galaxies : ndarray
        Array of galaxy properties with shape (n_galaxies, 16)
        Columns: x, y, z, vx, vy, vz, M, Nsat, Dvx, Dvy, Dvz, Dx, Dy, Dz, halo_id, is_central
    """
    n_halos = len(x)
    
    # Start with a reasonable initial estimate but allow growth
    initial_estimate = n_halos * 50 
    max_galaxies = initial_estimate
    
    # Pre-allocate output arrays
    galaxies = np.zeros((max_galaxies, 16))
    galaxy_count = 0
    
    for i in range(n_halos):
        # Extract halo properties
        halo_x, halo_y, halo_z = x[i], y[i], z[i]
        halo_vx, halo_vy, halo_vz = vx[i], vy[i], vz[i]
        halo_logM = logM[i]
        halo_id = halo_ids[i]
        
        # Set random seed for this halo
        np.random.seed(int(random_seeds[i]))
        
        # Convert to linear mass
        M = 10.0**halo_logM

        # Generate central and satellite counts
        Ncen, Nsat = calculate_hod_occupation(M, mu, Ac, As, alpha,
                                              sig, gamma, M0, M1, hodshape)

        # Check if we need to grow the array before processing this halo
        galaxies_needed = Ncen + Nsat
        if galaxy_count + galaxies_needed >= max_galaxies:
            # Grow array by doubling size or adding what we need, whichever is larger
            new_size = max(max_galaxies * 2, galaxy_count + galaxies_needed + 1000)
            new_galaxies = np.zeros((new_size, 16))
            new_galaxies[:galaxy_count, :] = galaxies[:galaxy_count, :]
            galaxies = new_galaxies
            max_galaxies = new_size
        
        # Write central galaxy if present
        if Ncen == 1:
            galaxies[galaxy_count, 0] = halo_x
            galaxies[galaxy_count, 1] = halo_y
            galaxies[galaxy_count, 2] = halo_z
            galaxies[galaxy_count, 3] = halo_vx
            galaxies[galaxy_count, 4] = halo_vy
            galaxies[galaxy_count, 5] = halo_vz
            galaxies[galaxy_count, 6] = M
            galaxies[galaxy_count, 7] = Nsat
            galaxies[galaxy_count, 8] = 0.0  # Dvx (no offset for central)
            galaxies[galaxy_count, 9] = 0.0  # Dvy
            galaxies[galaxy_count, 10] = 0.0  # Dvz
            galaxies[galaxy_count, 11] = 0.0  # Dx
            galaxies[galaxy_count, 12] = 0.0  # Dy
            galaxies[galaxy_count, 13] = 0.0  # Dz
            galaxies[galaxy_count, 14] = halo_id
            galaxies[galaxy_count, 15] = 1.0  # is_central
            galaxy_count += 1

        # Generate satellite galaxies
        for j in range(Nsat):
            # Generate position and velocity offsets
            Dx, Dy, Dz = rp.generate_nfw_position(M, K, zsnap, omega_M)
            Dvx, Dvy, Dvz = vp.generate_virial_velocity(M, zsnap, omega_M)
            # Apply periodic boundary conditions
            xgal = (halo_x + Dx) % Lbox
            ygal = (halo_y + Dy) % Lbox
            zgal = (halo_z + Dz) % Lbox
            
            # Store satellite galaxy
            galaxies[galaxy_count, 0] = xgal
            galaxies[galaxy_count, 1] = ygal
            galaxies[galaxy_count, 2] = zgal
            galaxies[galaxy_count, 3] = halo_vx + Dvx
            galaxies[galaxy_count, 4] = halo_vy + Dvy
            galaxies[galaxy_count, 5] = halo_vz + Dvz
            galaxies[galaxy_count, 6] = M
            galaxies[galaxy_count, 7] = Nsat
            galaxies[galaxy_count, 8] = Dvx
            galaxies[galaxy_count, 9] = Dvy
            galaxies[galaxy_count, 10] = Dvz
            galaxies[galaxy_count, 11] = Dx
            galaxies[galaxy_count, 12] = Dy
            galaxies[galaxy_count, 13] = Dz
            galaxies[galaxy_count, 14] = halo_id
            galaxies[galaxy_count, 15] = 0.0  # is_central
            galaxy_count += 1
    
    # Return only the filled portion of the array
    return galaxies[:galaxy_count, :]


def process_halo_file_chunked(params, chunk_size=c.chunk_size, verbose=False):
    """
    Process halo file in chunks with Numba optimization
    """
    start_time = time.time()
    line_count = 0
    galaxy_count = 0

    try:
        with open(params.outfile, 'w') as f_out:
            # Process file in chunks
            for chunk_data in io.read_halo_data_chunked(params.infile, params.ftype, chunk_size):
                if len(chunk_data) == 0:
                    continue
                
                # Extract arrays from chunk
                x = chunk_data[:, 0]
                y = chunk_data[:, 1]
                z = chunk_data[:, 2]
                vx = chunk_data[:, 3]
                vy = chunk_data[:, 4]
                vz = chunk_data[:, 5]
                logM = chunk_data[:, 6]
                halo_ids = chunk_data[:, 7].astype(np.int64)

                # Generate random seeds for each halo
                random_seeds = np.random.randint(0, 2**31, size=len(x))
                
                # Process chunk with Numba
                galaxies = process_halos_chunk(
                    x, y, z, vx, vy, vz, logM, halo_ids, random_seeds,
                    params.hodshape, params.mu, params.Ac, params.As,
                    params.alpha, params.sig, params.gamma,
                    params.vfact, params.beta, params.K, params.vt, params.vtdisp,
                    params.M0, params.M1,
                    params.omega_M, params.Lbox, params.zsnap,
                )

                # Write results to file
                io.write_galaxies_to_file(galaxies, f_out, True)

                line_count += len(chunk_data)
                galaxy_count += len(galaxies)
                
                # Progress report
                if line_count % c.report_after_nlines == 0:
                    elapsed = time.time() - start_time
                    rate = line_count / elapsed
                    if verbose:
                        print(f"Processed {line_count} halos, {galaxy_count} galaxies, rate: {rate:.1f} halos/sec")
        
        elapsed = time.time() - start_time
        rate = line_count / elapsed
        if verbose:
            print(f"Completed {line_count} halos, {galaxy_count} galaxies in {elapsed:.2f}s, rate: {rate:.1f} halos/sec")
        return line_count
        
    except Exception as e:
        print(f"ERROR: {e}")
        return None



def run_hod_model(params,verbose=False):
    """
    Run HOD model with automatic backend selection for optimal performance
    
    Parameters:
    -----------
    params : dict
        Dictionary containing the parameters for the HOD model
        
    Returns:
    --------
    int : 0 for success, non-zero for failure
    """
    sep = io.line_separator()
    print(sep+"\nHOD Galaxy Mock Generation\n"+sep)

    # Validate parameters
    if not io.validate_parameters(params):
        print("Parameter validation failed. Please check your parameters.")
        print_parameter_info()
        return -1

    # Initialize random seed and time 
    np.random.seed(params.seed)
    start_time = time.time()

    nhalos = process_halo_file_chunked(params,verbose=verbose)
    if nhalos is None:
        print(f"ERROR: Processing failed")
        return -1
    
    # End performance monitoring
    total_time = time.time() - start_time
    
    if verbose:
        io.print_run_summary(params)
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Processing rate: {nhalos/total_time:.1f} halos/second")
    print(f"SUCCESS: Completed processing {nhalos} halos")
    
    return 0
