"""
Core HOD simulation function that orchestrates the galaxy generation process
"""

import sys
import jax.numpy as jnp
import jax.random as random
from jax import random as jrand

import src.hod_io as io
from  src.hod_shape import HOD_powerlaw,HOD_gaussPL
import src.hod_radial_profile as rp
import src.hod_v_profile as vp
from src.hod_pdf import rand_gauss

# =====================================================
# MAIN PROCESSING FUNCTION
# =====================================================
def process_halo_line(line: str, f_out, key, params: io.HODParams, MORE: bool = True):
    """Process a single halo line and write galaxy outputs"""
    # Parse input line
    parts = line.strip().split()

    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
    vx, vy, vz = float(parts[3]), float(parts[4]), float(parts[5])
    logM = float(parts[6])
    halo_id = int(float(parts[7]))

    M = 10**logM

    # Generate satellite and central galaxies
    key, subkey1 = random.split(key)
    key, subkey2 = random.split(key)

    Nsat = HOD_powerlaw(subkey1, M, params)
    Ncent = HOD_gaussPL(subkey2, logM, params)

    # Write central galaxy if present
    if Ncent == 1:
        if MORE:
            f_out.write(f"{x:.5f} {y:.5f} {z:.5f} {vx:.5f} {vy:.5f} {vz:.5f} "
                       f"{M:.6e} {Nsat:d} 0.0 0.0 0.0 0.0 0.0 0.0 {halo_id:d}\n")
        else:
            f_out.write(f"{x:.5f} {y:.5f} {z:.5f} {vx:.5f} {vy:.5f} {vz:.5f} "
                       f"{M:.6e} {Nsat:d}\n")
    
    # Generate satellite galaxies
    for j in range(Nsat):
        key, subkey = random.split(key)
        
        # Generate position offset
        Dx, Dy, Dz = rp.NFW_to_pos(subkey, M, params)
        
        # Generate velocity offset
        key, subkey = random.split(key)
        Dvx, Dvy, Dvz = vp.vir_to_vel(subkey, M, params)
        
        # Add tangential velocity component
        key, subkey = random.split(key)
        vtrand = rand_gauss(subkey) * params.vtdisp + params.vt

        Dr = jnp.sqrt(Dx**2 + Dy**2 + Dz**2)
        Dr = jnp.maximum(Dr, 1e-10)  # Avoid division by zero
        
        ux = -Dx / Dr
        uy = -Dy / Dr
        uz = -Dz / Dr
        
        Dvx = params.vfact * Dvx + ux * vtrand
        Dvy = params.vfact * Dvy + uy * vtrand
        Dvz = params.vfact * Dvz + uz * vtrand
        
        # Apply periodic boundary conditions
        xgal = x + Dx
        ygal = y + Dy
        zgal = z + Dz
        
        xgal = xgal % params.Lbox
        ygal = ygal % params.Lbox
        zgal = zgal % params.Lbox
        
        # Write satellite galaxy
        if MORE:
            f_out.write(f"{xgal:.5f} {ygal:.5f} {zgal:.5f} "
                       f"{vx + Dvx:.5f} {vy + Dvy:.5f} {vz + Dvz:.5f} "
                       f"{M:.6e} {Nsat:d} {Dvx:.4f} {Dvy:.4f} {Dvz:.4f} "
                       f"{Dx:.4f} {Dy:.4f} {Dz:.4f} {halo_id:d}\n")
        else:
            f_out.write(f"{xgal:.5f} {ygal:.5f} {zgal:.5f} "
                       f"{vx + Dvx:.5f} {vy + Dvy:.5f} {vz + Dvz:.5f} "
                       f"{M:.6e} {Nsat:d}\n")


# =====================================================
# Test and run the HOD model
# =====================================================
def run_hod_model(params):
    """
    Run the HOD model
    
    Parameters:
    -----------
    params : dict
        Dictionary containing all simulation parameters
        
    Returns:
    --------
    int : 0 for success, non-zero for failure
    """
    sep = io.line_separator()
    print(sep+"\nHOD Galaxy Mock Generation\n"+sep)

    # Print parameter information if requested
    if params.get('verbose', False):
        io.print_parameter_info()
    
    # Print run summary
    io.print_run_summary(params)

    # Validate parameters
    if not io.validate_parameters(params):
        print("Parameter validation failed. Please check your parameters.")
        return -1

    # Check input file
    if not io.check_input_file(params['input_file']):
        return -1

    # Check output file
    if not io.check_output_file(params['output_file']):
        return -1

    # Count halos in input file
    nhalos = io.count_lines(params['input_file'])
    if nhalos <= 0:
        return -1

    # Print file information
    io.print_file_info(params['input_file'], nhalos)
    print(sep)

    # Initialize random seed
    key = jrand.PRNGKey(params['seed'])

    # Create HOD parameters structure
    hod_params = io.create_hod_params(
        zsnap=params['zsnap'],
        Lbox=params['Lbox'],
        omega_M=params['omega_M'],
        mu=params['mu'],
        Ac=params['Ac'],
        As=params['As'],
        vfact=params['vfact'],
        beta=params['beta'],
        K=params['K'],
        vt=params['vt'],
        vtdisp=params['vtdisp'],
        M0=params['M0'],
        M1=params['M1'],
        alpha=params['alpha'],
        sig=params['sig'],
        gamma=params['gamma']
    )
    
    print(f"Starting HOD with M1={params['M1']:.6e}")
    print("Processing halos...")

    # Process halos and generate galaxies
    try:
        with open(params['input_file'], 'r') as f_in, open(params['output_file'], 'w') as f_out:
            # Skip header if present
            first_line = f_in.readline().strip()
            if first_line.startswith('#'):
                first_line = f_in.readline().strip()
            
            # Process first line
            if first_line:
                process_halo_line(first_line, f_out, key, hod_params, params['MORE'])
                key = jrand.split(key)[0]  # Update key for next iteration
            
            # Process remaining lines with progress indicator
            line_count = 1
            for line in f_in:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, subkey = jrand.split(key)
                    process_halo_line(line, f_out, subkey, hod_params, params['MORE'])
                    line_count += 1
                    
                    # Print progress every 10000 halos
                    if line_count % 10000 == 0:
                        progress = (line_count / nhalos) * 100
                        print(f"Processed {line_count}/{nhalos} halos ({progress:.1f}%)...")
                        
    except FileNotFoundError:
        print(f"ERROR: Could not open input file {params['input_file']}")
        return -1
    except PermissionError:
        print(f"ERROR: Permission denied when writing to {params['output_file']}")
        return -1
    except Exception as e:
        print(f"ERROR: Error processing files: {e}")
        return -1
    
    print(f"SUCCESS: Completed processing {nhalos} halos")
    return 0
