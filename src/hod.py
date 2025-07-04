#!/usr/bin/env python3
"""
Main script for HOD_NFW simulation
Variables are defined within the script, similar to run_txtinput_tutorial.py style
"""

import sys
import jax.numpy as jnp
import jax.random as random
from jax import random as jrand
import numpy as np
from src.hod_core import HODCore

def main():
    # =====================================================
    # PARAMETER DEFINITION SECTION
    # =====================================================
    
    # HOD Model Parameters
    mu = 12.0        # Halo mass threshold parameter
    Ac = 1.0         # Central galaxy amplitude
    As = 0.5         # Satellite galaxy amplitude
    vfact = 1.0      # Velocity factor (alpha_v)
    beta = 0.0       # Scatter parameter (Poisson=0, nearest int=-2)
    K = 1.0          # NFW truncation parameter
    vt = 0.0         # Tangential velocity (=0 or =500)
    vtdisp = 0.0     # Tangential velocity dispersion (=0 or =200)
    
    # Cosmological Parameters
    zsnap = 0.8594   # Snapshot redshift
    OMEGA_M = 0.3089 # Matter density parameter (UNITSIM)
    
    # Simulation Box Parameters
    LBOX = 1000.0    # Box size in Mpc/h
    
    # HOD Model Selection and Parameters
    version = "1.4"
    
    # Define HOD parameters based on version
    # (These can be modified based on the desired HOD model)
    if version == "1.41":
        # Old Version
        M0_factor = 0.0    # mu offset for M0
        M1_factor = 1.3    # mu offset for M1
        alpha = 1.0        # Satellite power law slope
        sig = 0.15         # Central galaxy scatter
        gamma = None       # Not used in this version
    elif version == "1.42":
        # Version 1.42
        M0_factor = -0.1   # mu offset for M0
        M1_factor = 0.3    # mu offset for M1
        alpha = 0.8        # Satellite power law slope
        sig = 0.12         # Central galaxy scatter
        gamma = None       # Not used in this version
    else:  # version == "1.4" (default)
        # Current version
        M0_factor = -0.05  # mu offset for M0
        M1_factor = 0.35   # mu offset for M1
        alpha = 0.9        # Satellite power law slope
        sig = 0.08         # Central galaxy scatter
        gamma = -1.4       # Power law slope for high mass centrals
    
    # Calculate derived parameters
    M0 = 10**(mu + M0_factor)
    M1 = 10**(mu + M1_factor)
    
    # =====================================================
    # FILE PATH CONFIGURATION
    # =====================================================
    
    # Output format selection
    MORE = True  # Set to False for simpler output format
    
    # Define base paths
    if MORE:
        outbase = ("output/"
                  "galaxies_1000Mpc_V%smore_NFW_mu%.3f_Ac%.4f_As%.5f_"
                  "vfact%.2f_beta%.3f_K%.2f_vt%.0fpm%.0f_"
                  "BVG_product_nosubhalos_trunc_binomialextended.dat")
    else:
        outbase = ("output/"
                  "galaxies_1000Mpc_V%s_NFW_mu%.3f_Ac%.4f_As%.5f_"
                  "vfact%.2f_beta%.3f_K%.2f_vt%.0fpm%.0f_"
                  "BVG_product_nosubhalos_trunc_binomialextended.dat")
    
    # Input file path
    inbase = ("data/example/UNIT_haloes_logMmin13.500_logMmax14.500.txt")
    
    # Generate full file names
    output_file = outbase % (version, mu, Ac, As, vfact, beta, K, vt, vtdisp)
    input_file = inbase
    
    # Alternative file paths (uncomment and modify as needed)
    # input_file = "path/to/your/halo/catalog.txt"
    # output_file = "path/to/your/output/galaxies.dat"
    
    # =====================================================
    # RANDOM SEED CONFIGURATION
    # =====================================================
    
    seed = 42  # Random seed for reproducibility
    key = jrand.PRNGKey(seed)
    
    # =====================================================
    # EXECUTION SECTION
    # =====================================================
    
    # Print run information
    print("=" * 60)
    print("HOD_NFW Galaxy Generation")
    print("=" * 60)
    print(f"Version: {version}")
    print(f"Parameters:")
    print(f"  mu = {mu:.3f}")
    print(f"  Ac = {Ac:.4f}")
    print(f"  As = {As:.5f}")
    print(f"  vfact = {vfact:.2f}")
    print(f"  beta = {beta:.3f}")
    print(f"  K = {K:.2f}")
    print(f"  vt = {vt:.0f}")
    print(f"  vtdisp = {vtdisp:.0f}")
    print(f"Derived parameters:")
    print(f"  M0 = {M0:.6e}")
    print(f"  M1 = {M1:.6e}")
    print(f"  alpha = {alpha:.1f}")
    print(f"  sig = {sig:.2f}")
    if gamma is not None:
        print(f"  gamma = {gamma:.1f}")
    print(f"Cosmology:")
    print(f"  zsnap = {zsnap:.4f}")
    print(f"  OMEGA_M = {OMEGA_M:.4f}")
    print(f"  LBOX = {LBOX:.1f} Mpc/h")
    print("=" * 60)
    
    # Count lines in input file
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
            # Skip header lines starting with #
            data_lines = [line for line in lines if not line.strip().startswith('#')]
            nhalos = len(data_lines)
    except FileNotFoundError:
        print(f"ERROR: Could not open input file {input_file}")
        print("Please check the file path and try again.")
        return -1
    
    print(f"Found N={nhalos} halos in input file:")
    print(f"  {input_file}")
    print(f"Writing output to:")
    print(f"  {output_file}")
    print("=" * 60)
    
    # Initialize HOD core
    hod_core = HODCore(
        zsnap=zsnap,
        LBOX=LBOX,
        OMEGA_M=OMEGA_M,
        mu=mu,
        Ac=Ac,
        As=As,
        vfact=vfact,
        beta=beta,
        K=K,
        vt=vt,
        vtdisp=vtdisp,
        M0=M0,
        M1=M1,
        alpha=alpha,
        sig=sig,
        gamma=gamma if gamma is not None else -1.4,  # Default value
        MORE=MORE
    )
    
    print(f"Starting HOD with M1={M1:.6e}")
    
    # Process halos and generate galaxies
    try:
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            # Skip header if present
            first_line = f_in.readline().strip()
            if first_line.startswith('#'):
                first_line = f_in.readline().strip()
            
            # Process first line
            if first_line:
                hod_core.process_halo_line(first_line, f_out, key)
                key = jrand.split(key)[0]  # Update key for next iteration
            
            # Process remaining lines with progress indicator
            line_count = 1
            for line in f_in:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, subkey = jrand.split(key)
                    hod_core.process_halo_line(line, f_out, subkey)
                    line_count += 1
                    
                    # Print progress every 10000 halos
                    if line_count % 10000 == 0:
                        print(f"Processed {line_count}/{nhalos} halos...")
    
    except FileNotFoundError:
        print(f"ERROR: Could not open input file {input_file}")
        return -1
    except Exception as e:
        print(f"ERROR: Error processing files: {e}")
        return -1
    
    print("=" * 60)
    print(f"SUCCESS: Completed processing {nhalos} halos")
    print(f"Output written to: {output_file}")
    print("=" * 60)
    
    return 0

def print_parameter_info():
    """Print information about the parameters that can be modified"""
    print("""
    PARAMETER MODIFICATION GUIDE:
    ============================
    
    HOD Model Parameters:
    - mu: Log10 of characteristic halo mass (affects both centrals and satellites)
    - Ac: Amplitude for central galaxies (0-1, typically ~1)
    - As: Amplitude for satellite galaxies (affects satellite abundance)
    - alpha: Power law slope for satellite HOD (typically 0.8-1.2)
    - sig: Scatter in central galaxy occupation (log-normal width)
    - gamma: Power law slope for high-mass centrals (version 1.4 only)
    
    Velocity Parameters:
    - vfact: Velocity scaling factor for satellites (typically 0.5-1.5)
    - vt: Systematic tangential velocity (km/s, often 0 or ~500)
    - vtdisp: Random tangential velocity dispersion (km/s, often 0 or ~200)
    
    Model Parameters:
    - beta: Scatter model (0=Poisson, >0=negative binomial, <0=binomial)
    - K: NFW truncation parameter (typically ~1)
    
    Cosmological Parameters:
    - zsnap: Redshift of the snapshot
    - OMEGA_M: Matter density parameter
    - LBOX: Simulation box size (Mpc/h)
    
    To modify parameters, edit the values in the "PARAMETER DEFINITION SECTION"
    of the main() function.
    """)

if __name__ == "__main__":
    # Check if user wants parameter information
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', '--params']:
        print_parameter_info()
        sys.exit(0)
    
    # Run the main function
    sys.exit(main())
