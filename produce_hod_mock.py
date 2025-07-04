#!/usr/bin/env python3
"""
Main script for producing HOD galaxy mocks
Based on HOD_NFW_V14.c translated to JAX Python

This script defines all parameters and calls the core HOD functions
to generate galaxy catalogs from halo catalogs.
"""

from pathlib import Path
from src.hod_model import run_hod_model

def main():
    # =====================================================
    # PARAMETER DEFINITION SECTION
    # =====================================================
    
    # Control parameters
    verbose = True   # Set to True for detailed parameter info
    seed = 42        # Random seed for reproducibility

    # Cosmological Parameters
    zsnap = 0.8594   # Snapshot redshift
    omega_M = 0.3089 # Matter density parameter
    
    # Simulation Box Parameters
    Lbox = 1000.0    # Box size in Mpc/h
    
    # HOD Model Parameters
    mu = 12.0        # Log10 of characteristic halo mass
    Ac = 1.0         # Central galaxy amplitude
    As = 0.5         # Satellite galaxy amplitude
    vfact = 1.0      # Velocity factor (alpha_v)
    beta = 0.0       # Scatter parameter (Poisson=0, nearest int=-2)
    K = 1.0          # NFW truncation parameter
    vt = 0.0         # Tangential velocity (km/s, =0 or =500)
    vtdisp = 0.0     # Tangential velocity dispersion (km/s, =0 or =200)
    alpha = 0.9      # Satellite power law slope
    sig = 0.08       # Central galaxy scatter (HOD-2 and 3)
    gamma = -1.4     # Power law slope for high mass centrals (HOD-3)

    # Calculate derived parameters
    M0_factor = -0.05  # mu offset for M0
    M1_factor = 0.35   # mu offset for M1
    M0 = 10**(mu + M0_factor)
    M1 = 10**(mu + M1_factor)
    
    # =====================================================
    # FILE PATH CONFIGURATION
    # =====================================================
    
    # Define base directory paths
    base_output_dir = Path("output")
    base_input_dir = Path("data/example")

    # Default input file (modify as needed)
    input_filename = "UNIT_haloes_logMmin13.500_logMmax14.500.txt"
    
    # Output format selection
    MORE = True  # Set to False for simpler output format
    
    # Define file naming patterns
    if MORE:
        output_filename = (f"galaxies_{Lbox:.0f}Mpc_more_NFW_"
                          f"mu{mu:.3f}_Ac{Ac:.4f}_As{As:.5f}_"
                          f"vfact{vfact:.2f}_beta{beta:.3f}_K{K:.2f}_"
                          f"vt{vt:.0f}pm{vtdisp:.0f}_"
                          f"BVG_product_nosubhalos_trunc_binomialextended.dat")
    else:
        output_filename = (f"galaxies_1000Mpc_NFW_"
                          f"mu{mu:.3f}_Ac{Ac:.4f}_As{As:.5f}_"
                          f"vfact{vfact:.2f}_beta{beta:.3f}_K{K:.2f}_"
                          f"vt{vt:.0f}pm{vtdisp:.0f}_"
                          f"BVG_product_nosubhalos_trunc_binomialextended.dat")
    
    
    # Full paths
    output_file = base_output_dir / output_filename
    input_file = base_input_dir / input_filename
    
    # =====================================================
    # PARAMETER COLLECTION
    # =====================================================
    
    # Collect all parameters into a dictionary for easy passing
    hod_params = {
        # Control parameters
        'verbose': verbose,
        'seed': seed,
        'MORE': MORE,
        
        # File paths
        'input_file': input_file,
        'output_file': output_file,
        
        # Cosmological parameters
        'zsnap': zsnap,
        'omega_M': omega_M,
        'Lbox': Lbox,
        
        # HOD model parameters
        'mu': mu,
        'Ac': Ac,
        'As': As,
        'vfact': vfact,
        'beta': beta,
        'K': K,
        'vt': vt,
        'vtdisp': vtdisp,
        'alpha': alpha,
        'sig': sig,
        'gamma': gamma if gamma is not None else -1.4,  # Default value
        
        # Derived HOD parameters
        'M0': M0,
        'M1': M1,
        
        # Physical constants
        'rho_crit': 27.755e10
    }
    
    # =====================================================
    # Run the HOD model
    # =====================================================
    result = run_hod_model(hod_params)
        
    return result

def print_usage():
    """Print usage information"""
    print("""
    HOD Mock Generation Script
    =========================
    
    Usage:
        python produce_hod_mock.py
    
    This script generates galaxy catalogues from
    halo catalogues using Halo Occupation Distribution (HOD) models.
    
    To modify parameters:
    - Edit the values in the "PARAMETER DEFINITION SECTION" of this file
    - Adjust file paths in the "FILE PATH CONFIGURATION" section
    
    Text input file format:
    - Space-separated columns: x y z vx vy vz logM halo_id
    - Lines starting with '#' are treated as comments
    
    Output file format:
    - Galaxy positions, velocities, halo mass, and satellite count
    - Additional columns if MORE=True for detailed analysis
    
    For more information about parameters, set verbose=True in the script.
    """)

if __name__ == "__main__":
    # Check if user wants help
    if len(__import__('sys').argv) > 1 and __import__('sys').argv[1] in ['--help', '-h']:
        print_usage()
        __import__('sys').exit(0)
    
    # Run the main function
    exit_code = main()
    __import__('sys').exit(exit_code)
