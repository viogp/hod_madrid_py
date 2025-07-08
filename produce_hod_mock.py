#!/usr/bin/env python3
"""
Produce model galaxies from HOD models from
Avila+2020, Vos Gines+2024
"""

from pathlib import Path
from src.hod import run_hod_model
from src.hod_io import create_hod_params

def main():
    # =====================================================
    # PARAMETER DEFINITION SECTION
    # =====================================================
    verbose   = True   # Set to True for progress messages
    seed      = 42     # Random seed for reproducibility
    
    # Cosmological and simulation parameters
    zsnap = 0.8594   # Snapshot redshift
    omega_M = 0.3089 # Matter density parameter    
    Lbox = 1000.0    # Box size in Mpc/h
    
    # HOD Model Parameters
    hodshape = 1     # Shape of the average HOD: 1, 2 or 3
    mu = 12.0        # Log10 of characteristic halo mass
    Ac = 1.0         # Central galaxy amplitude
    As = 0.5         # Satellite galaxy amplitude
    vfact = 1.0      # Velocity factor (alpha_v)
    beta = 0.0       # PDF parameter (Poisson=0, nearest int=-2)
    K = 1.0          # NFW truncation parameter
    vt = 0.0         # Tangential velocity (km/s, =0 or =500)
    vtdisp = 0.0     # Tangential velocity dispersion (km/s, =0 or =200)
    alpha = 0.9      # Satellite power law slope
    sig = 0.08       # Central galaxy scatter (HOD2 and 3)
    gamma = -1.4     # Power law slope for high mass centrals (HOD3)

    # Calculate derived parameters
    M0_factor = -0.05  # mu offset for M0
    M1_factor = 0.35   # mu offset for M1
    M0 = 10**(mu + M0_factor)
    M1 = 10**(mu + M1_factor)
    
    # =====================================================
    # PATHS to files
    # =====================================================
    ftype = 'txt' # Possible input file type: txt
    output_dir = Path("output")

    # Default input file (modify as needed)
    input_dir = Path("data/example")
    infile = input_dir/"UNIT_haloes_logMmin13.500_logMmax14.500.txt"
        
    # =====================================================
    # Collect parameters into a dictionary
    # =====================================================
    hod_params = create_hod_params(infile,output_dir,ftype='txt',seed=seed,
                                   zsnap=zsnap, Lbox=Lbox, omega_M=omega_M,
                                   hodshape=hodshape, mu=mu, Ac=Ac, As=As, vfact=vfact,
                                   beta=beta, K=K, vt=vt, vtdisp=vtdisp,
                                   M0=M0, M1=M1, alpha=alpha, sig=sig, gamma=gamma)

    # =====================================================
    # Run the HOD model 
    # =====================================================
    if hod_params is not None:
        result = run_hod_model(hod_params,verbose=verbose)
    
    return result


if __name__ == "__main__":    
    # Run the main function
    exit_code = main()
    __import__('sys').exit(exit_code)
