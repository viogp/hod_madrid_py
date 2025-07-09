#!/usr/bin/env python3
"""
Produce model galaxies from HOD models from
Avila+2020, Vos Gines+2024
"""

from pathlib import Path
from src.hod import run_hod_model
from src.hod_io import create_hod_params

def main():
    verbose   = True   # Set to True for progress messages
    seed      = 42     # Random seed for reproducibility
    
    # Cosmological and simulation parameters
    zsnap = 0.8594   # Snapshot redshift
    omega_M = 0.3089 # Matter density parameter    
    Lbox = 1000.0    # Box size in Mpc/h
    
    ftype = 'txt' # Possible input file type: txt
    output_dir = Path("output")

    # Default input file (modify as needed)
    input_dir = Path("data/example")
    infile = input_dir/"UNIT_haloes_logMmin13.500_logMmax14.500.txt"

    # =====================================================
    # Average HOD shape
    # =====================================================
    analytical_shape = True
    
    # For an analytical shape, the following parameters are needed
    hodshape = 1     # Shape of the average HOD: 1, 2 or 3
    mu = 12.0        # Log10 of characteristic halo mass
    Ac = 1.0         # Central galaxy amplitude
    As = 0.5         # Satellite galaxy amplitude
    M0_factor = -0.05  # mu offset for M0
    M1_factor = 0.35   # mu offset for M1
    M0 = 10**(mu + M0_factor)
    M1 = 10**(mu + M1_factor)
    alpha = 0.9      # Satellite power law slope
    sig = 0.08       # Central galaxy scatter (HOD2 and 3)
    gamma = -1.4     # Power law slope for high mass centrals (HOD3)

    # If analytical_shape is False, a file with Mmin Mmax Ncen Nsat should be given
    hod_shape_file = None

    # =====================================================
    # Probability distribution function for satellites
    # =====================================================
    beta = 0.0       # Poisson=0, Nearest integer=-2
    
    # =====================================================
    # Radial distribution function for satellites
    # =====================================================
    analytical_rp = True
    
    # For analytical profile, the following parameters are needed
    K = 1.0          # NFW truncation parameter
    
    # If analytical_rp is False, a file with r Nsat should be given
    hod_rp_file = None

    # =====================================================
    # Velocity distribution function for satellites
    # =====================================================
    analytical_vp = True

    # For analytical profile, the following parameters are needed
    vfact = 1.0      # Velocity factor (alpha_v)
    vt = 0.0         # Tangential velocity (km/s, =0 or =500)
    vtdisp = 0.0     # Tangential velocity dispersion (km/s, =0 or =200)

    # If analytical_vp is False, a file with r Nsat should be given
    hod_vp_file = None
    
    # =====================================================
    # Collect parameters into a dictionary and check them
    # =====================================================
    hod_params = create_hod_params(infile,output_dir,ftype=ftype,seed=seed,
                                   zsnap=zsnap, Lbox=Lbox, omega_M=omega_M,
                                   analytical_shape=analytical_shape,
                                   hod_shape_file=hod_shape_file,
                                   hodshape=hodshape, mu=mu, Ac=Ac, As=As,
                                   M0=M0, M1=M1, alpha=alpha, sig=sig, gamma=gamma,
                                   beta=beta,
                                   analytical_rp=analytical_rp,
                                   hod_rp_file=hod_rp_file,K=K,                                    
                                   analytical_vp=analytical_vp,
                                   hod_vp_file=hod_vp_file,                                   
                                   vfact=vfact,vt=vt, vtdisp=vtdisp)
    
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
