"""
Input/output and information functions for HOD simulation
"""

import os
from pathlib import Path
from typing import Tuple, NamedTuple

# Define a parameters structure
class HODParams(NamedTuple):
    """Container for all HOD parameters"""
    zsnap: float
    Lbox: float
    omega_M: float
    mu: float
    Ac: float
    As: float
    vfact: float
    beta: float
    K: float
    vt: float
    vtdisp: float
    M0: float
    M1: float
    alpha: float
    sig: float
    gamma: float
    rho_crit: float = 27.755e10

def create_hod_params(zsnap, Lbox, omega_M, mu, Ac, As, vfact, beta, K, vt, vtdisp, M0, M1, alpha, sig, gamma):
    """Create HOD parameters structure"""
    return HODParams(
        zsnap=zsnap, Lbox=Lbox, omega_M=omega_M, mu=mu, Ac=Ac, As=As,
        vfact=vfact, beta=beta, K=K, vt=vt, vtdisp=vtdisp,
        M0=M0, M1=M1, alpha=alpha, sig=sig, gamma=gamma
    )

def print_parameter_info():
    """Print detailed information about the parameters that can be modified"""
    print("""
    PARAMETER MODIFICATION GUIDE:
    ============================
    
    HOD Model Parameters:
    ---------------------
    - mu: Log10 of characteristic halo mass (affects both centrals and satellites)
          Typical range: 11.0 - 13.0
          Higher values = more massive halos needed for galaxies
    
    - Ac: Amplitude for central galaxies (0-1, typically ~1)
          Controls the maximum fraction of halos that host central galaxies
          Ac = 1.0 means all halos above threshold can host centrals
    
    - As: Amplitude for satellite galaxies (affects satellite abundance)
          Typical range: 0.1 - 2.0
          Higher values = more satellites per halo
    
    - alpha: Power law slope for satellite HOD (typically 0.8-1.2)
             Controls how satellite number scales with halo mass
             Higher values = steeper scaling with mass
    
    - sig: Scatter in central galaxy occupation (log-normal width)
           Typical range: 0.05 - 0.2
           Controls the smoothness of the central galaxy transition
    
    - gamma: Power law slope for high-mass centrals 
             Typical range: -2.0 to -1.0
             Controls central galaxy behavior in massive halos
    
    Velocity Parameters:
    -------------------
    - vfact: Velocity scaling factor for satellites (typically 0.5-1.5)
             Controls how much satellites follow halo internal motions
             vfact = 1.0 means satellites follow halo velocities exactly
    
    - vt: Systematic tangential velocity (km/s, often 0 or ~500)
          Additional radial motion component for satellites
    
    - vtdisp: Random tangential velocity dispersion (km/s, often 0 or ~200)
              Random scatter in the tangential velocity component
    
    Model Parameters:
    ----------------
    - beta: Scatter model parameter
            beta = 0: Poisson distribution (no extra scatter)
            beta > 0: Negative binomial (super-Poissonian scatter)
            beta < 0: Sub-Poissonian scatter (more regular than Poisson)
    
    - K: NFW truncation parameter (typically ~1)
         Controls where satellite orbits are truncated
         K = 1.0 means truncation at virial radius
    
    
    Cosmological Parameters:
    -----------------------
    - zsnap: Redshift of the snapshot
             Affects concentration-mass relations and virial properties
    
    - omega_M: Matter density parameter
               Standard cosmological parameter (typically ~0.3)
    
    - Lbox: Simulation box size (Mpc/h)
            Used for periodic boundary conditions
    
    Physical Constants:
    ------------------
    - rho_crit: Critical density (typically 2.775e11 M_sun/(Mpc/h)^3)
    
    File Control:
    ------------
    - MORE: Boolean flag for extended output format
            True = additional columns with satellite details
            False = basic galaxy catalog format
    
    - verbose: Boolean flag for detailed parameter information
               True = print this guide and parameter summaries
               False = minimal output
        
    File Format:
    -----------
    Input file (halo catalog):
    - Columns: x y z vx vy vz logM halo_id
    - Units: positions in Mpc/h, velocities in km/s, masses in M_sun/h
    
    Output file (galaxy catalog):
    Basic format: x y z vx vy vz M Nsat
    Extended format: x y z vx vy vz M Nsat Dvx Dvy Dvz Dx Dy Dz halo_id
    
    Tips for Parameter Tuning:
    --------------------------
    1. Start with default values and modify one parameter at a time
    2. Check galaxy number density matches observations
    3. Verify clustering properties (correlation functions)
    4. Test different beta values to match observed scatter
    5. Use verbose=True to monitor parameter effects
    
    Common Parameter Combinations:
    -----------------------------
    - High galaxy density: Increase Ac, decrease mu
    - More satellites: Increase As, decrease alpha
    - Smoother central transition: Increase sig
    - More scattered satellite counts: Increase beta
    - Slower satellite velocities: Decrease vfact
    """)
    print(line_separator())
    return


def line_separator():
    width = os.get_terminal_size().columns
    separator = "=" * 60
    return separator
    
    
def count_lines(filename):
    """Count non-comment lines in input file"""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            # Skip header lines starting with #
            data_lines = [line for line in lines if not line.strip().startswith('#') and line.strip()]
            return len(data_lines)
    except FileNotFoundError:
        print(f"ERROR: Could not open input file {filename}")
        return -1
    except Exception as e:
        print(f"ERROR: Error reading file {filename}: {e}")
        return -1

def check_input_file(filename):
    """Check if input file exists and is readable"""
    if not os.path.exists(filename):
        print(f"ERROR: Input file does not exist: {filename}")
        return False
    
    if not os.path.isfile(filename):
        print(f"ERROR: Input path is not a file: {filename}")
        return False
    
    if not os.access(filename, os.R_OK):
        print(f"ERROR: Input file is not readable: {filename}")
        return False
    
    return True

def create_output_directory(output_file):
    """Create output directory if it doesn't exist"""
    output_dir = Path(output_file).parent
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"ERROR: Could not create output directory {output_dir}: {e}")
        return False

def check_output_file(filename):
    """Check if output file can be created"""
    output_path = Path(filename)
    
    # Check if parent directory exists or can be created
    if not create_output_directory(filename):
        return False
    
    # Check if file already exists and warn user
    if output_path.exists():
        print(f"WARNING: Output file already exists and will be overwritten: {filename}")
    
    # Check if directory is writable
    if not os.access(output_path.parent, os.W_OK):
        print(f"ERROR: Output directory is not writable: {output_path.parent}")
        return False
    
    return True

def print_run_summary(params):
    """Print summary of run parameters"""
    print(f"HOD Parameters:")
    print(f"  mu = {params['mu']:.3f}")
    print(f"  Ac = {params['Ac']:.4f}")
    print(f"  As = {params['As']:.5f}")
    print(f"  vfact = {params['vfact']:.2f}")
    print(f"  beta = {params['beta']:.3f}")
    print(f"  K = {params['K']:.2f}")
    print(f"  vt = {params['vt']:.0f}")
    print(f"  vtdisp = {params['vtdisp']:.0f}")
    print(f"  alpha = {params['alpha']:.1f}")
    print(f"  sig = {params['sig']:.2f}")
    print(f"  gamma = {params['gamma']:.1f}")
    
    print(f"Derived parameters:")
    print(f"  M0 = {params['M0']:.6e}")
    print(f"  M1 = {params['M1']:.6e}")
    
    print(f"Cosmology:")
    print(f"  zsnap = {params['zsnap']:.4f}")
    print(f"  omega_M = {params['omega_M']:.4f}")
    print(f"  Lbox = {params['Lbox']:.1f} Mpc/h")
    
    print(f"Files:")
    print(f"  Input: {params['input_file']}")
    print(f"  Output: {params['output_file']}")
    print(f"  Extended format: {params['MORE']}")
    print(line_separator())
    return

def validate_parameters(params):
    """Validate parameter ranges and combinations"""
    warnings = []
    errors = []
    
    # Check parameter ranges
    if params['mu'] < 10.0 or params['mu'] > 15.0:
        warnings.append(f"mu = {params['mu']:.3f} is outside typical range [10.0, 15.0]")
    
    if params['Ac'] < 0.0 or params['Ac'] > 2.0:
        warnings.append(f"Ac = {params['Ac']:.4f} is outside typical range [0.0, 2.0]")
    
    if params['As'] < 0.0 or params['As'] > 5.0:
        warnings.append(f"As = {params['As']:.5f} is outside typical range [0.0, 5.0]")
    
    if params['alpha'] < 0.0 or params['alpha'] > 3.0:
        warnings.append(f"alpha = {params['alpha']:.1f} is outside typical range [0.0, 3.0]")
    
    if params['sig'] <= 0.0 or params['sig'] > 1.0:
        warnings.append(f"sig = {params['sig']:.2f} is outside typical range (0.0, 1.0]")
    
    if params['vfact'] < 0.0 or params['vfact'] > 3.0:
        warnings.append(f"vfact = {params['vfact']:.2f} is outside typical range [0.0, 3.0]")
    
    if params['K'] <= 0.0 or params['K'] > 5.0:
        warnings.append(f"K = {params['K']:.2f} is outside typical range (0.0, 5.0]")
    
    # Check for potential issues
    if params['M0'] >= params['M1']:
        errors.append(f"M0 ({params['M0']:.2e}) must be less than M1 ({params['M1']:.2e})")
    
    if params['Lbox'] <= 0.0:
        errors.append(f"Lbox = {params['Lbox']:.1f} must be positive")
    
    if params['omega_M'] <= 0.0 or params['omega_M'] >= 1.0:
        errors.append(f"omega_M = {params['omega_M']:.4f} must be between 0 and 1")
    
    # Print warnings and errors
    if warnings:
        print("WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
        print()
    
    if errors:
        print("ERRORS:")
        for error in errors:
            print(f"  - {error}")
        print()
        return False
    
    return True

def print_file_info(input_file, nhalos):
    """Print information about input and output files"""
    print(f"Input file: {input_file}")
    print(f"Found {nhalos} halos in the catalogue")
    
    # Try to determine file size
    try:
        file_size = os.path.getsize(input_file)
        if file_size > 1024**3:  # GB
            size_str = f"{file_size / 1024**3:.1f} GB"
        elif file_size > 1024**2:  # MB
            size_str = f"{file_size / 1024**2:.1f} MB"
        elif file_size > 1024:  # KB
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size} bytes"
        
        print(f"File size: {size_str}")
    except:
        pass  # Don't fail if we can't get file size


    
