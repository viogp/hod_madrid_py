"""
Average HOD and scatter
"""
import numpy as np
from numba import jit
import math

import src.hod_pdf as pdf

@jit(nopython=True)
def erf_approx(x):
    """
    Error function approximation for Numba compatibility
    Using Abramowitz and Stegun approximation
    """
    # Constants for A&S approximation
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    
    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    
    return sign * y


@jit(nopython=True)
def HOD_erf(logM, mu, sig, As):
    """
    HOD using error function (HOD1)
    Direct translation from C code
    
    Parameters:
    -----------
    logM : float
        Log10 of halo mass
    mu : float
        Log10 of characteristic mass
    sig : float
        Scatter parameter
    As : float
        Amplitude parameter
        
    Returns:
    --------
    int : Number of central galaxies (0 or 1)
    """
    r = As * 0.5 * (1.0 + erf_approx((logM - mu) / sig))
    
    # High-precision random number generation as in C code
    rand_val = np.random.random()
    
    return 1 if rand_val < r else 0


@jit(nopython=True)
def HOD_gauss(logM, mu, sig, As):
    """
    HOD using Gaussian function (HOD2)
    Direct translation from C code
    
    Parameters:
    -----------
    logM : float
        Log10 of halo mass
    mu : float
        Log10 of characteristic mass
    sig : float
        Scatter parameter
    As : float
        Amplitude parameter
        
    Returns:
    --------
    int : Number of central galaxies (0 or 1)
    """
    r = As / (sig * math.sqrt(2.0 * math.pi)) * math.exp(-(logM - mu)**2 / (2.0 * sig**2))
    
    rand_val = np.random.random()
    
    return 1 if rand_val < r else 0


@jit(nopython=True)
def HOD_gaussPL(logM, mu, sig, Ac, gamma):
    """
    HOD using Gaussian + Power Law (HOD3)
    Direct translation from C code
    
    Parameters:
    -----------
    logM : float
        Log10 of halo mass
    mu : float
        Log10 of characteristic mass
    sig : float
        Scatter parameter
    Ac : float
        Central amplitude parameter
    gamma : float
        Power law slope for high masses
        
    Returns:
    --------
    int : Number of central galaxies (0 or 1)
    """
    if logM < mu:
        # Gaussian regime (low masses)
        r = Ac / (sig * math.sqrt(2.0 * math.pi)) * math.exp(-(logM - mu)**2 / (2.0 * sig**2))
    else:
        # Power law regime (high masses)
        r = Ac / (sig * math.sqrt(2.0 * math.pi)) * (10.0**(gamma * (logM - mu)))
    
    # Warning if r > 1.0 (as in C code)
    if r > 1.0:
        r = 1.0  # Cap at 1.0 to avoid issues
    
    rand_val = np.random.random()
    
    return 1 if rand_val < r else 0


@jit(nopython=True)  
def HOD_powerlaw(M, M0, M1, alpha, As, beta):
    """
    HOD power law for satellites
    Direct translation from C code with all beta cases
    
    Parameters:
    -----------
    M : float
        Halo mass (linear, not log)
    M0 : float
        Minimum mass for satellites
    M1 : float
        Characteristic satellite mass scale
    alpha : float
        Power law slope
    As : float
        Satellite amplitude
    beta : float
        Dispersion parameter
        
    Returns:
    --------
    int : Number of satellite galaxies
    """
    if M1 <= 0.0 or M < M0:
        return 0
    
    xsat = (M - M0) / M1
    mean_sat = As * (xsat**alpha)
    if mean_sat <= 0.0:
        return 0
    
    # Choose adequate PDF
    if beta < -1.0:
        return pdf.next_integer(mean_sat)
    elif beta <= 0.0 and beta >= -1.0/171.0:
        return pdf.poisson_sample(mean_sat)
    elif beta < -1.0/171.0 and beta >= -1.0:
        return pdf.binomial_sample(mean_sat, beta)
    elif beta > 0.0:
        return pdf.neg_binomial_sample(mean_sat, beta)
    else:
        return pdf.poisson_sample(mean_sat)  # Default case


@jit(nopython=True)
def get_hod_derived_params(mu, hodshape):
    """
    Get derived HOD parameters based on shape, following C code #ifdef logic
    
    Parameters:
    -----------
    mu : float
        Log10 characteristic mass
    hodshape : int
        HOD shape (1, 2, or 3)
        
    Returns:
    --------
    tuple : (M0, M1, alpha, sig, gamma)
    """
    if hodshape == 1:  # HOD1
        M0 = 10.0**mu
        M1 = 10.0**(mu + 1.3)
        alpha = 1.0
        sig = 0.15
        gamma = -1.4  # Default value
    elif hodshape == 2:  # HOD2
        M0 = 10.0**(mu - 0.1)
        M1 = 10.0**(mu + 0.3)
        alpha = 0.8
        sig = 0.12
        gamma = -1.4  # Default value
    else:  # HOD3 (default)
        M0 = 10.0**(mu - 0.05)
        M1 = 10.0**(mu + 0.35)
        alpha = 0.9
        sig = 0.08
        gamma = -1.4
    
    return M0, M1, alpha, sig, gamma


@jit(nopython=True)
def calculate_hod_occupation(M, mu, Ac, As, alpha, sig, gamma, M0, M1, hodshape, beta=0.0):
    """
    Calculate central and satellite occupation numbers for different HOD shapes
    Following the exact logic from C code
    
    Parameters:
    -----------
    M : float
        Halo mass (linear)
    mu : float
        Log10 of characteristic mass
    Ac : float
        Central amplitude
    As : float
        Satellite amplitude
    alpha : float
        Satellite power law slope
    sig : float
        Central scatter
    gamma : float
        Power law slope for centrals (HOD3 only)
    M0 : float
        Minimum satellite mass
    M1 : float
        Satellite mass scale
    hodshape : int
        HOD shape: 1 (erf), 2 (gauss), 3 (gaussPL)
    beta : float
        Satellite dispersion parameter
        
    Returns:
    --------
    tuple : (Ncen, Nsat) - number of central and satellite galaxies
    """
    # Calculate satellite galaxies (same for all haloes)
    Nsat = HOD_powerlaw(M, M0, M1, alpha, As, beta)

    # Calculate central galaxies based on HOD shape 
    if hodshape == 1:
        # HOD1: Error function
        Ncen = HOD_erf(M, mu, sig, Ac)
    elif hodshape == 2:
        # HOD2: Pure Gaussian
        Ncen = HOD_gauss(M, mu, sig, Ac)
    else:  # hodshape == 3 or default
        # HOD3: Asymmetric Gaussian
        Ncen = HOD_gaussPL(M, mu, sig, Ac, gamma)
        
    return Ncen, Nsat