import numpy as np
from numba import jit

import src.hod_const as c
from src.hod_cosmology import Delta_vir

@jit(nopython=True)
def concentration_klypin(M, z):
    """Concentration parameter calculation"""
    if z < 0.25:
        C0, gamma, M0 = 9.5, 0.09, 3.0e5 * 1.0e12
    elif z < 0.75:
        C0, gamma, M0 = 6.75, 0.088, 5000 * 1.0e12
    elif z < 1.22:
        C0, gamma, M0 = 5.0, 0.086, 450 * 1.0e12
    else:
        C0, gamma, M0 = 4.05, 0.085, 90.0 * 1.0e12
    
    return C0 * (M / 1.0e12)**(-gamma) * (1.0 + (M / M0)**0.4)

@jit(nopython=True)
def I_NFW(x):
    """NFW integral"""
    return (1.0 / (1.0 + x) + np.log(1.0 + x) - 1.0)


@jit(nopython=True)
def R_from_mass(Mass, OVD, rho_crit):
    """Radius from mass and overdensity"""
    return (3.0 / (4.0 * rho_crit * OVD * np.pi) * Mass)**(1.0/3.0)


@jit(nopython=True)
def generate_nfw_position(M, K, zsnap, omega_M):
    """NFW position generation"""
    x_max = concentration_klypin(M, zsnap)
    I_max = I_NFW(x_max)
    
    y_rand = np.random.random() * I_max

    tol = 0.001 * I_NFW(x_max)
    low = 0.0
    high = x_max
    mid = 0.5 * (low + high)
    y_try = I_NFW(mid)
    
    # Keep iterating until convergence
    while abs(y_try - y_rand) >= tol: 
        if y_try > y_rand:
            high = mid
        else:
            low = mid
        mid = 0.5 * (low + high)
        y_try = I_NFW(mid)
    
    # Calculate radius
    Delta_vir_val = Delta_vir(zsnap, omega_M)
    R_mass = R_from_mass(M, Delta_vir_val, c.rho_crit)
    c_val = concentration_klypin(M, zsnap)
    R = mid * R_mass / (c_val * K)
    
    # Generate spherical coordinates
    phi = np.random.random() * 2 * np.pi
    costh = np.random.random() * 2 - 1.0
    
    Dx = R * np.sqrt(1.0 - costh*costh) * np.cos(phi)
    Dy = R * np.sqrt(1.0 - costh*costh) * np.sin(phi)
    Dz = R * costh
    
    return Dx, Dy, Dz
