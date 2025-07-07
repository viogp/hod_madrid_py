import jax.numpy as jnp
import jax.random as random
from jax import jit, lax
from typing import Tuple, NamedTuple

import src.hod_io as io
from src.hod_cosmology import Delta_vir

@jit
def I_NFW(x: float) -> float:
    """NFW integral function"""
    return (1.0 / (1.0 + x) + jnp.log(1.0 + x) - 1.0)

@jit
def concentration_klypin(M: float, z: float) -> float:
    """Concentration parameter from Klypin et al."""
    def case1():  # z < 0.25
        C0, gamma, M0 = 9.5, 0.09, 3.0e5 * 1.0e12
        return C0 * (M / 1.0e12)**(-gamma) * (1.0 + (M / M0)**0.4)
    
    def case2():  # z < 0.75
        C0, gamma, M0 = 6.75, 0.088, 5000 * 1.0e12
        return C0 * (M / 1.0e12)**(-gamma) * (1.0 + (M / M0)**0.4)
    
    def case3():  # z < 1.22
        C0, gamma, M0 = 5.0, 0.086, 450 * 1.0e12
        return C0 * (M / 1.0e12)**(-gamma) * (1.0 + (M / M0)**0.4)
    
    def case4():  # z >= 1.22
        C0, gamma, M0 = 4.05, 0.085, 90.0 * 1.0e12
        return C0 * (M / 1.0e12)**(-gamma) * (1.0 + (M / M0)**0.4)
    
    return lax.cond(z < 0.25, case1,
                   lambda: lax.cond(z < 0.75, case2,
                                   lambda: lax.cond(z < 1.22, case3, case4)))

@jit
def R_from_mass(Mass: float, OVD: float, rho_crit: float) -> float:
    """Radius from mass and overdensity"""
    return (3.0 / (4.0 * rho_crit * OVD * jnp.pi) * Mass)**(1.0/3.0)


@jit
def NFW_to_pos(key, M: float, params:io.HODParams) -> Tuple[float, float, float]:
    """Generate position within NFW halo"""
    x_max = concentration_klypin(M, params.zsnap)
    I_max = I_NFW(x_max)
        
    key, subkey = random.split(key)
    y_rand = random.uniform(subkey) * I_max
        
    # Binary search for inverse
    def binary_search_body(carry):
        low, high, mid = carry
        y_try = I_NFW(mid)
        new_low = lax.cond(y_try > y_rand, lambda: low, lambda: mid)
        new_high = lax.cond(y_try > y_rand, lambda: mid, lambda: high)
        new_mid = 0.5 * (new_low + new_high)
        return (new_low, new_high, new_mid)
        
    def binary_search_cond(carry):
        low, high, mid = carry
        y_try = I_NFW(mid)
        return jnp.abs(y_try - y_rand) > 0.001 * I_max
        
    init_search = (0.0, x_max, 0.5 * x_max)
    _, _, final_mid = lax.while_loop(binary_search_cond, binary_search_body, init_search)
        
    # Calculate radius
    Delta_vir_val = Delta_vir(params.zsnap, params.omega_M)
    R_mass = R_from_mass(M, Delta_vir_val, params.rho_crit)
    c_val = concentration_klypin(M, params.zsnap)
    R = final_mid * R_mass / (c_val * params.K)
    
    # Generate spherical coordinates
    key, subkey1 = random.split(key)
    key, subkey2 = random.split(key)
    
    phi = random.uniform(subkey1) * 2 * jnp.pi
    costh = random.uniform(subkey2) * 2 - 1.0
    sinth = jnp.sqrt(1.0 - costh**2)
    
    Dx = R * sinth * jnp.cos(phi)
    Dy = R * sinth * jnp.sin(phi)
    Dz = R * costh

    return Dx, Dy, Dz
