import jax.numpy as jnp
import jax.random as random
from jax import jit, lax
import jax.scipy as jsp
from typing import Tuple, NamedTuple

import src.hod_io as io
from src.hod_cosmology import Delta_vir, E2
from src.hod_pdf import rand_gauss

@jit
def vir_to_vel(key, M: float, params:io.HODParams) -> Tuple[float, float, float]:
    """Generate velocity dispersion within halo"""
    Delta_vir_val = Delta_vir(params.zsnap, params.omega_M)
    E2_val = E2(params.zsnap, params.omega_M)
    
    sigma = 476 * 0.9 * (Delta_vir_val * E2_val)**(1.0/6.0) * (M / 1.0e15)**(1.0/3.0)
    
    key, subkey1 = random.split(key)
    key, subkey2 = random.split(key)
    key, subkey3 = random.split(key)

    Dvx = sigma * rand_gauss(subkey1)
    Dvy = sigma * rand_gauss(subkey2)
    Dvz = sigma * rand_gauss(subkey3)

    return Dvx, Dvy, Dvz
