import numpy as np
from numba import jit
import math

import src.hod_io as io
from src.hod_cosmology import Delta_vir, E2
from src.hod_pdf import rand_gauss

@jit(nopython=True)
def generate_virial_velocity(M, zsnap, omega_M):
    """Generate velocity dispersion within halo"""
    Delta_vir_val = Delta_vir(zsnap, omega_M)
    E2_val = E2(zsnap, omega_M)
    
    sigma = 476 * 0.9 * math.pow(Delta_vir_val * E2_val,1.0/6.0) *\
        math.pow(M / 1.0e15,1.0/3.0)
    
    Dvx = sigma * rand_gauss()
    Dvy = sigma * rand_gauss()
    Dvz = sigma * rand_gauss()
    
    return Dvx, Dvy, Dvz
