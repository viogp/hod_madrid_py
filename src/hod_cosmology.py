import math
from numba import jit

@jit(nopython=True)
def omega_Mz(z, omega_M):
    """Omega matter at redshift z"""
    return (omega_M * (1.0 + z)**3) / (omega_M * (1.0 + z)**3 + (1.0 - omega_M))


@jit(nopython=True)
def E2(z, omega_M):
    """Hubble parameter squared"""
    return omega_M * (1.0 + z)**3 + (1.0 - omega_M)


@jit(nopython=True)
def Delta_vir(z, omega_M):
    """Virial overdensity calculation"""
    d = 1.0 - omega_Mz(z, omega_M)
    return 18 * math.pi**2 + 82 * d - 39 * d**2

