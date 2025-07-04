import jax.numpy as jnp
import jax.random as random
from jax import jit

@jit
def E2(z: float, OMEGA_M: float) -> float:
    """Hubble parameter squared"""
    return OMEGA_M * (1.0 + z)**3 + (1.0 - OMEGA_M)

@jit
def Omega(z: float, OMEGA_M: float) -> float:
    """Omega matter at redshift z"""
    return (OMEGA_M * (1.0 + z)**3) / (OMEGA_M * (1.0 + z)**3 + (1.0 - OMEGA_M))

@jit
def Delta_vir(z: float, OMEGA_M: float) -> float:
    """Virial overdensity"""
    omega_z = Omega(z, OMEGA_M)
    d = 1.0 - omega_z
    return 18 * jnp.pi**2 + 82 * d - 39 * d**2

