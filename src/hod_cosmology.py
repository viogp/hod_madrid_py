import jax.numpy as jnp
import jax.random as random
from jax import jit

@jit
def E2(z: float, omega_M: float) -> float:
    """Hubble parameter squared"""
    return omega_M * (1.0 + z)**3 + (1.0 - omega_M)

@jit
def Omega(z: float, omega_M: float) -> float:
    """Omega matter at redshift z"""
    return (omega_M * (1.0 + z)**3) / (omega_M * (1.0 + z)**3 + (1.0 - omega_M))

@jit
def Delta_vir(z: float, omega_M: float) -> float:
    """Virial overdensity"""
    omega_z = Omega(z, omega_M)
    d = 1.0 - omega_z
    return 18 * jnp.pi**2 + 82 * d - 39 * d**2

