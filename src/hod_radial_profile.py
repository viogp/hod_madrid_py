import jax.numpy as jnp
from jax import jit, lax

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

