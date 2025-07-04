"""
HOD functions implemented as pure functions (alternative to class-based approach)
"""

import jax.numpy as jnp
import jax.random as random
from jax import jit, lax
import jax.scipy as jsp
from typing import Tuple, NamedTuple

# Define a parameters structure
class HODParams(NamedTuple):
    """Container for all HOD parameters"""
    zsnap: float
    LBOX: float
    OMEGA_M: float
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

# Pure mathematical functions
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
def rand_gauss(key) -> float:
    """Generate Gaussian random number"""
    return random.normal(key)

@jit
def poisson_sample(key, lam: float) -> int:
    """Sample from Poisson distribution"""
    def body_fun(carry):
        k, prob, key = carry
        key, subkey = random.split(key)
        prob += (lam**k * jnp.exp(-lam)) / jsp.special.gamma(k + 1.0)
        return (k + 1, prob, key)
    
    def cond_fun(carry):
        k, prob, key = carry
        key, subkey = random.split(key)
        r = random.uniform(subkey)
        return (prob < r) & (prob < 0.999999999999999)
    
    key, subkey = random.split(key)
    r = random.uniform(subkey)
    init_val = (0, 0.0, key)
    k, prob, _ = lax.while_loop(cond_fun, body_fun, init_val)
    return k

@jit
def HOD_powerlaw(key, M: float, params: HODParams) -> int:
    """Halo Occupation Distribution for satellites using power law"""
    def zero_case():
        return 0
    
    def main_case():
        xsat = (M - params.M0) / params.M1
        mean_sat = params.As * xsat**params.alpha
        
        def neg_binomial_case():
            return neg_binomial_sample(key, mean_sat, params.beta)
        
        def poisson_case():
            return poisson_sample(key, mean_sat)
        
        def binomial_case():
            return binomial_sample(key, mean_sat, params.beta)
        
        def next_int_case():
            return next_integer(key, mean_sat)
        
        return lax.cond(
            params.beta > 0.0, neg_binomial_case,
            lambda: lax.cond(
                (params.beta <= 0.0) & (params.beta >= -1.0/171.0), poisson_case,
                lambda: lax.cond(
                    (params.beta < -1.0/171.0) & (params.beta >= -1.0), binomial_case,
                    next_int_case
                )
            )
        )
    
    return lax.cond((params.M1 <= 0.0) | (M < params.M0), zero_case, main_case)

@jit
def HOD_gaussPL(key, logM: float, params: HODParams) -> int:
    """Halo Occupation Distribution for centrals using Gaussian + Power Law"""
    def gauss_case():
        return params.Ac / (params.sig * jnp.sqrt(2.0 * jnp.pi)) * jnp.exp(
            -(logM - params.mu)**2 / (2 * params.sig**2)
        )
    
    def powerlaw_case():
        return params.Ac / (params.sig * jnp.sqrt(2.0 * jnp.pi)) * 10.0**(
            params.gamma * (logM - params.mu)
        )
    
    r = lax.cond(logM < params.mu, gauss_case, powerlaw_case)
    r = jnp.minimum(r, 1.0)
    
    key, subkey = random.split(key)
    rand_val = random.uniform(subkey)
    
    return lax.cond(rand_val < r, lambda: 1, lambda: 0)

@jit
def NFW_to_pos(key, M: float, params: HODParams) -> Tuple[float, float, float]:
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
    Delta_vir_val = Delta_vir(params.zsnap, params.OMEGA_M)
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

@jit
def vir_to_vel(key, M: float, params: HODParams) -> Tuple[float, float, float]:
    """Generate velocity dispersion within halo"""
    Delta_vir_val = Delta_vir(params.zsnap, params.OMEGA_M)
    E2_val = E2(params.zsnap, params.OMEGA_M)
    
    sigma = 476 * 0.9 * (Delta_vir_val * E2_val)**(1.0/6.0) * (M / 1.0e15)**(1.0/3.0)
    
    key, subkey1 = random.split(key)
    key, subkey2 = random.split(key)
    key, subkey3 = random.split(key)
    
    Dvx = sigma * rand_gauss(subkey1)
    Dvy = sigma * rand_gauss(subkey2)
    Dvz = sigma * rand_gauss(subkey3)
    
    return Dvx, Dvy, Dvz

# Additional helper functions needed
@jit
def next_integer(key, x: float) -> int:
    """Sample integer based on fractional part"""
    low = jnp.floor(x).astype(int)
    key, subkey = random.split(key)
    rand01 = random.uniform(subkey)
    return lax.cond(rand01 > (x - low), lambda: low, lambda: low + 1)

@jit
def neg_binomial_sample(key, x: float, beta: float) -> int:
    """Sample from negative binomial distribution"""
    r = 1.0 / beta
    p = r / (r + x)
    
    def body_fun(carry):
        N, P, key = carry
        key, subkey = random.split(key)
        log_prob = (jsp.special.gammaln(N + r) - jsp.special.gammaln(r) - 
                   jsp.special.gammaln(N + 1) + r * jnp.log(p) + N * jnp.log(1 - p))
        P += jnp.exp(log_prob)
        return (N + 1, P, key)
    
    def cond_fun(carry):
        N, P, key = carry
        key, subkey = random.split(key)
        rand01 = random.uniform(subkey)
        return P < rand01
    
    key, subkey = random.split(key)
    rand01 = random.uniform(subkey)
    init_val = (0, 0.0, key)
    N, P, _ = lax.while_loop(cond_fun, body_fun, init_val)
    return N - 1

@jit
def binomial_sample(key, x: float, beta: float) -> int:
    """Sample from extended binomial distribution"""
    a = -beta
    n_val = jnp.ceil(1.0 / a)
    n_val = jnp.maximum(n_val, jnp.trunc(x + 1.0))
    p_val = x / n_val
    
    key, subkey = random.split(key)
    return random.binomial(subkey, n_val.astype(int), p_val)

# Main processing function
def process_halo_line(line: str, f_out, key, params: HODParams, MORE: bool = True):
    """Process a single halo line and write galaxy outputs"""
    # Parse input line
    parts = line.strip().split()
    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
    vx, vy, vz = float(parts[3]), float(parts[4]), float(parts[5])
    logM = float(parts[6])
    halo_id = int(parts[7])
    
    M = 10**logM
    
    # Generate satellite and central galaxies
    key, subkey1 = random.split(key)
    key, subkey2 = random.split(key)
    
    Nsat = HOD_powerlaw(subkey1, M, params)
    Ncent = HOD_gaussPL(subkey2, logM, params)
    
    # Write central galaxy if present
    if Ncent == 1:
        if MORE:
            f_out.write(f"{x:.5f} {y:.5f} {z:.5f} {vx:.5f} {vy:.5f} {vz:.5f} "
                       f"{M:.6e} {Nsat:d} 0.0 0.0 0.0 0.0 0.0 0.0 {halo_id:d}\n")
        else:
            f_out.write(f"{x:.5f} {y:.5f} {z:.5f} {vx:.5f} {vy:.5f} {vz:.5f} "
                       f"{M:.6e} {Nsat:d}\n")
    
    # Generate satellite galaxies
    for j in range(Nsat):
        key, subkey = random.split(key)
        
        # Generate position offset
        Dx, Dy, Dz = NFW_to_pos(subkey, M, params)
        
        # Generate velocity offset
        key, subkey = random.split(key)
        Dvx, Dvy, Dvz = vir_to_vel(subkey, M, params)
        
        # Add tangential velocity component
        key, subkey = random.split(key)
        vtrand = rand_gauss(subkey) * params.vtdisp + params.vt
        
        Dr = jnp.sqrt(Dx**2 + Dy**2 + Dz**2)
        Dr = jnp.maximum(Dr, 1e-10)  # Avoid division by zero
        
        ux = -Dx / Dr
        uy = -Dy / Dr
        uz = -Dz / Dr
        
        Dvx = params.vfact * Dvx + ux * vtrand
        Dvy = params.vfact * Dvy + uy * vtrand
        Dvz = params.vfact * Dvz + uz * vtrand
        
        # Apply periodic boundary conditions
        xgal = x + Dx
        ygal = y + Dy
        zgal = z + Dz
        
        xgal = xgal % params.LBOX
        ygal = ygal % params.LBOX
        zgal = zgal % params.LBOX
        
        # Write satellite galaxy
        if MORE:
            f_out.write(f"{xgal:.5f} {ygal:.5f} {zgal:.5f} "
                       f"{vx + Dvx:.5f} {vy + Dvy:.5f} {vz + Dvz:.5f} "
                       f"{M:.6e} {Nsat:d} {Dvx:.4f} {Dvy:.4f} {Dvz:.4f} "
                       f"{Dx:.4f} {Dy:.4f} {Dz:.4f} {halo_id:d}\n")
        else:
            f_out.write(f"{xgal:.5f} {ygal:.5f} {zgal:.5f} "
                       f"{vx + Dvx:.5f} {vy + Dvy:.5f} {vz + Dvz:.5f} "
                       f"{M:.6e} {Nsat:d}\n")

# Usage in main.py would be:
def create_hod_params(zsnap, LBOX, OMEGA_M, mu, Ac, As, vfact, beta, K, vt, vtdisp, M0, M1, alpha, sig, gamma):
    """Create HOD parameters structure"""
    return HODParams(
        zsnap=zsnap, LBOX=LBOX, OMEGA_M=OMEGA_M, mu=mu, Ac=Ac, As=As,
        vfact=vfact, beta=beta, K=K, vt=vt, vtdisp=vtdisp,
        M0=M0, M1=M1, alpha=alpha, sig=sig, gamma=gamma
    )
