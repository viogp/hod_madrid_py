"""
Core HOD (Halo Occupation Distribution) functions implemented in JAX
Translated from C code HOD_NFW_V14.c
"""

import jax.numpy as jnp
import jax.random as random
from jax import jit, lax
import jax.scipy as jsp
import numpy as np
from typing import Tuple

class HODCore:
    def __init__(self, zsnap: float, LBOX: float, OMEGA_M: float, 
                 mu: float, Ac: float, As: float, vfact: float, beta: float,
                 K: float, vt: float, vtdisp: float, M0: float, M1: float,
                 alpha: float, sig: float, gamma: float, MORE: bool = True):
        
        self.zsnap = zsnap
        self.LBOX = LBOX
        self.OMEGA_M = OMEGA_M
        self.mu = mu
        self.Ac = Ac
        self.As = As
        self.vfact = vfact
        self.beta = beta
        self.K = K
        self.vt = vt
        self.vtdisp = vtdisp
        self.M0 = M0
        self.M1 = M1
        self.alpha = alpha
        self.sig = sig
        self.gamma = gamma
        self.MORE = MORE
        
        # Physical constants
        self.rho_crit = 27.755e10
        
    @staticmethod
    @jit
    def factorial(n: int) -> float:
        """Compute factorial using gamma function"""
        return jsp.special.gamma(n + 1.0)
    
    @staticmethod
    @jit
    def rand_gauss(key) -> float:
        """Generate Gaussian random number"""
        return random.normal(key)
    
    @staticmethod
    @jit
    def E2(z: float, OMEGA_M: float) -> float:
        """Hubble parameter squared"""
        return OMEGA_M * (1.0 + z)**3 + (1.0 - OMEGA_M)
    
    @staticmethod
    @jit
    def Omega(z: float, OMEGA_M: float) -> float:
        """Omega matter at redshift z"""
        return (OMEGA_M * (1.0 + z)**3) / (OMEGA_M * (1.0 + z)**3 + (1.0 - OMEGA_M))
    
    @staticmethod
    @jit
    def Delta_vir(z: float, OMEGA_M: float) -> float:
        """Virial overdensity"""
        omega_z = HODCore.Omega(z, OMEGA_M)
        d = 1.0 - omega_z
        return 18 * jnp.pi**2 + 82 * d - 39 * d**2
    
    @staticmethod
    @jit
    def I_NFW(x: float) -> float:
        """NFW integral function"""
        return (1.0 / (1.0 + x) + jnp.log(1.0 + x) - 1.0)
    
    @staticmethod
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
    
    @staticmethod
    @jit
    def R_from_mass(Mass: float, OVD: float, rho_crit: float) -> float:
        """Radius from mass and overdensity"""
        return (3.0 / (4.0 * rho_crit * OVD * jnp.pi) * Mass)**(1.0/3.0)
    
    @jit
    def poisson_sample(self, key, lam: float) -> int:
        """Sample from Poisson distribution"""
        def body_fun(carry):
            k, prob, key = carry
            key, subkey = random.split(key)
            prob += (lam**k * jnp.exp(-lam)) / self.factorial(k)
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
    def neg_binomial_sample(self, key, x: float, beta: float) -> int:
        """Sample from negative binomial distribution"""
        r = 1.0 / beta
        p = r / (r + x)
        
        def body_fun(carry):
            N, P, key = carry
            key, subkey = random.split(key)
            # Use log-gamma to avoid overflow
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
        return N - 1  # Adjust for 0-based indexing
    
    @jit
    def binomial_sample(self, key, x: float, beta: float) -> int:
        """Sample from extended binomial distribution"""
        a = -beta
        n_val = jnp.ceil(1.0 / a)
        n_val = jnp.maximum(n_val, jnp.trunc(x + 1.0))
        p_val = x / n_val
        
        # Standard binomial sampling
        key, subkey = random.split(key)
        return random.binomial(subkey, n_val.astype(int), p_val)
    
    @jit
    def next_integer(self, key, x: float) -> int:
        """Sample integer based on fractional part"""
        low = jnp.floor(x).astype(int)
        key, subkey = random.split(key)
        rand01 = random.uniform(subkey)
        return lax.cond(rand01 > (x - low), lambda: low, lambda: low + 1)
    
    @jit
    def HOD_powerlaw(self, key, M: float) -> int:
        """Halo Occupation Distribution for satellites using power law"""
        def zero_case():
            return 0
        
        def main_case():
            xsat = (M - self.M0) / self.M1
            mean_sat = self.As * xsat**self.alpha
            
            def neg_binomial_case():
                return self.neg_binomial_sample(key, mean_sat, self.beta)
            
            def poisson_case():
                return self.poisson_sample(key, mean_sat)
            
            def binomial_case():
                return self.binomial_sample(key, mean_sat, self.beta)
            
            def next_int_case():
                return self.next_integer(key, mean_sat)
            
            return lax.cond(
                self.beta > 0.0, neg_binomial_case,
                lambda: lax.cond(
                    (self.beta <= 0.0) & (self.beta >= -1.0/171.0), poisson_case,
                    lambda: lax.cond(
                        (self.beta < -1.0/171.0) & (self.beta >= -1.0), binomial_case,
                        next_int_case
                    )
                )
            )
        
        return lax.cond((self.M1 <= 0.0) | (M < self.M0), zero_case, main_case)
    
    @jit
    def HOD_gaussPL(self, key, logM: float) -> int:
        """Halo Occupation Distribution for centrals using Gaussian + Power Law"""
        def gauss_case():
            return self.Ac / (self.sig * jnp.sqrt(2.0 * jnp.pi)) * jnp.exp(
                -(logM - self.mu)**2 / (2 * self.sig**2)
            )
        
        def powerlaw_case():
            return self.Ac / (self.sig * jnp.sqrt(2.0 * jnp.pi)) * 10.0**(
                self.gamma * (logM - self.mu)
            )
        
        r = lax.cond(logM < self.mu, gauss_case, powerlaw_case)
        
        # Check if probability > 1 (would be a warning in original code)
        r = jnp.minimum(r, 1.0)
        
        key, subkey = random.split(key)
        rand_val = random.uniform(subkey)
        
        return lax.cond(rand_val < r, lambda: 1, lambda: 0)
    
    @jit
    def NFW_to_pos(self, key, M: float) -> Tuple[float, float, float]:
        """Generate position within NFW halo"""
        x_max = self.concentration_klypin(M, self.zsnap)
        I_max = self.I_NFW(x_max)
        
        key, subkey = random.split(key)
        y_rand = random.uniform(subkey) * I_max
        
        # Binary search for inverse
        def binary_search_body(carry):
            low, high, mid = carry
            y_try = self.I_NFW(mid)
            new_low = lax.cond(y_try > y_rand, lambda: low, lambda: mid)
            new_high = lax.cond(y_try > y_rand, lambda: mid, lambda: high)
            new_mid = 0.5 * (new_low + new_high)
            return (new_low, new_high, new_mid)
        
        def binary_search_cond(carry):
            low, high, mid = carry
            y_try = self.I_NFW(mid)
            return jnp.abs(y_try - y_rand) > 0.001 * I_max
        
        init_search = (0.0, x_max, 0.5 * x_max)
        _, _, final_mid = lax.while_loop(binary_search_cond, binary_search_body, init_search)
        
        # Calculate radius
        Delta_vir_val = self.Delta_vir(self.zsnap, self.OMEGA_M)
        R_mass = self.R_from_mass(M, Delta_vir_val, self.rho_crit)
        c_val = self.concentration_klypin(M, self.zsnap)
        R = final_mid * R_mass / (c_val * self.K)
        
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
    def vir_to_vel(self, key, M: float) -> Tuple[float, float, float]:
        """Generate velocity dispersion within halo"""
        Delta_vir_val = self.Delta_vir(self.zsnap, self.OMEGA_M)
        E2_val = self.E2(self.zsnap, self.OMEGA_M)
        
        sigma = 476 * 0.9 * (Delta_vir_val * E2_val)**(1.0/6.0) * (M / 1.0e15)**(1.0/3.0)
        
        key, subkey1 = random.split(key)
        key, subkey2 = random.split(key)
        key, subkey3 = random.split(key)
        
        Dvx = sigma * self.rand_gauss(subkey1)
        Dvy = sigma * self.rand_gauss(subkey2)
        Dvz = sigma * self.rand_gauss(subkey3)
        
        return Dvx, Dvy, Dvz
    
    def process_halo_line(self, line: str, f_out, key):
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
        
        Nsat = self.HOD_powerlaw(subkey1, M)
        Ncent = self.HOD_gaussPL(subkey2, logM)
        
        # Write central galaxy if present
        if Ncent == 1:
            if self.MORE:
                f_out.write(f"{x:.5f} {y:.5f} {z:.5f} {vx:.5f} {vy:.5f} {vz:.5f} "
                           f"{M:.6e} {Nsat:d} 0.0 0.0 0.0 0.0 0.0 0.0 {halo_id:d}\n")
            else:
                f_out.write(f"{x:.5f} {y:.5f} {z:.5f} {vx:.5f} {vy:.5f} {vz:.5f} "
                           f"{M:.6e} {Nsat:d}\n")
        
        # Generate satellite galaxies
        for j in range(Nsat):
            key, subkey = random.split(key)
            
            # Generate position offset
            Dx, Dy, Dz = self.NFW_to_pos(subkey, M)
            
            # Generate velocity offset
            key, subkey = random.split(key)
            Dvx, Dvy, Dvz = self.vir_to_vel(subkey, M)
            
            # Add tangential velocity component
            key, subkey = random.split(key)
            vtrand = self.rand_gauss(subkey) * self.vtdisp + self.vt
            
            Dr = jnp.sqrt(Dx**2 + Dy**2 + Dz**2)
            Dr = jnp.maximum(Dr, 1e-10)  # Avoid division by zero
            
            ux = -Dx / Dr
            uy = -Dy / Dr
            uz = -Dz / Dr
            
            Dvx = self.vfact * Dvx + ux * vtrand
            Dvy = self.vfact * Dvy + uy * vtrand
            Dvz = self.vfact * Dvz + uz * vtrand
            
            # Apply periodic boundary conditions
            xgal = x + Dx
            ygal = y + Dy
            zgal = z + Dz
            
            xgal = xgal % self.LBOX
            ygal = ygal % self.LBOX
            zgal = zgal % self.LBOX
            
            # Write satellite galaxy
            if self.MORE:
                f_out.write(f"{xgal:.5f} {ygal:.5f} {zgal:.5f} "
                           f"{vx + Dvx:.5f} {vy + Dvy:.5f} {vz + Dvz:.5f} "
                           f"{M:.6e} {Nsat:d} {Dvx:.4f} {Dvy:.4f} {Dvz:.4f} "
                           f"{Dx:.4f} {Dy:.4f} {Dz:.4f} {halo_id:d}\n")
            else:
                f_out.write(f"{xgal:.5f} {ygal:.5f} {zgal:.5f} "
                           f"{vx + Dvx:.5f} {vy + Dvy:.5f} {vz + Dvz:.5f} "
                           f"{M:.6e} {Nsat:d}\n")
