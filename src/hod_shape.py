import jax.numpy as jnp
import jax.random as random
from jax import jit, lax

import src.hod_io as io
import src.hod_pdf as pdf

@jit
def HOD_powerlaw(key, M: float, params: io.HODParams) -> int:
    """Halo Occupation Distribution for satellites using power law"""
    def zero_case():
        return jnp.int32(0)  # Explicitly cast to int32
    
    def main_case():
        xsat = (M - params.M0) / params.M1
        mean_sat = params.As * xsat**params.alpha

        def neg_binomial_case():
            return pdf.neg_binomial_sample(key, mean_sat, params.beta)
        
        def poisson_case():
            return pdf.poisson_sample(key, mean_sat)
        
        def binomial_case():
            return pdf.binomial_sample(key, mean_sat, params.beta)
        
        def next_int_case():
            return pdf.next_integer(key, mean_sat)
        
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
def HOD_gaussPL(key, logM: float, params: io.HODParams) -> int:
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
