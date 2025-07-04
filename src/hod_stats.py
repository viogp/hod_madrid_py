import jax.numpy as jnp
import jax.random as random
from jax import jit, lax
import jax.scipy as jsp

@jit
def rand_gauss(key) -> float:
    """Generate Gaussian random number"""
    return random.normal(key)

# =====================================================
# RANDOM SAMPLING FUNCTIONS
# =====================================================

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
def neg_binomial_sample(key, x: float, beta: float) -> int:
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
def binomial_sample(key, x: float, beta: float) -> int:
    """Sample from extended binomial distribution"""
    a = -beta
    n_val = jnp.ceil(1.0 / a)
    n_val = jnp.maximum(n_val, jnp.trunc(x + 1.0))
    p_val = x / n_val
    
    # Standard binomial sampling
    key, subkey = random.split(key)
    return random.binomial(subkey, n_val.astype(int), p_val)

@jit
def next_integer(key, x: float) -> int:
    """Sample integer based on fractional part"""
    low = jnp.floor(x).astype(int)
    key, subkey = random.split(key)
    rand01 = random.uniform(subkey)
    return lax.cond(rand01 > (x - low), lambda: low, lambda: low + 1)

