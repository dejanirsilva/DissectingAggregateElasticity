# -*- coding: utf-8 -*-
# Chebyshev collocation (1D)
# Includes BOTH:
#   - Oversampled Least-Squares (LS) with Clenshaw eval, optional CC weights, degree-weighted ridge
#   - Square Collocation (root solve) with Lobatto nodes (no ridge; exactly determined)
#
# Toggle at the top:
#   USE_LS = True  -> LS solver
#   USE_LS = False -> Collocation solver

# %%
from jax.flatten_util import ravel_pytree
from scipy.optimize import least_squares, root
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple

jax.config.update("jax_enable_x64", True)

# %%
# -------- toggles --------
USE_LS       = True        # True: LS; False: Collocation
EVAL_MODE    = "basis"     # "basis" or "clenshaw"
NODES_KIND   = "gauss"   # for LS: "gauss" or "lobatto"
LS_WEIGHTING = "flat"      # "none" | "cc" | "flat"

# Ridge options (LS only)
RIDGE_ENABLED      = False
RIDGE_LAMBDA       = 1e-1
RIDGE_POWER        = 2.0
RIDGE_EXEMPT_FIRST = 1
# %%
# ------------------------
# Model configuration
# ------------------------
class Config(NamedTuple):
    N: tuple = (21, )                          # number of Chebyshev basis functions
    lower_bound: tuple = (0.0+1e-6,)
    upper_bound: tuple = (1.0-1e-6,)

class ModelParameters(NamedTuple):
    ρ: float = 0.04
    γa: float = 20.0
    γb: float = 2.0
    ψ: float = 2.0
    μ: float = 0.02
    σ: float = 0.04
    κ: float = 0.00
    θb: float = 0.25

def benchmark(params: ModelParameters):
    ρ, ψ, σ, γa, γb, θb, μ = params.ρ, params.ψ, params.σ, params.γa, params.γb, params.θb, params.μ
    γ = γa * (1-θb) + γb * θb
    r = ρ + 1 / ψ * μ  - (1+1/ψ) * γ * σ**2 / 2
    π = γ * σ**2 / 2
    y = ρ  - (1-1/ψ) * (μ - γ * σ**2 / 2)
    pd = 1/y
    ζ  = y ** (1/(1-ψ)) / ρ ** (ψ/(1-ψ))
    return dict(r=r, π=π, y=y, pd=pd, ζ=ζ)

# %%
# ------------------------
# Chebyshev utilities
# ------------------------
def map_to_std(x, a=-1.0, b=1.0):
    return 2.0*(x - a)/(b - a) - 1.0

def cheb_T_values(x, n, a=-1.0, b=1.0):
    x_std = map_to_std(x, a, b)
    T0 = jnp.ones_like(x_std)
    if n == 0:
        return T0
    T1 = x_std
    Ts = [T0, T1]
    for _ in range(1, n):
        T0, T1 = T1, 2.0*x_std*T1 - T0
        Ts.append(T1)
    return jnp.stack(Ts, axis=-1)  # (..., n+1)

def chebypol(x, K, a=-1.0, b=1.0):
    return cheb_T_values(x, K-1, a, b)

def clenshaw_T(coeffs, x, a=-1.0, b=1.0):
    x_std = map_to_std(x, a, b)
    a0 = coeffs[0]
    c = coeffs
    n = c.shape[0] - 1
    if n <= 0:
        return jnp.full_like(x_std, a0, dtype=jnp.result_type(x_std, a0))
    b_kplus1 = jnp.zeros_like(x_std)
    b_kplus2 = jnp.zeros_like(x_std)
    for k in range(n, 0, -1):
        b_k = 2.0*x_std*b_kplus1 - b_kplus2 + c[k]
        b_kplus2 = b_kplus1
        b_kplus1 = b_k
    return a0 + x_std*b_kplus1 - b_kplus2

def chebyshev_lobatto_nodes(a, b, n):
    k = jnp.arange(n)
    nodes_std = jnp.cos((jnp.pi / (n-1) * k))
    return a + (nodes_std + 1) * (b - a) / 2

def chebyshev_gauss_nodes(a, b, n):
    j = jnp.arange(1, n+1)
    nodes_std = jnp.cos((2*j - 1) * jnp.pi / (2*n))
    return a + (nodes_std + 1) * (b - a) / 2

def clenshaw_curtis_weights(n):
    N = n - 1
    k = np.arange(0, n)
    if N == 0:
        return jnp.asarray(np.array([2.0]))
    w = np.zeros(n, dtype=float)
    if N % 2 == 0:
        for j in range(1, N//2):
            w += (2.0/(1.0 - 4.0*j*j)) * np.cos(2.0*j*np.pi*k/N)
        w += (1.0/(1.0 - N*N)) * np.cos(np.pi*k)
        w = (2.0/N) * (1.0 + 2.0*w)
    else:
        for j in range(1, (N-1)//2 + 1):
            w += (2.0/(1.0 - 4.0*j*j)) * np.cos(2.0*j*np.pi*k/N)
        w = (2.0/N) * (1.0 + 2.0*w)
    return jnp.asarray(w, dtype=jnp.float64)

# %%
# ------------------------
# Initialization
# ------------------------
def initialize_chebyshev_benchmark(config: Config):
    N_total = int(np.prod(config.N))
    θa = jnp.zeros(N_total)
    θb = jnp.zeros(N_total)
    ζ = benchmark(ModelParameters())['ζ']
    θa = θa.at[0].set(ζ)
    θb = θb.at[0].set(ζ)
    return dict(θa=θa, θb=θb)

def initialize_chebyshev(config: Config, θ_initial=None):
    if θ_initial is None:
        θ_initial = initialize_chebyshev_benchmark(config)
    θa = jnp.zeros(config.N[0])
    θb = jnp.zeros(config.N[0])
    n_initial = len(θ_initial['θa'])
    θa = θa.at[0:n_initial].set(θ_initial['θa'][0:n_initial])
    θb = θb.at[0:n_initial].set(θ_initial['θb'][0:n_initial])
    return dict(θa=θa, θb=θb)

# %%
# ------------------------
# Model building blocks
# ------------------------
def _eval_series(coeffs, x, a, b):
    if EVAL_MODE == "clenshaw":
        return clenshaw_T(coeffs, x, a, b)
    else:
        basis = chebypol(x, coeffs.shape[0], a, b)
        return jnp.dot(basis, coeffs)

def assemble_functions(config: Config, θ, params: ModelParameters, x):
    a, b = config.lower_bound[0], config.upper_bound[0]
    wa = _eval_series(θ['θa'], x, a, b)
    wb = _eval_series(θ['θb'], x, a, b)
    
    ca = jnp.exp(wa)
    cb = jnp.exp(wb)

    q = 1.0 / ((1.0-x)*ca + x*cb)
    return dict(wa=wa, wb=wb, ca=ca, cb=cb, q=q)

def compute_derivatives(config: Config, θ, params: ModelParameters, x):
    wa_fn = lambda x: assemble_functions(config, θ, params, x)['wa']
    wb_fn = lambda x: assemble_functions(config, θ, params, x)['wb']
    q_fn  = lambda x: assemble_functions(config, θ, params, x)['q']

    wax_fn = jax.jacfwd(wa_fn)
    wbx_fn = jax.jacfwd(wb_fn)
    qx_fn  = jax.jacfwd(q_fn)

    waxx_fn = jax.jacfwd(wax_fn)
    wbxx_fn = jax.jacfwd(wbx_fn)
    qxx_fn  = jax.jacfwd(qx_fn)

    cax_over_ca  = wax_fn(x) 
    cbx_over_cb  = wbx_fn(x)
    caxx_over_ca = waxx_fn(x) + wax_fn(x)**2
    cbxx_over_cb = wbxx_fn(x) + wbx_fn(x)**2

    qx_over_q = qx_fn(x) / q_fn(x)
    qxx_over_q = qxx_fn(x) / q_fn(x)

    return dict(
        wax = wax_fn(x), wbx = wbx_fn(x), qx = qx_fn(x),
        waxx = waxx_fn(x), wbxx = wbxx_fn(x), qxx = qxx_fn(x),
        cax_over_ca = cax_over_ca, cbx_over_cb = cbx_over_cb, caxx_over_ca = caxx_over_ca, cbxx_over_cb = cbxx_over_cb,
        qx_over_q = qx_over_q, qxx_over_q = qxx_over_q
    )

# %%
# ------------------------
# HJB residual at a single x
# ------------------------
def H(config: Config, θ, params: ModelParameters, x):
    ρ, ψ, σ, γa, γb, θb, μ, κ = params.ρ, params.ψ, params.σ, params.γa, params.γb, params.θb, params.μ, params.κ
    eps = 1e-12

    # Levels and derivatives
    F  = assemble_functions(config, θ, params, x)
    dF = compute_derivatives(config, θ, params, x)

    q, ca, cb = F['q'], F['ca'], F['cb']
    cax_over_ca, cbx_over_cb, qx_over_q      = dF['cax_over_ca'], dF['cbx_over_cb'], dF['qx_over_q']
    caxx_over_ca, cbxx_over_cb, qxx_over_q   = dF['caxx_over_ca'], dF['cbxx_over_cb'], dF['qxx_over_q']

    γt = 1.0 / ((1.0 - x)/γa + x/γb)

    Δ = (γa - 1.0) / (1-ψ) * cax_over_ca - (γb - 1.0) / (1-ψ) * cbx_over_cb
    

    base = x * (1.0 - x) * γt / (γa * γb)
    mix  = (γa - γb) * qx_over_q  + Δ

    σx  = (base * (γa - γb)) / (1.0 - base * mix) * σ
    σζa = cax_over_ca / (1.0 -ψ) * σx
    σζb = cbx_over_cb / (1.0 -ψ) * σx
    σq  = qx_over_q * σx

    η  = γt * ((σ + σq) + (1.0 - x) * (γa - 1.0)/γa * σζa + x * (γb - 1.0)/γb * σζb)
    σa = η/γa + (1.0 - γa)/γa * σζa
    σb = η/γb + (1.0 - γb)/γb * σζb

    μx  = x * (1.0 - x) * ((σb - σa) * (η - (σ + σq)) + ca - cb) - κ * (x - θb)
    μζa = cax_over_ca / (1.0 -ψ) * μx + 0.5 * caxx_over_ca / (1.0 -ψ) * σx**2
    μζb = cbx_over_cb / (1.0 -ψ) * μx + 0.5 * cbxx_over_cb / (1.0 -ψ) * σx**2
    μq  = (qx  / (q  + eps)) * μx + 0.5 * (qxx  / (q  + eps)) * σx**2

    Φa  = σa * η - 0.5 * γa * (σa**2 - 2.0 * (1.0 - γa)/γa * σζa * σa + σζa**2)
    Φb  = σb * η - 0.5 * γb * (σb**2 - 2.0 * (1.0 - γb)/γb * σζb * σb + σζb**2)

    # r = 1.0/(q + eps) + μ + μq + σ*σq - η * (σ + σq)
    μt = (1.0 - x) * μζa + x * μζb
    Φt = (1.0 - x) * Φa + x * Φb
    r = ρ + ψ**(-1) * (μ + σ*σq + μq - η * (σ + σq)) - (1.0 - ψ**(-1)) * (μt + Φt)

    residuals_a = -(1.0 - (ψ * ρ + (1.0 - ψ) * (r + Φa + μζa))/ca)
    residuals_b = -(1.0 - (ψ * ρ + (1.0 - ψ) * (r + Φb + μζb))/cb)
    residuals   = jnp.array([residuals_a, residuals_b])

    return residuals, dict(
        ζa=ζa, ζb=ζb, q=q, ca=ca, cb=cb, σR = σ + σq,
        σx=σx, σζa=σζa, σζb=σζb, σq=σq, qx=qx, b = x * qx / q, qxx = qxx, d = x ** 2 * qxx / q,
        η=η, σa=σa, σb=σb, r=r,
        residuals=residuals
    )

# %%
# ------------------------
# Residual wrappers & weights
# ------------------------
def make_weighted_residual_fn(config: Config, params: ModelParameters, state_grid, weights=None):
    sg = jnp.asarray(state_grid)
    if weights is None:
        w = jnp.ones_like(sg)
    else:
        w = jnp.asarray(weights)
    @jax.jit
    def fθ(θ):
        res, _ = jax.vmap(partial(H, config, θ, params))(sg)  # shape (M,2)
        res_w = res * w[:, None]
        return res_w.ravel()
    return fθ

def ls_weights_for_nodes(nodes, nodes_kind: str, scheme: str, a: float, b: float):
    nodes_kind = nodes_kind.lower()
    scheme = scheme.lower()
    n = nodes.shape[0]
    if scheme == "none":
        return jnp.ones_like(nodes)
    if nodes_kind == "lobatto" and scheme == "cc":
        w = clenshaw_curtis_weights(n)
        w = jnp.maximum(w, 0)
        return jnp.sqrt(w + 1e-16)
    if nodes_kind == "gauss" and scheme == "flat":
        x_std = map_to_std(nodes, a, b)
        w = jnp.sqrt(jnp.maximum(1.0 - x_std**2, 0.0))
        return w
    return jnp.ones_like(nodes)

# %%
# ------------------------
# Ridge helpers (LS only)
# ------------------------
def degree_weights_1d(N: int, power: float = 2.0, exempt_first: int = 1):
    k = jnp.arange(N, dtype=jnp.float64)
    w = (k / jnp.maximum(N-1, 1)) ** power
    w = w.at[:exempt_first].set(0.0)
    return w

def ridge_residuals(θ_structured, λ: float, power: float, exempt_first: int):
    if λ <= 0.0:
        return jnp.zeros((0,), dtype=jnp.float64)
    θa = θ_structured['θa']
    θb = θ_structured['θb']
    N = θa.shape[0]
    w = degree_weights_1d(N, power=power, exempt_first=exempt_first)
    wλ = jnp.sqrt(λ) * w
    reg_a = wλ * θa
    reg_b = wλ * θb
    return jnp.concatenate([reg_a, reg_b], axis=0)

# %%
# ------------------------
# Solvers
# ------------------------
def model_solution_least_squares(config: Config, params: ModelParameters, θ_initial,
                                 oversample_factor: float = 5.0,
                                 nodes_kind: str = NODES_KIND,
                                 weighting: str = LS_WEIGHTING,
                                 ridge_enabled: bool = RIDGE_ENABLED,
                                 ridge_lambda: float = RIDGE_LAMBDA,
                                 ridge_power: float = RIDGE_POWER,
                                 ridge_exempt_first: int = RIDGE_EXEMPT_FIRST,
                                 use_jax_jacobian: bool = False):
    """Oversampled LS with optional degree-weighted ridge; returns θ, result, grid, weights."""
    N = config.N[0]
    M = max(N, int(np.ceil(oversample_factor * N)))

    a, b = config.lower_bound[0], config.upper_bound[0]
    if nodes_kind.lower().startswith("gauss"):
        grid = chebyshev_gauss_nodes(a, b, M)
    else:
        grid = chebyshev_lobatto_nodes(a, b, M)

    w = ls_weights_for_nodes(grid, nodes_kind, weighting, a, b)
    fθ = make_weighted_residual_fn(config, params, grid, weights=w)

    Θ_flat, unravel = ravel_pytree(θ_initial)

    def jax_resid(θ):
        res_data = fθ(θ)  # (2M,)
        if ridge_enabled and ridge_lambda > 0.0:
            res_ridge = ridge_residuals(θ, ridge_lambda, ridge_power, ridge_exempt_first)  # (2N,)
            return jnp.concatenate([res_data, res_ridge], axis=0)
        else:
            return res_data

    if use_jax_jacobian:
        jac_fun = jax.jit(jax.jacrev(jax_resid))
        def jac_wrapper(θ_flat):
            θ_structured = unravel(θ_flat)
            J = jac_fun(θ_structured)
            return np.array(J)
    else:
        jac_wrapper = '2-point'

    def residual_wrapper(θ_flat):
        θ_structured = unravel(θ_flat)
        return np.array(jax_resid(θ_structured))

    res = least_squares(
        residual_wrapper,
        x0=np.array(Θ_flat),
        jac=jac_wrapper,
        method='trf',
        xtol=1e-9, ftol=1e-9, gtol=1e-8,
        max_nfev=30000,
        verbose=0,
    )
    Θ_solved = unravel(res.x)
    return Θ_solved, res, grid, w

def model_solution_collocation(config: Config, params: ModelParameters, θ_initial):
    """Square collocation at Lobatto nodes (M = N). No ridge; uses root solve."""
    N = config.N[0]
    a, b = config.lower_bound[0], config.upper_bound[0]
    grid = chebyshev_lobatto_nodes(a, b, N)  # include endpoints

    Θ_flat, unravel = ravel_pytree(θ_initial)
    fθ = make_weighted_residual_fn(config, params, grid, weights=None)  # weights=1

    def residual_wrapper(θ_flat):
        θ_structured = unravel(θ_flat)
        return np.array(fθ(θ_structured))  # length 2N

    res = root(
        residual_wrapper,
        x0=np.array(Θ_flat),
        method='hybr',
        options={'maxfev': 10000, 'xtol': 1e-8},
    )
    Θ_solved = unravel(res.x)
    return Θ_solved, res, grid

# %%
# -------- toggles --------
USE_LS       = True        # True: LS; False: Collocation
EVAL_MODE    = "basis"     # "basis" or "clenshaw"
NODES_KIND   = "gauss"   # for LS: "gauss" or "lobatto"
LS_WEIGHTING = "flat"      # "none" | "cc" | "flat"

# Ridge options (LS only)
RIDGE_ENABLED      = False
RIDGE_LAMBDA       = 1e-1
RIDGE_POWER        = 2.0
RIDGE_EXEMPT_FIRST = 1

# %%
# ------------------------
# Minimal driver
# ------------------------
if __name__ == "__main__":
    config = Config(N=(81,), lower_bound=(0.01,), upper_bound=(0.99,))
    params = ModelParameters(γb=0.7, γa=30.0, ρ = 0.021, ψ = 2.0, κ = 0.00)
    θ_initial = initialize_chebyshev(config)

    if USE_LS:
        θ_new, resinfo, os_grid, weights = model_solution_least_squares(
            config, params, θ_initial,
            oversample_factor=5.0,
            nodes_kind=NODES_KIND,
            weighting=LS_WEIGHTING,
            ridge_enabled=RIDGE_ENABLED,
            ridge_lambda=RIDGE_LAMBDA,
            ridge_power=RIDGE_POWER,
            ridge_exempt_first=RIDGE_EXEMPT_FIRST,
            use_jax_jacobian=False
        )
        print("LS: cost =", getattr(resinfo, "cost", "n/a"), "| ‖res‖ =", np.linalg.norm(resinfo.fun))
    else:
        θ_new, resinfo, grid = model_solution_collocation(config, params, θ_initial)
        print("Collocation:", resinfo.message, "| ‖res‖ =", np.linalg.norm(resinfo.fun))

    # Quick diagnostic on a Lobatto eval grid
    eval_grid = chebyshev_lobatto_nodes(config.lower_bound[0], config.upper_bound[0], 100)
    res_eval, outcomes = jax.vmap(partial(H, config, θ_new, params))(eval_grid)
    print("Max |residual| on eval grid:", float(jnp.max(jnp.abs(res_eval))))
# %%
plt.subplot(2,2,1)
plt.plot(eval_grid, outcomes['σR'])
plt.subplot(2,2,2)
plt.plot(eval_grid, outcomes['q'])
plt.subplot(2,2,3)
plt.plot(eval_grid, outcomes['ca'])
plt.subplot(2,2,4)
plt.plot(eval_grid, outcomes['cb'])
# %%
plt.plot(eval_grid, outcomes['d'])
plt.xlim(0, 0.2)
# %%
plt.plot(eval_grid, outcomes['qxx'])
# %%
plt.plot(eval_grid, outcomes['qx'])
# %%
plt.plot(eval_grid, outcomes['d'])
plt.xlim(0, 0.2)
# %%
plt.plot(jnp.log(eval_grid), outcomes['qxx'])
# %%
