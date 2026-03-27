# %% 
from jax.flatten_util import ravel_pytree
from scipy.optimize import least_squares, root
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple

# %%
# -------- toggles --------
USE_LS = True        # True: LS; False: Collocation
EVAL_MODE = "basis"     # "basis" or "clenshaw"
NODES_KIND = "gauss"   # for LS: "gauss" or "lobatto"
LS_WEIGHTING = "flat"      # "none" | "cc" | "flat"
DIAGNOSTIC_NODES = "lobatto" # Diagnostic eval grid nodes: "lobatto" (extrema) or "gauss" (zeros)

# Ridge options (LS only)
RIDGE_ENABLED = False
RIDGE_LAMBDA = 1e-1
RIDGE_POWER = 2.0
RIDGE_EXEMPT_FIRST = 1

jax.config.update("jax_enable_x64", True)

# %% 
# ------------------------
# Model configuration
# ------------------------
class Config(NamedTuple):
    # number of Chebyshev basis functions
    N: tuple = (21, )
    lower_bound: tuple = (0.0+1e-6,)
    upper_bound: tuple = (1.0-1e-6,)
    # if True, use s = -log(x) as Chebyshev variable on [0, S]
    # with S chosen so that exp(-S) = lower_bound[0]
    use_log_state: bool = False
    # if True, compute derivatives with respect to s using auto-diff (more stable)
    # if False, compute derivatives with respect to x using auto-diff
    use_s_derivatives: bool = True


class ModelParameters(NamedTuple):
    ρ: float = 0.04
    γa: float = 10.0
    γp: float = 10.0
    ψ: float = 2.0
    μ: float = 0.02
    σ: float = 0.04
    αp: float = 1.0
    κ: float = 0.00
    θa: float = 0.25


def benchmark(params: ModelParameters):
    ρ, ψ, σ, γa, γp, θa, μ = params.ρ, params.ψ, params.σ, params.γa, params.γp, params.θa, params.μ
    γ = γa * θa + γp * (1-θa)
    r = ρ + 1 / ψ * μ - (1+1/ψ) * γ * σ**2 / 2
    π = γ * σ**2 / 2
    y = ρ - (1-1/ψ) * (μ - γ * σ**2 / 2)
    ca = ρ - (1-1/ψ) * (μ - γa * σ**2 / 2)
    cp = ρ - (1-1/ψ) * (μ - γp * σ**2 / 2)
    lca = jnp.log(ca+ 1e-18)
    lcp = jnp.log(cp+ 1e-18)
    ly = jnp.log(y+ 1e-18)
    return dict(r=r, π=π, y=y, pd=1/y, lca = lca, lcp = lcp, ly = ly, ca = ca, cp = cp)

# %%
# ------------------------
# Chebyshev utilities
# ------------------------
def map_to_std(x, a=-1.0, b=1.0):
    return 2.0*(x - a)/(b - a) - 1.0


def basis_bounds(config: Config):
    """Return (a,b) bounds for the Chebyshev basis variable.
    If use_log_state is False, basis variable is x in [x_min, x_max].
    If True, basis variable is s = -log(x) in [s_min, s_max],
    where s_min = -log(x_max) and s_max = -log(x_min), so that mapping
    covers exactly x ∈ [x_min, x_max].
    """
    if getattr(config, 'use_log_state', False):
        x_min = float(config.lower_bound[0])
        x_max = float(config.upper_bound[0])
        s_min = -np.log(x_max+ 1e-18)
        s_max = -np.log(x_min+ 1e-18)
        return float(s_min), float(s_max)
    else:
        return float(config.lower_bound[0]), float(config.upper_bound[0])


def to_basis_var(config: Config, x):
    """Map physical state x to basis variable z (x or s).
    z = x if not using log-state; z = -log(x) otherwise."""
    return -jnp.log(x+ 1e-18) if getattr(config, 'use_log_state', False) else x


def from_basis_var(config: Config, z):
    """Inverse map from basis variable z to physical x.
    x = z if not using log-state; x = exp(-z) otherwise."""
    return jnp.exp(-z) if getattr(config, 'use_log_state', False) else z


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
    θp = jnp.zeros(N_total)
    lca = benchmark(ModelParameters())['lca'] # initial condition
    lcp = benchmark(ModelParameters())['lcp'] # initial condition
    θa = θa.at[0].set(lca)
    θp = θp.at[0].set(lcp)
    return dict(θa=θa, θp=θp)


def initialize_chebyshev(config: Config, θ_initial=None):
    if θ_initial is None:
        θ_initial = initialize_chebyshev_benchmark(config)
    θa = jnp.zeros(config.N[0])
    θp = jnp.zeros(config.N[0])
    n_initial = len(θ_initial['θa'])
    θa = θa.at[0:n_initial].set(θ_initial['θa'][0:n_initial])
    θp = θp.at[0:n_initial].set(θ_initial['θp'][0:n_initial])
    return dict(θa=θa, θp=θp)

# ------------------------
# Model building blocks
# ------------------------


def _eval_series(coeffs, x, a, b, config: Config = None):
    # Evaluate Chebyshev series w.r.t. basis variable z.
    # If using log-state, z = -log(x) on [0,S]; else z = x on [a,b].
    if config is not None and getattr(config, 'use_log_state', False):
        z = to_basis_var(config, x)
        az, bz = basis_bounds(config)
    else:
        z = x
        az, bz = a, b
    if EVAL_MODE == "clenshaw":
        return clenshaw_T(coeffs, z, az, bz)
    else:
        basis = chebypol(z, coeffs.shape[0], az, bz)
        return jnp.dot(basis, coeffs)


def assemble_functions(config: Config, θ, params: ModelParameters, x):
    a, b = config.lower_bound[0], config.upper_bound[0]
    wa = _eval_series(θ['θa'], x, a, b, config)
    wp = _eval_series(θ['θp'], x, a, b, config)
    ca = jnp.exp(wa)
    cp = jnp.exp(wp)
    return dict(ca=ca, cp=cp, wa = wa, wp = wp)

def compute_derivatives(config: Config, θ, params: ModelParameters, x, return_s: bool = False):
    def wa_fn(x): return assemble_functions(config, θ, params, x)['wa']
    def wp_fn(x): return assemble_functions(config, θ, params, x)['wp']

    wax_fn = jax.jacfwd(wa_fn)
    wpx_fn = jax.jacfwd(wp_fn)

    waxx_fn = jax.jacfwd(wax_fn)
    wpxx_fn = jax.jacfwd(wpx_fn)

    wax = wax_fn(x)
    wpx = wpx_fn(x)
    waxx = waxx_fn(x)
    wpxx = wpxx_fn(x)

    d = dict(
        wax=wax, wpx=wpx,
        waxx=waxx, wpxx=wpxx
    )

    # Optionally include derivatives with respect to s = -log(x):
    # w_s = -x * w_x; w_ss = x * w_x + x^2 * w_xx
    if return_s:
        xs = jnp.asarray(x)
        was = -xs * wax
        wps = -xs * wpx
        wass = xs * wax + (xs ** 2) * waxx
        wpss = xs * wpx + (xs ** 2) * wpxx
        d.update(dict(was=was, wps=wps, wass=wass, wpss=wpss))

    return d


def compute_derivatives_s_direct(config: Config, θ, params: ModelParameters, x, return_x: bool = False):
    """
    Compute derivatives directly with respect to s = -log(x) using automatic differentiation.
    This should be more numerically stable than computing x-derivatives first.
    
    Args:
        config: Configuration object
        θ: Chebyshev coefficients
        params: Model parameters
        x: State variable (wealth share)
        return_x: If True, also compute x-derivatives from s-derivatives
    
    Returns:
        Dictionary with derivatives. Keys include:
        - was, wps: first derivatives w.r.t. s
        - wass, wpss: second derivatives w.r.t. s
        - wax, wpx, waxx, wpxx: derivatives w.r.t. x (if return_x=True)
    """
    # Convert x to s = -log(x)
    s = -jnp.log(x + 1e-18)
    
    # Define functions in terms of s
    def wa_fn_s(s_val): 
        x_val = jnp.exp(-s_val)
        return assemble_functions(config, θ, params, x_val)['wa']
    
    def wp_fn_s(s_val): 
        x_val = jnp.exp(-s_val)
        return assemble_functions(config, θ, params, x_val)['wp']
    
    # Compute derivatives with respect to s using auto-diff
    was_fn = jax.jacfwd(wa_fn_s)
    wps_fn = jax.jacfwd(wp_fn_s)
    
    wass_fn = jax.jacfwd(was_fn)
    wpss_fn = jax.jacfwd(wps_fn)
    
    # Evaluate at s
    was = was_fn(s)
    wps = wps_fn(s)
    wass = wass_fn(s)
    wpss = wpss_fn(s)
    
    d = dict(
        was=was, wps=wps,
        wass=wass, wpss=wpss
    )
    
    # Optionally compute x-derivatives from s-derivatives
    if return_x:
        # Chain rule: w_x = w_s * ds/dx = w_s * (-1/x) = -w_s / x
        # w_xx = d/dx(w_x) = d/dx(-w_s / x) = w_s / x^2 - w_ss / x^2 = (w_s - w_ss) / x^2
        # But we need to be careful: w_ss is the second derivative w.r.t. s, not the mixed derivative
        # The correct formula is: w_xx = w_s / x^2 + w_ss / x^2 = (w_s + w_ss) / x^2
        xs = jnp.asarray(x)
        wax = -was / xs
        wpx = -wps / xs
        waxx = (was + wass) / (xs ** 2)
        wpxx = (wps + wpss) / (xs ** 2)
        
        d.update(dict(wax=wax, wpx=wpx, waxx=waxx, wpxx=wpxx))
    
    return d


def compare_derivative_methods(config: Config, θ, params: ModelParameters, x):
    """
    Compare the x-derivatives computed using the two different methods:
    1. Direct x-derivatives (original method)
    2. s-derivatives converted to x-derivatives (new method)
    
    Returns a dictionary with the differences to check numerical accuracy.
    """
    # Method 1: Direct x-derivatives
    dF_x = compute_derivatives(config, θ, params, x)
    
    # Method 2: s-derivatives converted to x-derivatives
    dF_s = compute_derivatives_s_direct(config, θ, params, x, return_x=True)
    
    # Compare the results
    differences = {}
    for key in ['wax', 'wpx', 'waxx', 'wpxx']:
        if key in dF_x and key in dF_s:
            diff = jnp.abs(dF_x[key] - dF_s[key])
            rel_diff = diff / (jnp.abs(dF_x[key]) + 1e-16)
            differences[f'{key}_abs_diff'] = diff
            differences[f'{key}_rel_diff'] = rel_diff
    
    return differences


def test_s_derivatives_stability(config: Config, θ, params: ModelParameters, x_test_points=None):
    """
    Test the numerical stability of s-derivatives vs x-derivatives.
    
    Args:
        config: Configuration object
        θ: Chebyshev coefficients
        params: Model parameters
        x_test_points: Array of x values to test (default: near boundaries where stability matters)
    
    Returns:
        Dictionary with comparison results
    """
    if x_test_points is None:
        # Test points near boundaries where numerical stability is most important
        x_test_points = jnp.array([1e-6, 1e-4, 1e-2, 0.1, 0.5, 0.9, 0.99, 1.0 - 1e-6])
    
    results = {}
    
    for i, x in enumerate(x_test_points):
        print(f"Testing x = {x:.2e}")
        
        # Compare methods
        diff = compare_derivative_methods(config, θ, params, x)
        
        # Store results
        results[f'x_{i}'] = {
            'x_value': float(x),
            's_value': float(-jnp.log(x + 1e-18)),
            'differences': diff
        }
        
        # Print summary
        max_abs_diff = max([float(diff[key]) for key in diff.keys() if 'abs_diff' in key])
        max_rel_diff = max([float(diff[key]) for key in diff.keys() if 'rel_diff' in key])
        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(f"  Max relative difference: {max_rel_diff:.2e}")
        print()
    
    return results


# %%
# ------------------------
# HJB residual at a single x
# ------------------------

def homotopy_transform(params, ε):
    # γ_average = (params.γa+params.γb)/2
    # γa = γ_average * (1.0 - ε) + params.γa * ε
    # γb = γ_average * (1.0 - ε) + params.γb * ε
    αp = 1.0 * (1.0 - ε) + params.αp * ε
    return params._replace(αp=αp)


def H(config: Config, θ, params: ModelParameters, x):
    ρ, γa, γp, ψ, σ, αp, θa, μ, κ = params.ρ, params.γa, params.γp, params.ψ, params.σ, params.αp, params.θa, params.μ, params.κ

    # Wealth share
    xp, xa = 1-x, x

    # Levels and derivatives
    F   = assemble_functions(config, θ, params, x)
    ca, cp        = F['ca'], F['cp']
    ca, cp        = jnp.squeeze(ca), jnp.squeeze(cp)

    # Choose derivative computation method based on config
    if config.use_s_derivatives:
        dF = compute_derivatives_s_direct(config, θ, params, x, return_x=True)
        # Get x-derivatives from s-derivatives (already computed by the function)
        wax, wpx, waxx, wpxx = dF['wax'], dF['wpx'], dF['waxx'], dF['wpxx']
        wax, wpx      = jnp.squeeze(wax), jnp.squeeze(wpx)
        waxx, wpxx    = jnp.squeeze(waxx), jnp.squeeze(wpxx)
    else:
        dF = compute_derivatives(config, θ, params, x)
        wax, wpx, waxx, wpxx = dF['wax'], dF['wpx'], dF['waxx'], dF['wpxx']
        wax, wpx      = jnp.squeeze(wax), jnp.squeeze(wpx)
        waxx, wpxx    = jnp.squeeze(waxx), jnp.squeeze(wpxx)

    # Derivatives
    cax_over_ca   = wax
    cpx_over_cp   = wpx
    caxx_over_ca  = waxx + wax ** 2
    cpxx_over_cp  = wpxx + wpx ** 2

    # Derivatives
    y             = xp * cp + xa * ca
    yx_over_y     = (ca-cp) / y + (xp * cp / y * cpx_over_cp +
                                   xa * ca / y * cax_over_ca)
    yxx_over_y    = 2.0 * (ca * cax_over_ca - cp * cpx_over_cp) / y + (
                            xp * cp / y * cpxx_over_cp +
                            xa * ca / y * caxx_over_ca)
    # Diffusion terms
    αa  = (1-xp * αp)/xa
    # σx  = x * (αa - 1.0) * σ / (1.0 + x * (αa - 1.0) * yx_over_y)
    σx  = (1.0 - x) * (1.0 - αp) * σ / (1.0 + (1.0 - x) * (1.0 - αp) * yx_over_y)
    σca = cax_over_ca * σx
    σcp = cpx_over_cp * σx
    σy  = yx_over_y * σx
    σR  = σ - σy

    # Hedging demand
    va = (1-1/γa)/(ψ-1) * cax_over_ca * σx / σR

    # Risk premium
    pi = γa * (σR ** 2) * (αa - va )

    # Drift terms
    μx  = x * (1.0 - x) * ((αa - αp) * (pi - σR ** 2) + cp - ca) - κ * (x - θa)
    μca = cax_over_ca * μx + 0.5 * caxx_over_ca * σx**2
    μcp = cpx_over_cp * μx + 0.5 * cpxx_over_cp * σx**2
    μy  = yx_over_y * μx + 0.5 * yxx_over_y * σx**2

    # Interest rate
    r = y + μ - μy + σy ** 2 - σ*σy - pi

    # Consumption-wealth ratio
    xia = μca + (1-γa) * σca * σR * αa + (ψ-γa)/(1-ψ) * (σca ** 2) / 2
    xip = μcp + (1-γp) * σcp * σR * αp + (ψ-γp)/(1-ψ) * (σcp ** 2) / 2
    ca_tgt = ψ * ρ + (1.0 - ψ) * (r + pi * αa - γa * (αa * σR) ** 2 / 2) + xia
    cp_tgt = ψ * ρ + (1.0 - ψ) * (r + pi * αp - γp * (αp * σR) ** 2 / 2) + xip

    residuals_a = (ca - ca_tgt)
    residuals_p = (cp - cp_tgt)
    residuals = jnp.array([residuals_a, residuals_p])

    return residuals, dict(
        q=1/y, ca=ca, cp=cp, σR=σR,
        σx=σx, σca=σca, σcp=σcp, σy=σy,
        μx=μx, μca=μca, μcp=μcp, μy=μy,
        pi=pi, αa=αa, αp=αp, r=r,
        residuals=residuals
    )
def price_dividend_ratio(p):
    ya = p.ρ + (1/p.ψ-1) * (p.μ - p.γa * p.σ**2/2)
    yp = p.ρ + (1/p.ψ-1) * (p.μ - p.γp * p.σ**2/2)

    πa = p.γa * p.σ**2
    πp = p.γp * p.σ**2
    αp = πa / (p.γp * p.σ**2)
    αa = πp / (p.γa * p.σ**2)

    cp_lim = p.ψ * p.ρ + (1.0 - p.ψ) * (ya + p.μ + πa * (αp - 1.0) - p.γp * p.σ**2 * αp ** 2 / 2)
    ca_lim = p.ψ * p.ρ + (1.0 - p.ψ) * (yp + p.μ + πp * (αa - 1.0) - p.γa * p.σ**2 * αa ** 2 / 2)
    return dict(qa=1/ya, qp=1/yp, ca_lim=ca_lim, cp_lim=cp_lim, ca = ya, cp = yp, αa=αa, αp=αp)

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

    # Build nodes in basis variable space (x or s) then map to physical x
    if getattr(config, 'use_log_state', False):
        az, bz = basis_bounds(config)
    else:
        az, bz = config.lower_bound[0], config.upper_bound[0]
    if nodes_kind.lower().startswith("gauss"):
        z_grid = chebyshev_gauss_nodes(az, bz, M)
    else:
        z_grid = chebyshev_lobatto_nodes(az, bz, M)
    grid = from_basis_var(config, z_grid)

    w = ls_weights_for_nodes(z_grid if getattr(config, 'use_log_state', False) else grid,
                             nodes_kind, weighting, az, bz)
    fθ = make_weighted_residual_fn(config, params, grid, weights=w)

    Θ_flat, unravel = ravel_pytree(θ_initial)

    def jax_resid(θ):
        res_data = fθ(θ)  # (2M,)
        if ridge_enabled and ridge_lambda > 0.0:
            res_ridge = ridge_residuals(
                θ, ridge_lambda, ridge_power, ridge_exempt_first)  # (2N,)
            return jnp.concatenate([res_data, res_ridge], axis=0)
        else:
            return res_data

    if use_jax_jacobian:

        def res_flat(Θ_flat):
            θ_structured = unravel(Θ_flat)
            return jax_resid(θ_structured)

        jac_fun = jax.jit(jax.jacrev(res_flat))
        jac_fun(Θ_flat).shape

        def jac_wrapper(Θ_flat):
            J = jac_fun(Θ_flat)
            return np.array(J)
    else:
        jac_wrapper = '2-point'

    def residual_wrapper(Θ_flat):
        θ_structured = unravel(Θ_flat)
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
    if getattr(config, 'use_log_state', False):
        az, bz = basis_bounds(config)
        z_grid = chebyshev_lobatto_nodes(az, bz, N)
        grid = from_basis_var(config, z_grid)
    else:
        a, b = config.lower_bound[0], config.upper_bound[0]
        grid = chebyshev_lobatto_nodes(a, b, N)  # include endpoints

    Θ_flat, unravel = ravel_pytree(θ_initial)
    fθ = make_weighted_residual_fn(
        config, params, grid, weights=None)  # weights=1

    def residual_wrapper(Θ_flat):
        θ_structured = unravel(Θ_flat)
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
# ------------------------
# Minimal driver
# ------------------------

if __name__ == "__main__":
    config = Config(N=(21,), lower_bound=(0.015,), upper_bound=(1.0 - 0.015,), use_log_state=True, use_s_derivatives=True)
    params_original = ModelParameters(γp=10.0, γa=10.0, αp=0.25, ρ=0.021, ψ=2.0, κ=0.02)
    θ_initial = initialize_chebyshev(config)

    ε = 0.0

    # Plotting style
    try:
        plt.style.use('seaborn-whitegrid')
    except Exception:
        pass

    for ε in np.arange(0.0, 1.01, 0.25):
        params = homotopy_transform(params_original, ε)

        if USE_LS:
            θ_new, resinfo, os_grid, weights = model_solution_least_squares(
                config, params, θ_initial,
                oversample_factor=2.0,
                nodes_kind=NODES_KIND,
                weighting=LS_WEIGHTING,
                ridge_enabled=RIDGE_ENABLED,
                ridge_lambda=RIDGE_LAMBDA,
                ridge_power=RIDGE_POWER,
                ridge_exempt_first=RIDGE_EXEMPT_FIRST,
                use_jax_jacobian=True
            )
            # RMSE per equation over data residuals only (exclude ridge)
            M = len(os_grid)
            fθ_data = make_weighted_residual_fn(config, params, os_grid, weights=weights)
            res_w = np.array(fθ_data(θ_new))  # shape (E*M,)
            E = res_w.size // M
            res_w_mat = res_w.reshape(M, E)
            per_eq_rmse = np.sqrt(np.mean(res_w_mat**2, axis=0))
            avg_rmse = float(np.mean(per_eq_rmse))
            print("LS: cost =", getattr(resinfo, "cost", "n/a"),
                  "| Avg RMSE =", avg_rmse)
        else:
            θ_new, resinfo, grid = model_solution_collocation(
                config, params, θ_initial)
            # RMSE per equation (no ridge in collocation path)
            N = len(grid)
            fθ_data = make_weighted_residual_fn(config, params, grid, weights=None)
            res = np.array(fθ_data(θ_new))  # shape (E*N,)
            E = res.size // N
            res_mat = res.reshape(N, E)
            per_eq_rmse = np.sqrt(np.mean(res_mat**2, axis=0))
            avg_rmse = float(np.mean(per_eq_rmse))
            print("Collocation:", resinfo.message,
                  "| Avg RMSE =", avg_rmse)

        θ_initial = θ_new
        # Quick diagnostic on an eval grid in basis var (z), mapped back to x
        nodes_fn = chebyshev_gauss_nodes if DIAGNOSTIC_NODES.lower().startswith("gauss") else chebyshev_lobatto_nodes
        if getattr(config, 'use_log_state', False):
            az, bz = basis_bounds(config)
            z_eval_grid = nodes_fn(az, bz, 1000)
            eval_grid = from_basis_var(config, z_eval_grid)
        else:
            a, b = config.lower_bound[0], config.upper_bound[0]
            eval_grid = nodes_fn(a, b, 1000)
        res_eval, outcomes = jax.vmap(
            partial(H, config, θ_new, params))(eval_grid)
        print("Max |residual| on eval grid:",
              float(jnp.max(jnp.abs(res_eval))))
        print("Epsilon:", ε)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

        # Common labels
        x_min, x_max = float(config.lower_bound[0]), float(config.upper_bound[0])
        x_label = 'State x'

        # (1,1): sigma_R
        ax = axes[0, 0]
        ax.plot(eval_grid, outcomes['σR'], color='C0', label='σ_R(x)')
        ax.set_title('Excess return volatility')
        ax.set_ylabel('σ_R')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

        # (1,2): q and baselines
        ax = axes[0, 1]
        q = price_dividend_ratio(params)
        ax.plot(eval_grid, outcomes['q'], color='C1', label='q(x)')
        # ax.axhline(y=q['qa'], color='tab:green', linestyle='--', alpha=0.7, label=f"q_a={q['qa']:.2f}")
        # ax.axhline(y=q['qp'], color='tab:blue', linestyle='--', alpha=0.7, label=f"q_b={q['qp']:.2f}")
        ax.set_title('Price–dividend ratio')
        ax.set_ylabel('q')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

        # (2,1): c_a
        ax = axes[1, 0]
        ax.plot(eval_grid, outcomes['ca'], color='C2', label='c_a(x)')
        ax.set_title('Consumption of agent a')
        ax.set_xlabel(x_label)
        ax.set_ylabel('c_a')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

        # (2,2): c_b
        ax = axes[1, 1]
        ax.plot(eval_grid, outcomes['cp'], color='C3', label='cp(x)')
        # ax.axhline(y=q['cb_lim'], color='tab:green', linestyle='--', alpha=0.7, label=f"c_b, lim={q['cb_lim']:.2f}")
        # ax.axhline(y=0.0, color='black', linestyle='--', alpha=0.4, label='0')
        ax.set_title('Consumption of agent p')
        ax.set_xlabel(x_label)
        ax.set_ylabel('c_p')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

        # x-limits
        for ax in axes[1, :]:
            ax.set_xlim(x_min, x_max)

        # Title and layout
        fig.suptitle(f'Epsilon = {ε:.2f}')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        plt.pause(1e-6)

# %%
# Example plotting code (commented out to avoid module-level execution)
# Uncomment and provide config, eval_grid, outcomes variables to use
"""
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

# Common labels
x_min, x_max = float(config.lower_bound[0]), float(config.upper_bound[0])
x_label = 'State x'

# (1,1): sigma_R
ax = axes[0, 0]
ax.plot(eval_grid, outcomes['σR'], color='C0', label='σ_R(x)')
ax.set_title('Excess return volatility')
ax.set_ylabel('σ_R')
ax.grid(True, alpha=0.3)
ax.legend(frameon=False)

# (1,2): q and baselines
ax = axes[0, 1]
q = price_dividend_ratio(params)
ax.plot(eval_grid, outcomes['q'], color='C1', label='q(x)')
# ax.axhline(y=q['qa'], color='tab:green', linestyle='--', alpha=0.7, label=f"q_a={q['qa']:.2f}")
# ax.axhline(y=q['qp'], color='tab:blue', linestyle='--', alpha=0.7, label=f"q_b={q['qp']:.2f}")
ax.set_title('Price–dividend ratio')
ax.set_ylabel('q')
ax.grid(True, alpha=0.3)
ax.legend(frameon=False)

# (2,1): c_a
ax = axes[1, 0]
ax.plot(eval_grid, outcomes['pi'], color='C2', label='c_a(x)')
ax.set_title('Consumption of agent a')
ax.set_xlabel(x_label)
ax.set_ylabel('c_a')
ax.grid(True, alpha=0.3)
ax.legend(frameon=False)

# (2,2): c_b
ax = axes[1, 1]
ax.plot(eval_grid, outcomes['r'], color='C3', label='cp(x)')
# ax.axhline(y=q['cb_lim'], color='tab:green', linestyle='--', alpha=0.7, label=f"c_b, lim={q['cb_lim']:.2f}")
# ax.axhline(y=0.0, color='black', linestyle='--', alpha=0.4, label='0')
ax.set_title('Consumption of agent p')
ax.set_xlabel(x_label)
ax.set_ylabel('c_p')
ax.grid(True, alpha=0.3)
ax.legend(frameon=False)

# x-limits
for ax in axes[1, :]:
    ax.set_xlim(x_min, x_max)

# Title and layout
fig.suptitle(f'Epsilon = {ε:.2f}')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
plt.pause(1e-6)
"""

# %%
# =============================================================================
# Example usage of s-derivatives for improved numerical stability:
# =============================================================================
#
# # Method 1: Use s-derivatives directly (more stable)
# dF_s = compute_derivatives_s_direct(config, θ, params, x, return_x=False)
# was, wps = dF_s['was'], dF_s['wps']  # First derivatives w.r.t. s
# wass, wpss = dF_s['wass'], dF_s['wpss']  # Second derivatives w.r.t. s
# 
# # Method 2: Use s-derivatives but also get x-derivatives when needed
# dF_s_with_x = compute_derivatives_s_direct(config, θ, params, x, return_x=True)
# # Now you have both s-derivatives and x-derivatives available
# 
# # Method 3: Use in HJB residual computation (controlled by config)
# config_s = config._replace(use_s_derivatives=True)   # Uses s-derivatives
# config_x = config._replace(use_s_derivatives=False)  # Uses original x-derivatives
# residuals, info = H(config_s, θ, params, x)  # Uses s-derivatives
# residuals, info = H(config_x, θ, params, x)  # Uses original x-derivatives
# 
# # Method 4: Compare both approaches for numerical accuracy
# differences = compare_derivative_methods(config, θ, params, x)
# print("Differences between methods:", differences)
#
# # Method 5: Test stability across different x values
# stability_results = test_s_derivatives_stability(config, θ, params)
# =============================================================================

# %%
