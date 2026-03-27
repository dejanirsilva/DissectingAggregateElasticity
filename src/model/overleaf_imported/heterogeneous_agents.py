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
    r = ρ + 1 / ψ * μ - (1+1/ψ) * γ * σ**2 / 2
    π = γ * σ**2 / 2
    y = ρ - (1-1/ψ) * (μ - γ * σ**2 / 2)
    ca = ρ - (1-1/ψ) * (μ - γa * σ**2 / 2)
    cb = ρ - (1-1/ψ) * (μ - γb * σ**2 / 2)
    lca = jnp.log(ca+ 1e-12)
    lcb = jnp.log(cb+ 1e-12)
    pd = 1/y
    ly = jnp.log(y+ 1e-12)
    return dict(r=r, π=π, y=y, pd=pd, lca = lca, lcb = lcb, ly = ly, ca = ca, cb = cb)

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
        s_min = -np.log(x_max+ 1e-12)
        s_max = -np.log(x_min+ 1e-12)
        return float(s_min), float(s_max)
    else:
        return float(config.lower_bound[0]), float(config.upper_bound[0])


def to_basis_var(config: Config, x):
    """Map physical state x to basis variable z (x or s).
    z = x if not using log-state; z = -log(x) otherwise."""
    return -jnp.log(x+ 1e-12) if getattr(config, 'use_log_state', False) else x


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
    θb = jnp.zeros(N_total)
    lca = benchmark(ModelParameters())['lca'] # initial condition
    lcb = benchmark(ModelParameters())['lcb'] # initial condition
    θa = θa.at[0].set(lca)
    θb = θb.at[0].set(lcb)
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
    wb = _eval_series(θ['θb'], x, a, b, config)
    ca = jnp.exp(wa)
    cb = jnp.exp(wb)
    return dict(ca=ca, cb=cb, wa = wa, wb = wb)

def compute_derivatives(config: Config, θ, params: ModelParameters, x, return_s: bool = False):
    def wa_fn(x): return assemble_functions(config, θ, params, x)['wa']
    def wb_fn(x): return assemble_functions(config, θ, params, x)['wb']

    wax_fn = jax.jacfwd(wa_fn)
    wbx_fn = jax.jacfwd(wb_fn)

    waxx_fn = jax.jacfwd(wax_fn)
    wbxx_fn = jax.jacfwd(wbx_fn)

    wax = wax_fn(x)
    wbx = wbx_fn(x)
    waxx = waxx_fn(x)
    wbxx = wbxx_fn(x)

    d = dict(
        wax=wax, wbx=wbx,
        waxx=waxx, wbxx=wbxx
    )

    # Optionally include derivatives with respect to s = -log(x):
    # w_s = -x * w_x; w_ss = x * w_x + x^2 * w_xx
    if return_s:
        xs = jnp.asarray(x)
        was = -xs * wax
        wbs = -xs * wbx
        wass = xs * wax + (xs ** 2) * waxx
        wbss = xs * wbx + (xs ** 2) * wbxx
        d.update(dict(was=was, wbs=wbs, wass=wass, wbss=wbss))

    return d

# %%
# ------------------------
# HJB residual at a single x
# ------------------------

def homotopy_transform(params, ε):
    γ_average = (params.γa+params.γb)/2
    γa = γ_average * (1.0 - ε) + params.γa * ε
    γb = γ_average * (1.0 - ε) + params.γb * ε
    return params._replace(γa=γa, γb=γb)


def H(config: Config, θ, params: ModelParameters, x):
    ρ, ψ, σ, γa, γb, θb, μ, κ = params.ρ, params.ψ, params.σ, params.γa, params.γb, params.θb, params.μ, params.κ

    # Wealth share
    xa, xb = 1-x, x

    # Levels and derivatives
    F   = assemble_functions(config, θ, params, x)
    dF  = compute_derivatives(config, θ, params, x)

    # Derivatives
    ca, cb        = F['ca'], F['cb']
    wax, wbx      = dF['wax'], dF['wbx']
    waxx, wbxx    = dF['waxx'], dF['wbxx']

    ca, cb        = jnp.squeeze(ca), jnp.squeeze(cb)
    wax, wbx      = jnp.squeeze(wax), jnp.squeeze(wbx)
    waxx, wbxx    = jnp.squeeze(waxx), jnp.squeeze(wbxx)

    cax_over_ca   = wax
    cbx_over_cb   = wbx
    caxx_over_ca  = waxx + wax ** 2
    cbxx_over_cb  = wbxx + wbx ** 2

    y             = xb * cb + xa * ca
    yx_over_y     = (cb-ca) / y + (xb * cb / y * cbx_over_cb +
                                   xa * ca / y * cax_over_ca)
    yxx_over_y    = 2.0 * (cb * cbx_over_cb - ca * cax_over_ca) / y + (
                            xb * cb / y * cbxx_over_cb +
                            xa * ca / y * caxx_over_ca)

    γt          = 1.0 / ((1.0 - x)/γa + x/γb)

    χ           = (γt/γa -γt/γb) * (yx_over_y +
                                    xb * (1-1/γb)/(ψ-1) * cbx_over_cb +
                                    xa * (1-1/γa)/(ψ-1) * cax_over_ca) + (
                                        (1-1/γb)/(ψ-1) * cbx_over_cb -
                                        (1-1/γa)/(ψ-1) * cax_over_ca
                                    )

    # Diffusion terms
    σx  = x * (1.0 - x) * (γt/γb -γt/γa) * σ / (1.0 - x * (1.0 - x) * χ)
    σca = cax_over_ca * σx
    σcb = cbx_over_cb * σx
    σy  = yx_over_y * σx
    σR  = σ - σy

    # Hedging demand
    va = (1-1/γa)/(ψ-1) * cax_over_ca * σx / σR
    vb = (1-1/γb)/(ψ-1) * cbx_over_cb * σx / σR

    # Risk premium
    pi = γt * (σR ** 2) * (1 - xa * va - xb * vb )

    # Portfolio share
    pa = γt / γa * (1 - xa * va - xb * vb ) + va
    pb = γt / γb * (1 - xa * va - xb * vb ) + vb

    # Drift terms
    μx  = x * (1.0 - x) * ((pb - pa) * (pi - σR ** 2) + ca - cb) - κ * (x - θb)
    μca = cax_over_ca * μx + 0.5 * caxx_over_ca * σx**2
    μcb = cbx_over_cb * μx + 0.5 * cbxx_over_cb * σx**2
    μy  = yx_over_y * μx + 0.5 * yxx_over_y * σx**2

    # Interest rate
    r = y + μ - μy + σy ** 2 - σ*σy - pi

    # Consumption-wealth ratio
    xia = μca + (1-γa) * σca * σR * pa + (ψ-γa)/(1-ψ) * (σca ** 2) / 2
    xib = μcb + (1-γb) * σcb * σR * pb + (ψ-γb)/(1-ψ) * (σcb ** 2) / 2
    ca_tgt = ψ * ρ + (1.0 - ψ) * (r + pi * pa - γa * (pa * σR) ** 2 / 2) + xia
    cb_tgt = ψ * ρ + (1.0 - ψ) * (r + pi * pb - γb * (pb * σR) ** 2 / 2) + xib

    residuals_a = ca - ca_tgt
    residuals_b = cb - cb_tgt
    residuals = jnp.array([residuals_a, residuals_b])

    return residuals, dict(
        q=1/y, ca=ca, cb=cb, σR=σR,
        σx=σx, σca=σca, σcb=σcb, σy=σy,
        μx=μx, μca=μca, μcb=μcb, μy=μy,
        pi=pi, pa=pa, pb=pb, r=r,
        residuals=residuals
    )
def price_dividend_ratio(p):
    ya = p.ρ + (1/p.ψ-1) * (p.μ - p.γa * p.σ**2/2)
    yb = p.ρ + (1/p.ψ-1) * (p.μ - p.γb * p.σ**2/2)

    πa = p.γa * p.σ**2
    πb = p.γb * p.σ**2
    αb = πa / (p.γb * p.σ**2)
    αa = πb / (p.γa * p.σ**2)

    cb_lim = p.ψ * p.ρ + (1.0 - p.ψ) * (ya + p.μ + πa * (αb - 1.0) - p.γb * p.σ**2 * αb ** 2 / 2)
    ca_lim = p.ψ * p.ρ + (1.0 - p.ψ) * (yb + p.μ + πb * (αa - 1.0) - p.γa * p.σ**2 * αa ** 2 / 2)
    return dict(qa=1/ya, qb=1/yb, ca_lim=ca_lim, cb_lim=cb_lim, ca = ya, cb = yb, αa=αa, αb=αb)

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
    config = Config(N=(11,), lower_bound=(0.01/128,), upper_bound=(1.00-0.01/128,), use_log_state=True)
    params_original = ModelParameters(γb=1.5/2, γa=30.0, ρ=0.021, ψ=2.0, κ=0.02)
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
        # ax.axhline(y=q['qb'], color='tab:blue', linestyle='--', alpha=0.7, label=f"q_b={q['qb']:.2f}")
        ax.set_title('Price–dividend ratio')
        ax.set_ylabel('q')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

        # (2,1): c_a
        ax = axes[1, 0]
        ax.plot(eval_grid, outcomes['pi']/outcomes['σR'], color='C2', label='c_a(x)')
        ax.set_title('Sharpe ratio')
        ax.set_xlabel(x_label)
        ax.set_ylabel('η')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

        # (2,2): c_b
        ax = axes[1, 1]
        ax.plot(eval_grid, outcomes['cb'], color='C3', label='cb(x)')
        # ax.axhline(y=q['cb_lim'], color='tab:green', linestyle='--', alpha=0.7, label=f"c_b, lim={q['cb_lim']:.2f}")
        ax.axhline(y=0.0, color='black', linestyle='--', alpha=0.4, label='0')
        ax.set_title('Consumption of agent b')
        ax.set_xlabel(x_label)
        ax.set_ylabel('c_b')
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
# ax.axhline(y=q['qb'], color='tab:blue', linestyle='--', alpha=0.7, label=f"q_b={q['qb']:.2f}")
ax.set_title('Price–dividend ratio')
ax.set_ylabel('q')
ax.grid(True, alpha=0.3)
ax.legend(frameon=False)

# (2,1): c_a
ax = axes[1, 0]
ax.plot(eval_grid, outcomes['pi']/outcomes['σR'], color='C2', label='c_a(x)')
ax.set_title('Sharpe ratio')
ax.set_xlabel(x_label)
ax.set_ylabel('η')
ax.grid(True, alpha=0.3)
ax.legend(frameon=False)

# (2,2): c_b
ax = axes[1, 1]
ax.plot(eval_grid, outcomes['cb'], color='C3', label='cb(x)')
# ax.axhline(y=q['cb_lim'], color='tab:green', linestyle='--', alpha=0.7, label=f"c_b, lim={q['cb_lim']:.2f}")
ax.axhline(y=0.0, color='black', linestyle='--', alpha=0.4, label='0')
ax.set_title('Consumption of agent b')
ax.set_xlabel(x_label)
ax.set_ylabel('c_b')
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

# %%
