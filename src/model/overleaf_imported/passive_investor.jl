using LinearAlgebra, Printf
using NLsolve
using Plots

# -------------------------------
# Utilities: Chebyshev–Lobatto grid and differentiation
# -------------------------------

"""
cheb_lobatto(N)
  Return Chebyshev–Lobatto nodes t ∈ [-1,1] and the first-derivative matrix Dt w.r.t. t
  (Spectral differentiation, standard formula).
"""
function cheb_lobatto(N::Int)
    @assert N >= 2
    t = [cos(pi*j/(N-1)) for j in 0:N-1]
    c = [2; ones(N-2); 2] .* (-1).^(0:N-1)
    Dt = zeros(Float64, N, N)
    for i in 1:N, j in 1:N
        if i != j
            Dt[i,j] = (c[i]/c[j]) * ( (-1)^(i+j) / (t[i] - t[j]) )
        end
    end
    Dt[diagind(Dt)] .= 0.0
    Dt[1,1]   = (2*(N-1)^2 + 1)/6
    Dt[N,N]   = -Dt[1,1]
    for i in 2:N-1
        Dt[i,i] = -t[i]/(2*(1 - t[i]^2))
    end
    return t, Dt
end

"""
map_to_interval(t, a, b)
  Affine map from t ∈ [-1,1] to s ∈ [a,b].
  Returns s(t) and the scaling factor ds/dt.
"""
map_to_interval(t, a, b) = ((b - a).*(t .+ 1)./2 .+ a, (b - a)/2)

# -------------------------------
# Model primitives and helpers
# -------------------------------

Base.@kwdef struct Params
    # Preferences & technology
    psi::Float64 = 1.5         # EIS-related parameter (ψ)
    rho::Float64 = 0.02        # rate in c(.) block
    gamma0::Float64 = 5.0      # RA of passive type (enters ξ0)
    gamma1::Float64 = 5.0      # RA of active type

    mu::Float64 = 0.015        # dividend drift μ
    sigma::Float64 = 0.04      # dividend vol σ

    # Market structure / frictions
    alpha_p::Float64 = 0.30    # fixed risky share for passive type
    kappa::Float64 = 0.02      # mean-reversion intensity in x dynamics
    omega1::Float64 = 0.50     # active type stationary share target

    # Domain in s = -log x
    sL::Float64 = 1e-3         # left boundary (x ≈ 1)
    sR::Float64 = 8.0          # right boundary (x ≈ 0)

    # Numerics
    N::Int = 121               # number of collocation nodes (≥ 4). Use odd for symmetry.
    newton_tol::Float64 = 1e-10
    max_iter::Int = 200
    denom_eps::Float64 = 1e-10 # safeguard for denominators
end

"""
RA_cstar(gamma, psi, rho, mu, sigma)
  Representative-agent steady c* used for boundary anchoring.
  From outer solution (y* constant):
    c* = ρ + (ψ^{-1}-1)*( μ - 0.5*γ*σ^2 ).
"""
function RA_cstar(gamma, psi, rho, mu, sigma)
    return rho + (1/psi - 1)*(mu - 0.5*gamma*sigma^2)
end

# -------------------------------
# Collocation assembly
# -------------------------------

"""
assemble_system(p::Params)
  Build collocation nodes in s, the derivative matrices Ds, D2s, and initial guess for u0,u1.
"""
function assemble_system(p::Params)
    t, Dt = cheb_lobatto(p.N)
    s, dsdt = map_to_interval(t, p.sL, p.sR)
    Ds  = Dt .* (1/dsdt)              # derivative w.r.t. s
    D2s = Ds*Ds                       # second derivative w.r.t. s

    # Coordinates
    x = exp.(-s)                      # x = e^{-s}

    # Boundary anchors (values of u at boundaries)
    cstar1 = max(RA_cstar(p.gamma1, p.psi, p.rho, p.mu, p.sigma), 1e-6)
    cstar0 = cstar1                    # default: same anchor; tweak if desired
    u1L    = log(cstar1)               # at s = sL (x ≈ 1): active dominates
    u0R    = log(cstar0)               # at s = sR (x ≈ 0): bounded passive side

    # Initial guess: flat u0,u1 at anchors with slight tilt
    u0 = fill(u0R, p.N)
    u1 = fill(u1L, p.N)
    u0 .+= 0.02 .* (s .- p.sR)        # tiny slope towards interior
    u1 .+= 0.02 .* (s .- p.sL)

    return (s, x, Ds, D2s, u0, u1, u1L, u0R)
end


# -------------------------------
# Residual evaluation
# -------------------------------

"""
model_residual!(F, U, data, p::Params)
  Compute nonlinear residuals at collocation nodes.
  Unknown vector U stacks [u0; u1] at all N nodes.
  Residual vector F stacks [R0; R1] at interior nodes and boundary conditions at ends.

  Boundary conditions (default):
    u1(sL) = log c* (active RA limit),    (Dirichlet)
    u0(sR) = log c* (bounded inner limit), (Dirichlet)
    (Du1)(sL) = 0, (Du0)(sR) = 0          (Neumann)

  Interior nodes: enforce c_j equations for j=0,1.
"""
function model_residual!(F::AbstractVector{T}, U::AbstractVector{T}, data, p::Params) where {T}
    s, x, Ds, D2s, u1L, u0R = data.s, data.x, data.Ds, data.D2s, data.u1L, data.u0R
    N = length(s)

    # unpack unknowns
    u0 = @view U[1:N]
    u1 = @view U[N+1:2N]

    # compute derivatives w.r.t. s
    u0_s  = Ds * u0
    u1_s  = Ds * u1
    u0_ss = D2s * u0
    u1_ss = D2s * u1

    # levels of consumption-wealth ratios (Dual-friendly, strictly positive)
    c0 = exp.(u0)
    c1 = exp.(u1)
    c0 = ifelse.(c0 .> 1e-12, c0, 1e-12)
    c1 = ifelse.(c1 .> 1e-12, c1, 1e-12)

    # c_x and c_xx via chain rule x = e^{-s}
    # c_x  = -(c * u_s)/x
    # c_xx =  c * (u_ss + u_s + u_s^2)/x^2
    invx   = 1.0 ./ x
    invx2  = invx .* invx

    c0_x  = @. -(c0 * u0_s) * invx
    c1_x  = @. -(c1 * u1_s) * invx
    c0_xx = @.  c0 * (u0_ss + u0_s + u0_s^2) * invx2
    c1_xx = @.  c1 * (u1_ss + u1_s + u1_s^2) * invx2

    # dividend yield and its x-derivatives
    y   = @. (1 - x)*c0 + x*c1
    y_x = @. -c0 + (1 - x)*c0_x + c1 + x*c1_x
    y_xx= @. -2*(c0_x) + (1 - x)*c0_xx + 2*(c1_x) + x*c1_xx

    # w_s = d/ds log y(e^{-s}) = (-x * y_x)/y (guard y)
    y_safe = ifelse.(y .> 1e-14, y, 1e-14)
    ws  = .- (x .* y_x) ./ y_safe

    # sigma_x and sigma_R (s-chart formulas). Safeguard denom.
    denom = @. x - (1 - x)*(1 - p.alpha_p)*ws
    sign_denom = sign.(denom)
    sign_denom = ifelse.(sign_denom .== 0.0, 1.0, sign_denom)
    floor_val = max.(p.denom_eps, 0.05 .* x)  # scale floor with x to avoid blowups
    denom = ifelse.(abs.(denom) .> floor_val, denom, sign_denom .* floor_val)

    sigma_x = @. x*(1 - x)*(1 - p.alpha_p)*p.sigma / denom
    sigma_R = @. x * p.sigma / denom
    # cap extreme vols to avoid overflow during solver exploration
    sigma_cap = 50.0
    sigma_x = clamp.(sigma_x, -sigma_cap, sigma_cap)
    sigma_R = clamp.(sigma_R, -sigma_cap, sigma_cap)

    # portfolio shares
    alpha0 = fill(p.alpha_p, N)
    x_safe = ifelse.(x .> 1e-14, x, 1e-14)
    alpha1 = @. (1 - (1 - x)*p.alpha_p) / x_safe

    # sigma_y and mu_y
    sigma_y = @. y_x * sigma_x
    mu_y    = @. y_x * ( x*((0.0)) )   # placeholder for structure; will replace right below
    # Compute mu_x then finish mu_y
    # mu_x = x * ((π - σ_R^2) * (α1 - 1)) + κ*(ω1 - x)

    # hedging term for active
    sigma_c0 = @. (c0_x / c0) * sigma_x
    sigma_c1 = @. (c1_x / c1) * sigma_x

    # ϛ1 = (1 - 1/γ1) * ψ^{-1} * σ_{c1} / σ_R (guard σ_R)
    sigmaR_safe = ifelse.(abs.(sigma_R) .> p.denom_eps, sigma_R, sign.(sigma_R .+ p.denom_eps) .* p.denom_eps)
    zeta1    = @. (1 - 1/p.gamma1) * (1/p.psi) * (sigma_c1 / sigmaR_safe)

    # π from pricing condition: π = γ1 σ_R^2 (α1 - ϛ1)
    pi = @. p.gamma1 * (sigma_R^2) * (alpha1 - zeta1)

    # now μ_x and μ_y
    mu_x = @. x * ((pi - sigma_R^2) * (alpha1 - 1.0)) + p.kappa*(p.omega1 - x)
    mu_y = @. y_x * mu_x + 0.5 * y_xx * (sigma_x^2)

    # interest rate from identity: r = y + μ - μ_y + σ_y^2 - σ σ_y - π
    r = @. y + p.mu - mu_y + sigma_y^2 - p.sigma * sigma_y - pi

    # ξ_j pieces
    mu_c0 = @. c0_x * mu_x + 0.5 * c0_xx * (sigma_x^2)
    mu_c1 = @. c1_x * mu_x + 0.5 * c1_xx * (sigma_x^2)

    xi0 = @. mu_c0 + (1 - p.gamma0) * sigma_c0 * sigma_R * alpha0 + (p.psi - p.gamma0)/(1 - p.psi) * 0.5 * (sigma_c0^2)
    xi1 = @. mu_c1 + (1 - p.gamma1) * sigma_c1 * sigma_R * alpha1 + (p.psi - p.gamma1)/(1 - p.psi) * 0.5 * (sigma_c1^2)

    # c_j equations residuals
    target0 = @. p.psi*p.rho + (1 - p.psi) * ( r + pi*alpha0 - 0.5*p.gamma0*(sigma_R^2)*(alpha0^2) ) + xi0
    target1 = @. p.psi*p.rho + (1 - p.psi) * ( r + pi*alpha1 - 0.5*p.gamma1*(sigma_R^2)*(alpha1^2) ) + xi1

    R0 = @. c0 - target0
    R1 = @. c1 - target1

    # assemble residual vector with boundary conditions
    # order: [ interior R0; interior R1; BCs ]
    # Interior: nodes 2:(N-1)
    k1, k2 = 2, N-1
    m = (k2 - k1 + 1)

    idx = 1
    # R0 interior
    for i in k1:k2
        F[idx] = R0[i]; idx += 1
    end
    # R1 interior
    for i in k1:k2
        F[idx] = R1[i]; idx += 1
    end

    # Boundary conditions (4 equations)
    # Left (sL): active flat slope; dividend-yield flat (Dy = 0)
    F[idx]   = (Ds * u1)[1]; idx += 1          # u1_s(sL) = 0
    F[idx]   = (Ds * y)[1];  idx += 1          # y_s(sL)  = 0
    # Right (sR): passive flat slope; dividend-yield flat
    F[idx]   = (Ds * u0)[end]; idx += 1        # u0_s(sR) = 0
    F[idx]   = (Ds * y)[end]; idx += 1         # y_s(sR)  = 0

    return F
end

# -------------------------------
# Solve wrapper
# -------------------------------

mutable struct CollocationData
    s::Vector{Float64}
    x::Vector{Float64}
    Ds::Matrix{Float64}
    D2s::Matrix{Float64}
    u1L::Float64
    u0R::Float64
end

function solve_passive_model(p::Params; verbose::Bool=true)
    # assemble grid and initial guess
    s, x, Ds, D2s, u0, u1, u1L, u0R = assemble_system(p)
    data = CollocationData(s, x, Ds, D2s, u1L, u0R)

    N = length(s)
    U0 = vcat(u0, u1)

    # residual closure for NLsolve
    function F!(F, U)
        resize!(F, 2*(N-2) + 4)
        model_residual!(F, U, data, p)
    end

    function J!(J, U)
        # Let NLsolve approximate Jacobian by finite differences for simplicity
        nothing
    end

    # Use autodiff for Jacobian to avoid unstable finite-difference steps
    sol = nlsolve(F!, U0; xtol=p.newton_tol, ftol=p.newton_tol, iterations=p.max_iter, autodiff=:forward)

    resn = sol.residual_norm
    conv = isfinite(resn) && resn <= p.newton_tol
    if verbose
        @show conv, sol.iterations, resn
    end

    U = sol.zero
    u0 = U[1:N]
    u1 = U[N+1:2N]

    # Return a dictionary with useful outputs
    return (; s, x, u0, u1, Ds, D2s, params=p, status=(conv, sol.iterations, resn))
end

# -------------------------------
# Post-processing helpers
# -------------------------------

"""
recover_paths(sol)
  Given solution from solve_passive_model, compute c0(x), c1(x), y, π, r, σ_R.
"""
function recover_paths(sol)
    s, x, u0, u1 = sol.s, sol.x, sol.u0, sol.u1
    p = sol.params
    Ds = sol.Ds; D2s = sol.D2s

    c0 = exp.(u0)
    c1 = exp.(u1)
    c0 = ifelse.(c0 .> 1e-12, c0, 1e-12)
    c1 = ifelse.(c1 .> 1e-12, c1, 1e-12)

    u0_s  = Ds * u0; u1_s = Ds * u1
    u0_ss = D2s * u0;    u1_ss = D2s * u1

    invx  = 1.0 ./ x; invx2 = invx .* invx
    c0_x  = @. -(c0 * u0_s) * invx
    c1_x  = @. -(c1 * u1_s) * invx
    c0_xx = @.  c0 * (u0_ss + u0_s + u0_s^2) * invx2
    c1_xx = @.  c1 * (u1_ss + u1_s + u1_s^2) * invx2

    y    = @. (1 - x)*c0 + x*c1
    y_x  = @. -c0 + (1 - x)*c0_x + c1 + x*c1_x
    y_xx = @. -2*c0_x + (1 - x)*c0_xx + 2*c1_x + x*c1_xx
    y_safe = ifelse.(y .> 1e-14, y, 1e-14)
    ws   = .- (x .* y_x) ./ y_safe

    denom   = @. x - (1 - x)*(1 - p.alpha_p)*ws
    sign_denom = sign.(denom)
    sign_denom = ifelse.(sign_denom .== 0.0, 1.0, sign_denom)
    floor_val = max.(p.denom_eps, 0.05 .* x)
    denom   = ifelse.(abs.(denom) .> floor_val, denom, sign_denom .* floor_val)
    sigma_x = @. x*(1 - x)*(1 - p.alpha_p)*p.sigma / denom
    sigma_R = @. x * p.sigma / denom

    x_safe = ifelse.(x .> 1e-14, x, 1e-14)
    alpha1  = @. (1 - (1 - x)*p.alpha_p) / x_safe

    sigma_c1 = @. (c1_x / c1) * sigma_x
    sigmaR_safe = ifelse.(abs.(sigma_R) .> p.denom_eps, sigma_R, sign.(sigma_R .+ p.denom_eps) .* p.denom_eps)
    zeta1    = @. (1 - 1/p.gamma1) * (1/p.psi) * (sigma_c1 / sigmaR_safe)
    pi       = @. p.gamma1 * (sigma_R^2) * (alpha1 - zeta1)

    sigma_y  = @. y_x * sigma_x
    mu_x     = @. x * ((pi - sigma_R^2) * (alpha1 - 1.0)) + p.kappa*(p.omega1 - x)
    mu_y     = @. y_x * mu_x + 0.5 * y_xx * (sigma_x^2)
    r        = @. y + p.mu - mu_y + sigma_y^2 - p.sigma * sigma_y - pi

    return (; x, c0, c1, y, pi, r, sigma_R)
end

# -------------------------------
# Example run
# -------------------------------

p = Params(N = 21, sL = 1e-3, sR = 4.0)
sol = solve_passive_model(p)
println("Converged: ", sol.status)
out = recover_paths(sol)
# simple ASCII preview
println("x      c0      c1      y       pi      r       sigma_R")
for i in 1:10:length(out.x)
    @printf("%6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f\n",
            out.x[i], out.c0[i], out.c1[i], out.y[i], out.pi[i], out.r[i], out.sigma_R[i])
end

# 2x2 plot: sigma_R, y, c0, c1 vs x
plt = plot(layout=(2,2), legend=:topright)
plot!(plt, out.x, out.sigma_R, label="sigma_R(x)", xlabel="x", ylabel="sigma_R", subplot=1)
plot!(plt, out.x, out.y,       label="y(x)",       xlabel="x", ylabel="y",        subplot=2)
plot!(plt, out.x, out.c0,      label="c0(x)",      xlabel="x", ylabel="c0",       subplot=3)
plot!(plt, out.x, out.c1,      label="c1(x)",      xlabel="x", ylabel="c1",       subplot=4)
savefig(plt, joinpath(figdir, "sigmaR_y_c0_c1_vs_x.png"))
morprintln("Saved plot: ", joinpath(figdir, "sigmaR_y_c0_c1_vs_x.png"))

