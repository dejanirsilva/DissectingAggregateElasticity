# ===== Dependencies =====
using LinearAlgebra
using NLsolve

# ===== Model parameters =====
struct Params
    ψ::Float64; ρ::Float64; μ::Float64; σ::Float64
    γa::Float64; γb::Float64
end

# ===== Grid in s = -log x =====
function make_grid(N::Int=200, S::Float64=8.0)
    s = range(0, S; length=N)
    x = exp.(-s)
    Δs = s[2]-s[1]
    return s, x, Δs
end

# 2nd-order finite differences in s (one-sided at ends)
function diffs_s(v, Δs)
    N = length(v)
    vs  = similar(v); vss = similar(v)
    @inbounds begin
        for i in 2:N-1
            vs[i]  = (v[i+1] - v[i-1])/(2Δs)
            vss[i] = (v[i+1] - 2v[i] + v[i-1])/(Δs^2)
        end
        vs[1]  = (-3v[1] + 4v[2] - v[3])/(2Δs)
        vss[1] = (2v[1] - 5v[2] + 4v[3] - v[4])/(Δs^2)
        vs[N]  = (3v[N] - 4v[N-1] + v[N-2])/(2Δs)
        vss[N] = (2v[N] - 5v[N-1] + 4v[N-2] - v[N-3])/(Δs^2)
    end
    return vs, vss
end

# ===== Pack / unpack helpers =====
pack(u,wa,wb) = [u; wa; wb]
function unpack(v, N)
    u  = view(v, 1:N)
    wa = view(v, N+1:2N)
    wb = view(v, 2N+1:3N)
    return u,wa,wb
end

# ===== Residual builder =====
"""
Residuals at all nodes for unknowns u(s)=log y, wa=log ca, wb=log cb.

Equations per node:
1) Agent a Euler (Eq. 14 with κ=0, includes Itô)
2) Agent b Euler
3) Market clearing: exp(u) = (1-x) exp(wa) + x exp(wb)
"""
function residual!(F, v, p::Params, s, x, Δs)
    N = length(x); ψ=p.ψ; ρ=p.ρ; μ=p.μ; σ=p.σ; γa=p.γa; γb=p.γb
    u,wa,wb = unpack(v, N)

    # Levels
    y  = @. exp(u)
    ca = @. exp(wa)
    cb = @. exp(wb)

    # Derivatives in s
    us,  uss  = diffs_s(u,  Δs)
    was, wass = diffs_s(wa, Δs)
    wbs, wbss = diffs_s(wb, Δs)

    # Convert to x-derivatives (using x = e^{-s})
    # u_x = -u_s / x ;  u_xx = (uss + us)/x^2 + (us/x)^2 - but we only need yxx/y = u_xx + u_x^2
    q   = .-us ./ x                          # q = y_x / y = u_x
    yxx_over_y = (uss .+ us) ./ (x.^2) .+ (us.^2) ./ (x.^2)  # = u_xx + u_x^2

    wax = .-was ./ x
    waxx_over = (wass .+ was) ./ (x.^2) .+ (was.^2) ./ (x.^2)

    wbx = .-wbs ./ x
    wbxx_over = (wbss .+ wbs) ./ (x.^2) .+ (wbs.^2) ./ (x.^2)

    # Harmonic risk aversion & portfolio tilts
    γbar = 1.0 ./ (x./γa .+ (1 .- x)./γb)
    αa   = γbar ./ γa
    αb   = γbar ./ γb

    # Exact volatility closure (κ = 0)
    σx = similar(x); σR = similar(x)
    @inbounds for i in eachindex(x)
        denom = 1 + x[i]*(αa[i]-1)*q[i]              # algebraic, not series
        σx[i] = x[i]*(αa[i]-1) * σ / denom
        σR[i] = σ - q[i]*σx[i]
    end
    σy = q .* σx
    π  = γbar .* (σR.^2)

    # State drift (κ=0 part only)
    μx = x .* (αa .- 1) .* (π .- σR.^2)

    # Drifts with Itô (ratio + Ito)
    μy  = q .* μx .+ 0.5 .* yxx_over_y .* (σx.^2)
    μca = wax .* μx .+ 0.5 .* waxx_over   .* (σx.^2)
    μcb = wbx .* μx .+ 0.5 .* wbxx_over   .* (σx.^2)

    # Optional ξ_j volatility add-ons (set to zero unless you have a spec)
    ξa = zero.(x)
    ξb = zero.(x)

    # Euler residuals (agent a, b) — your Eq. (14), κ=0
    Ra = ca .- ( ψ*ρ .+ (1-ψ).*( y .+ μ .+ σy.^2 .- σ.*σy .+ π.*(αa .- 1) .- 0.5*γa .* (σR.^2) .- μy ) .+ μca .+ ξa )
    Rb = cb .- ( ψ*ρ .+ (1-ψ).*( y .+ μ .+ σy.^2 .- σ.*σy .+ π.*(αb .- 1) .- 0.5*γb .* (σR.^2) .- μy ) .+ μcb .+ ξb )

    # Market clearing residual
    Ry = @. y - ((1-x)*ca + x*cb)

    F[:] = [Ra; Rb; Ry]
    return F
end

# ===== Solver driver =====
function solve_system(p::Params; N::Int=300, S::Float64=9.0, maxit::Int=200)
    s, x, Δs = make_grid(N, S)

    # Initial guess: homogeneous (single-agent) values for ca, cb; y from market clearing
    c0a = p.ρ + (1/p.ψ - 1)*(p.μ - 0.5*p.γa*p.σ^2)
    c0b = p.ρ + (1/p.ψ - 1)*(p.μ - 0.5*p.γb*p.σ^2)
    ca0 = fill(c0a, N)
    cb0 = fill(c0b, N)
    y0  = x .* cb0 .+ (1 .- x) .* ca0
    u0  = log.(y0 .+ 1e-14)  # keep positive

    v0 = pack(u0, log.(ca0), log.(cb0))
    F  = similar(v0)

    f!(F_, v_) = residual!(F_, v_, p, s, x, Δs)

    sol = nlsolve(f!, v0; method=:trust_region, xtol=1e-10, ftol=1e-10, iterations=maxit)
    u,wa,wb = unpack(sol.zero, N)

    return (; s, x, u, y=exp.(u), ca=exp.(wa), cb=exp.(wb), converged=sol.f_converged, iters=sol.iterations)
end

# ===== Example usage =====
p = Params(2.0, 0.02, 0.01, 0.2, 20.0, 2.0)   # (ψ, ρ, μ, σ, γ_a, γ_b)
sol = solve_system(p; N=400, S=10.0)
@show sol.converged, sol.iters
# Example diagnostics:
q = -diffs_s(sol.u, sol.s[2]-sol.s[1])[1] ./ sol.x
Bx = sol.x .* q
Hx = sol.x.^2 .* ( (diffs_s(q, sol.s[2]-sol.s[1])[1].*(-1) ./ sol.x) .+ q.^2 )

# === Diagnostics helpers (drop in the same file) =============================

"""
Compute B(x)=x*u_x and H(x)=x^2*(y_xx/y) on the solved grid,
plus Eq.(14) residuals per node for agents a,b and market clearing.
Returns a NamedTuple.
"""
function diagnostics(sol, p::Params)
    s, x, u = sol.s, sol.x, sol.u
    Δs = s[2]-s[1]
    # derivatives in s
    us, uss = diffs_s(u, Δs)
    # q = u_x = -(1/x) u_s
    q  = .-us ./ x
    # B = x q
    Bx = x .* q
    # y_xx / y = u_xx + u_x^2 = (uss+us)/x^2 + (us/x)^2
    Hx = (uss .+ us) ./ (x.^2) .+ (us.^2) ./ (x.^2)
    # rebuild residuals (so you can inspect them)
    F = similar([sol.u; log.(sol.ca); log.(sol.cb)])
    residual!(F, pack(sol.u, log.(sol.ca), log.(sol.cb)), p, s, x, Δs)
    N = length(x)
    Ra = view(F, 1:N)
    Rb = view(F, N+1:2N)
    Ry = view(F, 2N+1:3N)
    # nodewise residual magnitudes
    return (; x, Bx, Hx, q, Ra=copy(Ra), Rb=copy(Rb), Ry=copy(Ry))
end

"""
Convenience: scalar summary of residuals (infinity norm per block).
"""
function residual_summary(sol, p::Params)
    d = diagnostics(sol, p)
    (; Ra∞ = maximum(abs.(d.Ra)),
       Rb∞ = maximum(abs.(d.Rb)),
       Ry∞ = maximum(abs.(d.Ry)))
end

# === Continuation in heterogeneity ===========================================

"""
Linearly interpolates risk aversions from (γa0,γb0) to (γaT,γbT) in `K` steps.
Warm-starts each solve with the previous solution. Returns a vector of NamedTuples.
Optional keyword args forwarded to `solve_system` (e.g., N, S, maxit).
"""
function continue_heterogeneity(p0::Params, γaT::Float64, γbT::Float64;
                                K::Int=10, N::Int=300, S::Float64=9.0, maxit::Int=200)
    sols = Vector{Any}(undef, K+1)
    # first solve at start params
    p = Params(p0.ψ, p0.ρ, p0.μ, p0.σ, p0.γa, p0.γb)
    sol = solve_system(p; N=N, S=S, maxit=maxit)
    sols[1] = (p=p, sol=sol, res=residual_summary(sol, p))

    # linear schedule
    γa_sched = range(p0.γa, γaT; length=K+1)
    γb_sched = range(p0.γb, γbT; length=K+1)

    # use last solution as initial guess by seeding solve_system (small tweak below)
    v_guess = pack(sol.u, log.(sol.ca), log.(sol.cb))

    for k in 2:K+1
        p = Params(p0.ψ, p0.ρ, p0.μ, p0.σ, γa_sched[k], γb_sched[k])
        sol = solve_system_with_guess(p, v_guess; N=N, S=S, maxit=maxit)
        sols[k] = (p=p, sol=sol, res=residual_summary(sol, p))
        # update guess
        v_guess = pack(sol.u, log.(sol.ca), log.(sol.cb))
    end
    return sols
end

# Variant of solve_system that accepts an initial vector guess `v0`
function solve_system_with_guess(p::Params, v0; N::Int=300, S::Float64=9.0, maxit::Int=200)
    s, x, Δs = make_grid(N, S)
    F  = similar(v0)
    f!(F_, v_) = residual!(F_, v_, p, s, x, Δs)
    sol = nlsolve(f!, v0; method=:trust_region, xtol=1e-10, ftol=1e-10, iterations=maxit)
    u,wa,wb = unpack(sol.zero, N)
    return (; s, x, u, y=exp.(u), ca=exp.(wa), cb=exp.(wb),
             converged=sol.f_converged, iters=sol.iterations)
end

# Parameters (example)
p = Params(2.0, 0.02, 0.01, 0.20, 20.0, 2.0)  # (ψ,ρ,μ,σ, γ_a, γ_b)

# 1) Single solve
sol = solve_system(p; N=400, S=10.0)
println(residual_summary(sol, p))  # quick check of Eq.(14) & market clearing residuals
d = diagnostics(sol, p)
# d.Bx, d.Hx, d.q are your boundary diagnostics

# 2) Continuation to stronger heterogeneity
traj = continue_heterogeneity(p, 10.0, 1.5; K=8, N=400, S=10.0)
for (k,step) in enumerate(traj)
    @show k, step.p.γa, step.p.γb, step.res
end
last_sol = traj[end].sol
last_diag = diagnostics(last_sol, traj[end].p)

################################################################################

using Plots

"""
Plot diagnostics and solution profiles from `diagnostics` + `solve_system`.
"""
function plot_solution(sol, p; filename=nothing)
    d = diagnostics(sol, p)
    x = d.x

    plt1 = plot(x, d.Bx;
        lw=2, label="B(x) = x·q(x)",
        xlabel="x", ylabel="B(x)",
        xscale=:log10, title="Scaled slope B(x)")

    plt2 = plot(x, d.Hx;
        lw=2, label="H(x) = x²·yₓₓ/y",
        xlabel="x", ylabel="H(x)",
        xscale=:log10, title="Scaled curvature H(x)")

    plt3 = plot(x, sol.y;
        lw=2, label="p(x)",
        xlabel="x", ylabel="price-dividend ratio",
        xscale=:log10, title="Price–dividend ratio")

    plt4 = plot(x, sol.ca;
        lw=2, label="c_a(x)",
        xlabel="x", ylabel="consumption",
        xscale=:log10, title="cₐ and c_b")
    plot!(plt4, x, sol.cb; lw=2, label="c_b(x)")

    layout = @layout [a b; c d]
    pgrid = plot(plt1, plt2, plt3, plt4; layout, size=(900,600))
    if filename !== nothing
        savefig(pgrid, filename)
    end
    return pgrid
end

using Plots

p = Params(0.5, 0.02, 0.01, 0.20, 6.0, 2.0)
sol = solve_system(p; N=400, S=10.0)

# Quick residual check
println(residual_summary(sol, p))

# Make the plots
plt = plot_solution(sol, p; filename="solution_profiles.png")
display(plt)


################################################################################

function diagnostics(sol, p::Params)
    s, x, u = sol.s, sol.x, sol.u
    Δs = s[2]-s[1]
    ψ, ρ, μ, σ, γa, γb = p.ψ, p.ρ, p.μ, p.σ, p.γa, p.γb

    # derivatives in s
    us, uss = diffs_s(u, Δs)

    # q, B, H
    q  = .-us ./ x                      # q = u_x
    Bx = x .* q
    Hx = (uss .+ us) ./ (x.^2) .+ (us.^2) ./ (x.^2)   # y_xx / y

    # levels (for residuals)
    ca = sol.ca; cb = sol.cb; y = sol.y

    # harmonic RA & tilts
    γbar = 1.0 ./ (x./γa .+ (1 .- x)./γb)
    αa   = γbar ./ γa

    # exact closure
    σx = similar(x); σR = similar(x)
    @inbounds for i in eachindex(x)
        denom = 1 + x[i]*(αa[i]-1)*q[i]
        σx[i] = x[i]*(αa[i]-1) * σ / denom
        σR[i] = σ - q[i]*σx[i]
    end
    σy = q .* σx

    # rebuild residuals (so you can inspect them)
    F = similar([sol.u; log.(sol.ca); log.(sol.cb)])
    residual!(F, pack(sol.u, log.(sol.ca), log.(sol.cb)), p, s, x, Δs)
    N = length(x)
    Ra = view(F, 1:N) |> copy
    Rb = view(F, N+1:2N) |> copy
    Ry = view(F, 2N+1:3N) |> copy

    return (; x, q, Bx, Hx, σR, σy, Ra, Rb, Ry, y, ca, cb)
end

using Plots

function plot_solution(sol, p; filename=nothing)
    d = diagnostics(sol, p)
    x = d.x

    plt1 = plot(x, d.Bx; lw=2, label="B(x)=x·q(x)",
        xlabel="x", ylabel="B(x)", xscale=:log10, title="Scaled slope B(x)")

    plt2 = plot(x, d.Hx; lw=2, label="H(x)=x²·yₓₓ/y",
        xlabel="x", ylabel="H(x)", xscale=:log10, title="Scaled curvature H(x)")

    plt3 = plot(x, d.σR; lw=2, label="σ_R(x)",
        xlabel="x", ylabel="volatility", xscale=:log10, title="σ_R and σ_y")
    plot!(plt3, x, d.σy; lw=2, ls=:dash, label="σ_y(x)")

    plt4 = plot(x, d.y; lw=2, label="p(x)",
        xlabel="x", ylabel="price-dividend ratio", xscale=:log10, title="Price–dividend ratio")

    plt5 = plot(x, d.ca; lw=2, label="c_a(x)",
        xlabel="x", ylabel="consumption", xscale=:log10, title="cₐ and c_b")
    plot!(plt5, x, d.cb; lw=2, label="c_b(x)")

    layout = @layout [a b; c d; e]
    pgrid = plot(plt1, plt2, plt3, plt4, plt5; layout, size=(900,900))
    if filename !== nothing
        savefig(pgrid, filename)
    end
    return pgrid
end

p = Params(0.5, 0.02, 0.01, 0.20, 6.0, 2.0)
sol = solve_system(p; N=400, S=10.0)

println(residual_summary(sol, p))   # quick Eq.(14) + market-clearing check

plt = plot_solution(sol, p; filename="solution_profiles.png")
display(plt)