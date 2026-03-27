# Local perturbation at x̄ using TaylorSeries.jl (ε, δ)-expansion
# δ ≡ x - x̄. We build c_j(x;ε) as a bivariate Taylor polynomial in (ε,δ):
#   c_j(x;ε) = c0 + Σ_{k=1..K} Σ_{ℓ=0..L} a_{j,k,ℓ} ε^k δ^ℓ
# Coefficients a_{j,k,ℓ} are solved by enforcing your equilibrium equations
# (Eqs. 10, 14, 15 with definitions 5, 9–13) after expanding all objects as
# TaylorN polynomials and matching the ε^k δ^ℓ getcoeffficients at each order k.
# This uses true multivariate Taylor calculus, so y_x and y_xx are obtained
# automatically from δ-derivatives (no Chebyshev needed).

module PerturbationLocalTaylor

export Params, solve_local, eval_local

using LinearAlgebra
using TaylorSeries
using TaylorSeries: getcoeff

# -----------------------
# Parameters
# -----------------------
Base.@kwdef mutable struct Params
    ρ::Float64      = 0.02
    ψ::Float64      = 2.00
    μ::Float64      = 0.02
    σ::Float64      = 0.10
    γ::Float64      = 5.0
    γhat1::Float64  = -0.3
    γhat2::Float64  = +0.3
    κhat::Float64   = 0.02
    ω1::Float64     = 0.5
end

c0(p::Params) = p.ρ + (1/p.ψ - 1.0) * (p.μ - 0.5*p.γ*p.σ^2)

# Helpers for heterogeneous γ and κ
γs_and_κ(p::Params, ε) = ( p.γ*(1 + p.γhat1*ε), p.γ*(1 + p.γhat2*ε), p.κhat*ε )
γbar(x, γ1, γ2) = 1 / ( x*(1/γ1) + (1-x)*(1/γ2) )

# -----------------------
# Core: build residuals as TaylorN polynomials at (ε,δ)
# -----------------------
function residuals_taylor(a1::Matrix{Float64}, a2::Matrix{Float64}; K::Int, L::Int, xbar::Float64, p::Params)
    # a1, a2 are (K, L+1) matrices of coefficients a_{k,ℓ} for c1 and c2 (k starts at 1)
    # Construct multivariate Taylor variables: v[1]=ε (order K), v[2]=δ (order L)
    # Set order high enough to accommodate all operations (squaring, etc.)
    order = max(2*K, 2*L, K+L+2)
    v = set_variables("ε δ"; numvars=2, order=order)
    ε, δ = v[1], v[2]

    # c1, c2 series
    c0val = c0(p)
    c1 = TaylorN(c0val, order)
    c2 = TaylorN(c0val, order)
    for k in 1:K
        for ℓ in 0:L
            c1 += a1[k,ℓ+1] * (ε^k) * (δ^ℓ)
            c2 += a2[k,ℓ+1] * (ε^k) * (δ^ℓ)
        end
    end

    # x and weights
    x  = xbar + δ
    x1 = x
    x2 = 1 - x

    # y(x;ε) and derivatives via δ
    y  = x1*c1 + x2*c2
    yδ = differentiate(y, 2)             # ∂/∂δ acts as ∂/∂x
    yδδ = differentiate(yδ, 2)

    # Heterogeneous γ, κ
    γ1, γ2, κ = γs_and_κ(p, ε)

    σ = p.σ; μ = p.μ; ψ = p.ψ

    # Close σ_x – α – σ_R loop (Eq. 12) by fixed-point in TaylorN
    function close_sigmax(y, yδ, x)
        σx = zero(y)
        α1 = one(y)
        α2 = one(y)
        for _ in 1:8
            px_over_p = - yδ / y                   # p_x/p
            σp = px_over_p * σx                    # (p_x/p) σ_x
            σR = σ + σp
            γ̄ = γbar(x, γ1, γ2)
            π  = γ̄ * (σR^2)
            α1 = π / (γ1 * (σR^2))
            α2 = π / (γ2 * (σR^2))
            denom = 1 - x*(α1 - 1)*px_over_p      # from rearranged Eq. (12)
            σx = ( x*(α1 - 1) * σ ) / denom
        end
        return σx, α1, α2
    end

    σx, α1, α2 = close_sigmax(y, yδ, x)

    px_over_p  = - yδ / y
    yxx_term   = yδδ             # since δ-derivative equals x-derivative

    σp = px_over_p * σx
    σR = σ + σp

    γ̄ = γbar(x, γ1, γ2)
    π  = γ̄ * (σR^2)

    # x-drift (Eq. 11)
    μx = x * ( (π - σR^2)*(α1 - 1) + κ*(p.ω1 - x)/x )

    # Itô for y(x):
    σy = yδ * σx
    μy = yδ * μx + 0.5 * yxx_term * (σx^2)

    # μ_p via p_x/p, p_xx/p
    pxx_over_p = (px_over_p^2) - (yxx_term / y)
    μp = px_over_p * μx + 0.5 * pxx_over_p * (σx^2)

    r  = y + μ + μp + σ*σp - π        # Eq. (10)

    # Eq. (14): residuals Rj = c_j - RHS_j
    rhs_c1 = ψ*p.ρ + (1 - ψ) * ( y + μ - μy + (σy^2) - σ*σy + π*(α1 - 1) - 0.5*γ1*(σR^2)*(α1^2) )
    rhs_c2 = ψ*p.ρ + (1 - ψ) * ( y + μ - μy + (σy^2) - σ*σy + π*(α2 - 1) - 0.5*γ2*(σR^2)*(α2^2) )

    R1 = c1 - rhs_c1
    R2 = c2 - rhs_c2

    # Eq. (15): y-consistency residual (soft)
    term = μ - μy + (σy^2) - σ*σy - ( x*(γ1/2)*(σR^2)*(α1^2) + (1-x)*(γ2/2)*(σR^2)*(α2^2) )
    Ry = y - ( p.ρ + (1/ψ - 1)*term )

    return R1, R2, Ry
end

# -----------------------
# Solve getcoeffficients order-by-order in ε, matching δ up to L
# -----------------------
function solve_local(; K::Int=2, L::Int=2, xbar::Float64=0.5, p::Params=Params())
    # Unknowns a_{j,k,ℓ} arranged in matrices A1[k,ℓ+1], A2[k,ℓ+1]
    A1 = zeros(K, L+1)
    A2 = zeros(K, L+1)

    # Loop over k and solve linear system for a_{·,k,·}
    for k in 1:K
        # Build base residuals with current known coeffs (k-th row zero)
        R1, R2, Ry = residuals_taylor(A1, A2; K=K, L=L, xbar=xbar, p=p)
        # Collect coefficients at ε^k δ^ℓ into a right-hand side
        # TaylorN stores multivariate coefficients with factorial normalization;
        # we can read them via getcoeff(R, [k,ℓ]).
        rhs = zeros(3*(L+1))
        for ℓ in 0:L
            rhs[ℓ+1]        = -getcoeff(R1, [k, ℓ])
            rhs[(L+1)+ℓ+1]  = -getcoeff(R2, [k, ℓ])
            rhs[2*(L+1)+ℓ+1]= -getcoeff(Ry, [k, ℓ])
        end

        # Build Jacobian by unit-bump each a_{1,k,ℓ} and a_{2,k,ℓ}
        J = zeros(3*(L+1), 2*(L+1))
        for ℓ in 0:L
            # bump a1_{k,ℓ}
            A1b = copy(A1); A2b = copy(A2)
            A1b[k,ℓ+1] = A1b[k,ℓ+1] + 1.0
            R1b, R2b, Ryb = residuals_taylor(A1b, A2b; K=K, L=L, xbar=xbar, p=p)
            for m in 0:L
                J[m+1,           ℓ+1] = getcoeff(R1b, [k, m]) - getcoeff(R1, [k, m])
                J[(L+1)+m+1,     ℓ+1] = getcoeff(R2b, [k, m]) - getcoeff(R2, [k, m])
                J[2*(L+1)+m+1,   ℓ+1] = getcoeff(Ryb, [k, m]) - getcoeff(Ry, [k, m])
            end

            # bump a2_{k,ℓ}
            A1b = copy(A1); A2b = copy(A2)
            A2b[k,ℓ+1] = A2b[k,ℓ+1] + 1.0
            R1b, R2b, Ryb = residuals_taylor(A1b, A2b; K=K, L=L, xbar=xbar, p=p)
            for m in 0:L
                J[m+1,           (L+1)+ℓ+1] = getcoeff(R1b, [k, m]) - getcoeff(R1, [k, m])
                J[(L+1)+m+1,     (L+1)+ℓ+1] = getcoeff(R2b, [k, m]) - getcoeff(R2, [k, m])
                J[2*(L+1)+m+1,   (L+1)+ℓ+1] = getcoeff(Ryb, [k, m]) - getcoeff(Ry, [k, m])
            end
        end

        θ = J \ rhs
        A1[k, :] .= θ[1:(L+1)]
        A2[k, :] .= θ[(L+2):end]
    end

    return (A1=A1, A2=A2, K=K, L=L, xbar=xbar, p=p)
end

# -----------------------
# Evaluate c1,c2 at (x, ε) from local expansion
# -----------------------
function eval_local(sol, x::Float64, εval::Float64)
    A1, A2 = sol.A1, sol.A2
    K, L, xbar, p = sol.K, sol.L, sol.xbar, sol.p
    δ = x - xbar
    c = c0(p)
    c1 = c; c2 = c
    for k in 1:K
        εk = εval^k
        for ℓ in 0:L
            δℓ = δ^ℓ
            c1 += A1[k,ℓ+1] * εk * δℓ
            c2 += A2[k,ℓ+1] * εk * δℓ
        end
    end
    return c1, c2
end

end # module

module PerturbationPlots

using ..PerturbationLocalTaylor
using Plots
using Statistics: mean

include(joinpath(@__DIR__, "perturbation_closed_form.jl"))
using .PerturbationClosedForm

function generate_plot1(; p=PerturbationLocalTaylor.Params(), xbar=0.5, L=10, ε=1.0,
                        gammahat1::Union{Nothing,Float64}=nothing,
                        gammahat2::Union{Nothing,Float64}=nothing,
                        kappahat::Union{Nothing,Float64}=nothing)
    p_eff = if gammahat1 === nothing && gammahat2 === nothing && kappahat === nothing
        p
    else
        PerturbationLocalTaylor.Params(
            ρ=p.ρ, ψ=p.ψ, μ=p.μ, σ=p.σ, γ=p.γ,
            γhat1 = (gammahat1 === nothing ? p.γhat1 : gammahat1),
            γhat2 = (gammahat2 === nothing ? p.γhat2 : gammahat2),
            κhat  = (kappahat === nothing ? p.κhat : kappahat),
            ω1=p.ω1
        )
    end
    sol_1 = PerturbationLocalTaylor.solve_local(K=1, L=L, xbar=xbar, p=p_eff)
    sol_2 = PerturbationLocalTaylor.solve_local(K=2, L=L, xbar=xbar, p=p_eff)
    x_vals = range(0.01, 0.95, length=50)

    c1_vals_taylor_1, c2_vals_taylor_1 = Float64[], Float64[]
    c1_vals_taylor_2, c2_vals_taylor_2 = Float64[], Float64[]
    c1_vals_closed_1, c2_vals_closed_1 = Float64[], Float64[]
    c1_vals_closed_2, c2_vals_closed_2 = Float64[], Float64[]

    p_cf = PerturbationClosedForm.Params(
        ρ=p_eff.ρ, ψ=p_eff.ψ, μ=p_eff.μ, σ=p_eff.σ, γ=p_eff.γ,
        γhat1=p_eff.γhat1, γhat2=p_eff.γhat2, κhat=p_eff.κhat, ω1=p_eff.ω1
    )

    for x in x_vals
        c1_t1, c2_t1 = PerturbationLocalTaylor.eval_local(sol_1, x, ε)
        push!(c1_vals_taylor_1, c1_t1)
        push!(c2_vals_taylor_1, c2_t1)

        c1_t2, c2_t2 = PerturbationLocalTaylor.eval_local(sol_2, x, ε)
        push!(c1_vals_taylor_2, c1_t2)
        push!(c2_vals_taylor_2, c2_t2)

        cf_result = PerturbationClosedForm.series_at_x(x, ε, p_cf)
        push!(c1_vals_closed_1, cf_result.y0 + cf_result.c1_1)
        push!(c2_vals_closed_1, cf_result.y0 + cf_result.c2_1)
        push!(c1_vals_closed_2, cf_result.c1)
        push!(c2_vals_closed_2, cf_result.c2)
    end

    pplt = plot(x_vals, c1_vals_taylor_1, label="c₁ - 1st order (Taylor)", linewidth=2, color=:blue, linestyle=:dash)
    plot!(pplt, x_vals, c2_vals_taylor_1, label="c₂ - 1st order (Taylor)", linewidth=2, color=:red, linestyle=:dash)
    plot!(pplt, x_vals, c1_vals_taylor_2, label="c₁ - 2nd order (Taylor)", linewidth=3, color=:darkblue, linestyle=:solid)
    plot!(pplt, x_vals, c2_vals_taylor_2, label="c₂ - 2nd order (Taylor)", linewidth=3, color=:darkred, linestyle=:solid)
    plot!(pplt, x_vals, c1_vals_closed_1, label="c₁ - 1st order (Closed-form)", linewidth=2, color=:lightblue, linestyle=:dash, markershape=:square, markersize=2, markerstrokewidth=0)
    plot!(pplt, x_vals, c2_vals_closed_1, label="c₂ - 1st order (Closed-form)", linewidth=2, color=:lightcoral, linestyle=:dash, markershape=:square, markersize=2, markerstrokewidth=0)
    plot!(pplt, x_vals, c1_vals_closed_2, label="c₁ - 2nd order (Closed-form)", linewidth=3, color=:lightblue, linestyle=:solid, markershape=:circle, markersize=3, markerstrokewidth=0)
    plot!(pplt, x_vals, c2_vals_closed_2, label="c₂ - 2nd order (Closed-form)", linewidth=3, color=:lightcoral, linestyle=:solid, markershape=:circle, markersize=3, markerstrokewidth=0)
    xlabel!(pplt, "Wealth Share (x)")
    ylabel!(pplt, "Consumption Rate")
    title!(pplt, "First and Second-order Perturbation: Taylor vs Closed-form\n(ψ=$(p_eff.ψ), σ=$(p_eff.σ), κ̂=$(p_eff.κhat))")
    plot!(pplt, [xbar], [PerturbationLocalTaylor.eval_local(sol_2, xbar, ε)[1]], seriestype=:scatter, color=:black, markersize=5, label="x̄ = $(xbar)")
    return pplt
end

function generate_plot2(; p=PerturbationLocalTaylor.Params(), xbar=0.5, L=10, ε=1.0,
                        gammahat1::Union{Nothing,Float64}=nothing,
                        gammahat2::Union{Nothing,Float64}=nothing,
                        kappahat::Union{Nothing,Float64}=nothing)
    # Optionally override heterogeneity parameters
    if gammahat1 === nothing && gammahat2 === nothing && kappahat === nothing
        p_eff = p
    else
        p_eff = PerturbationLocalTaylor.Params(
            ρ=p.ρ, ψ=p.ψ, μ=p.μ, σ=p.σ, γ=p.γ,
            γhat1 = (gammahat1 === nothing ? p.γhat1 : gammahat1),
            γhat2 = (gammahat2 === nothing ? p.γhat2 : gammahat2),
            κhat  = (kappahat === nothing ? p.κhat : kappahat),
            ω1=p.ω1
        )
    end

    sol_0 = PerturbationLocalTaylor.solve_local(K=0, L=L, xbar=xbar, p=p_eff)
    sol_1 = PerturbationLocalTaylor.solve_local(K=1, L=L, xbar=xbar, p=p_eff)
    sol_2 = PerturbationLocalTaylor.solve_local(K=2, L=L, xbar=xbar, p=p_eff)
    sol_3 = PerturbationLocalTaylor.solve_local(K=3, L=L, xbar=xbar, p=p_eff)
    sol_4 = PerturbationLocalTaylor.solve_local(K=4, L=L, xbar=xbar, p=p_eff)
    sol_5 = PerturbationLocalTaylor.solve_local(K=5, L=L, xbar=xbar, p=p_eff)
    x_vals = range(0.01, 0.95, length=50)

    c1_vals_0, c2_vals_0 = Float64[], Float64[]
    c1_vals_1, c2_vals_1 = Float64[], Float64[]
    c1_vals_2, c2_vals_2 = Float64[], Float64[]
    c1_vals_3, c2_vals_3 = Float64[], Float64[]
    c1_vals_4, c2_vals_4 = Float64[], Float64[]
    c1_vals_5, c2_vals_5 = Float64[], Float64[]

    for x in x_vals
        c1_0, c2_0 = PerturbationLocalTaylor.eval_local(sol_0, x, ε)
        push!(c1_vals_0, c1_0); push!(c2_vals_0, c2_0)
        c1_1, c2_1 = PerturbationLocalTaylor.eval_local(sol_1, x, ε)
        push!(c1_vals_1, c1_1); push!(c2_vals_1, c2_1)
        c1_2, c2_2 = PerturbationLocalTaylor.eval_local(sol_2, x, ε)
        push!(c1_vals_2, c1_2); push!(c2_vals_2, c2_2)
        c1_3, c2_3 = PerturbationLocalTaylor.eval_local(sol_3, x, ε)
        push!(c1_vals_3, c1_3); push!(c2_vals_3, c2_3)
        c1_4, c2_4 = PerturbationLocalTaylor.eval_local(sol_4, x, ε)
        push!(c1_vals_4, c1_4); push!(c2_vals_4, c2_4)
        c1_5, c2_5 = PerturbationLocalTaylor.eval_local(sol_5, x, ε)
        push!(c1_vals_5, c1_5); push!(c2_vals_5, c2_5)
    end

    p1 = plot(x_vals, c1_vals_0, label="0th order", linewidth=2, color=:lightblue, linestyle=:dot)
    plot!(p1, x_vals, c1_vals_1, label="1st order", linewidth=2, color=:blue, linestyle=:dash)
    plot!(p1, x_vals, c1_vals_2, label="2nd order", linewidth=3, color=:darkblue, linestyle=:solid)
    plot!(p1, x_vals, c1_vals_3, label="3rd order", linewidth=3, color=:navy, linestyle=:dashdot)
    plot!(p1, x_vals, c1_vals_4, label="4th order", linewidth=3, color=:purple, linestyle=:dashdotdot)
    plot!(p1, x_vals, c1_vals_5, label="5th order", linewidth=3, color=:indigo, linestyle=:solid, alpha=0.8)
    plot!(p1, [xbar], [PerturbationLocalTaylor.eval_local(sol_5, xbar, ε)[1]], seriestype=:scatter, color=:black, markersize=5, label="x̄ = $(xbar)")
    xlabel!(p1, "Wealth Share (x)")
    ylabel!(p1, "c₁ Consumption Rate")
    title!(p1, "c₁: Taylor Series Convergence (Orders 0-5)")

    p2 = plot(x_vals, c2_vals_0, label="0th order", linewidth=2, color=:lightcoral, linestyle=:dot)
    plot!(p2, x_vals, c2_vals_1, label="1st order", linewidth=2, color=:red, linestyle=:dash)
    plot!(p2, x_vals, c2_vals_2, label="2nd order", linewidth=3, color=:darkred, linestyle=:solid)
    plot!(p2, x_vals, c2_vals_3, label="3rd order", linewidth=3, color=:maroon, linestyle=:dashdot)
    plot!(p2, x_vals, c2_vals_4, label="4th order", linewidth=3, color=:brown, linestyle=:dashdotdot)
    plot!(p2, x_vals, c2_vals_5, label="5th order", linewidth=3, color=:darkmagenta, linestyle=:solid, alpha=0.8)
    plot!(p2, [xbar], [PerturbationLocalTaylor.eval_local(sol_5, xbar, ε)[2]], seriestype=:scatter, color=:black, markersize=5, label="x̄ = $(xbar)")
    xlabel!(p2, "Wealth Share (x)")
    ylabel!(p2, "c₂ Consumption Rate")
    title!(p2, "c₂: Taylor Series Convergence (Orders 0-5)")

    combined_plot = plot(p1, p2, layout=(2,1), size=(800, 1000))
    plot!(combined_plot, plot_title="Taylor Series Perturbation Convergence\n(ψ=$(p_eff.ψ), σ=$(p_eff.σ), κ̂=$(p_eff.κhat); γ̂₁=$(p_eff.γhat1), γ̂₂=$(p_eff.γhat2))")
    return combined_plot
end

function taylor_value_and_derivatives(sol, x::Float64, ε::Float64)
    A1, A2 = sol.A1, sol.A2
    K, L, xbar, p = sol.K, sol.L, sol.xbar, sol.p
    δ = x - xbar
    c0val = PerturbationLocalTaylor.c0(p)

    c1 = c0val
    c2 = c0val
    c1δ = 0.0
    c2δ = 0.0
    c1δδ = 0.0
    c2δδ = 0.0

    for k in 1:K
        εk = ε^k
        for ℓ in 0:L
            coeff1 = A1[k, ℓ+1] * εk
            coeff2 = A2[k, ℓ+1] * εk
            # value terms
            c1 += coeff1 * (δ^ℓ)
            c2 += coeff2 * (δ^ℓ)
            # first derivative terms (ℓ>=1)
            if ℓ >= 1
                c1δ += coeff1 * ℓ * (δ^(ℓ-1))
                c2δ += coeff2 * ℓ * (δ^(ℓ-1))
            end
            # second derivative terms (ℓ>=2)
            if ℓ >= 2
                c1δδ += coeff1 * ℓ*(ℓ-1) * (δ^(ℓ-2))
                c2δδ += coeff2 * ℓ*(ℓ-1) * (δ^(ℓ-2))
            end
        end
    end

    # y and derivatives via δ
    x1 = x
    x2 = 1.0 - x
    y = x1*c1 + x2*c2
    yδ = (c1 - c2) + x1*c1δ + x2*c2δ
    yδδ = (2*c1δ - 2*c2δ) + x1*c1δδ + x2*c2δδ
    return (; c1, c2, y, yδ, yδδ, c1x=c1δ, c2x=c2δ)
end

function compute_numeric_residuals_at_x(sol, x::Float64, ε::Float64)
    p = sol.p
    vals = taylor_value_and_derivatives(sol, x, ε)
    y, yδ, yδδ = vals.y, vals.yδ, vals.yδδ
    γ1, γ2, κ = PerturbationLocalTaylor.γs_and_κ(p, ε)
    σ = p.σ; μ = p.μ; ψ = p.ψ

    # Fixed-point to close σx – α – σR loop
    function close_sigmax_float(y, yδ, x)
        σx = 0.0
        α1 = 1.0
        α2 = 1.0
        for _ in 1:8
            px_over_p = - yδ / max(y, 1e-12)
            σp = px_over_p * σx
            σR = σ + σp
            γ̄ = PerturbationLocalTaylor.γbar(x, γ1, γ2)
            π  = γ̄ * (σR^2)
            α1 = π / (γ1 * (σR^2))
            α2 = π / (γ2 * (σR^2))
            denom = 1 - x*(α1 - 1)*px_over_p
            σx = ( x*(α1 - 1) * σ ) / max(denom, 1e-12)
        end
        return σx, α1, α2
    end

    σx, α1, α2 = close_sigmax_float(y, yδ, x)
    px_over_p  = - yδ / max(y, 1e-12)
    yxx_term   = yδδ
    σp = px_over_p * σx
    σR = σ + σp
    γ̄ = PerturbationLocalTaylor.γbar(x, γ1, γ2)
    π  = γ̄ * (σR^2)
    μx = x * ( (π - σR^2)*(α1 - 1) + κ*(p.ω1 - x)/max(x, 1e-12) )
    σy = yδ * σx
    μy = yδ * μx + 0.5 * yxx_term * (σx^2)
    pxx_over_p = (px_over_p^2) - (yxx_term / max(y, 1e-12))
    μp = px_over_p * μx + 0.5 * pxx_over_p * (σx^2)

    rhs_c1 = ψ*p.ρ + (1 - ψ) * ( y + μ - μy + (σy^2) - σ*σy + π*(α1 - 1) - 0.5*γ1*(σR^2)*(α1^2) )
    rhs_c2 = ψ*p.ρ + (1 - ψ) * ( y + μ - μy + (σy^2) - σ*σy + π*(α2 - 1) - 0.5*γ2*(σR^2)*(α2^2) )
    # Relative residuals: 1 - rhs/cj with guard for tiny cj
    denom1 = abs(vals.c1) > 1e-12 ? vals.c1 : (vals.c1 >= 0 ? 1e-12 : -1e-12)
    denom2 = abs(vals.c2) > 1e-12 ? vals.c2 : (vals.c2 >= 0 ? 1e-12 : -1e-12)
    R1 = 1.0 - rhs_c1 / denom1
    R2 = 1.0 - rhs_c2 / denom2
    return R1, R2
end

function generate_residuals_by_order_plot(; p=PerturbationLocalTaylor.Params(), xbar=0.5, L=10, ε=1.0, x_vals=range(0.01, 0.95, length=50), orders=0:5)
    plots_R1 = []
    plots_R2 = []
    colors = [:lightgray, :gray, :blue, :darkblue, :purple, :indigo]
    labels = ["order $(k)" for k in orders]

    # Build solutions for each order
    sols = Dict{Int, Any}()
    for (i, k) in enumerate(orders)
        sols[k] = PerturbationLocalTaylor.solve_local(K=k, L=L, xbar=xbar, p=p)
    end

    # Collect residuals for each order
    series_R1 = Dict{Int, Vector{Float64}}()
    series_R2 = Dict{Int, Vector{Float64}}()
    for k in orders
        R1s = Float64[]
        R2s = Float64[]
        solk = sols[k]
        for x in x_vals
            R1, R2 = compute_numeric_residuals_at_x(solk, x, ε)
            push!(R1s, R1)
            push!(R2s, R2)
        end
        series_R1[k] = R1s
        series_R2[k] = R2s
    end

    # Helper: log10 |residual| with floor to avoid -Inf
    function log_abs(v::Vector{Float64})
        epsv = 1e-16
        return log10.(max.(abs.(v), epsv))
    end

    # Plot R1 (top) and R2 (bottom) on log10 scale
    pR1 = plot(x_vals, log_abs(series_R1[first(orders)]); label=labels[1], linewidth=2, color=colors[1])
    for (idx, k) in enumerate(orders)
        if idx == 1; continue; end
        plot!(pR1, x_vals, log_abs(series_R1[k]); label=labels[idx], linewidth=3, color=colors[min(idx, length(colors))])
    end
    xlabel!(pR1, "Wealth Share (x)")
    ylabel!(pR1, "log10 |R₁|")
    title!(pR1, "Residuals by order (R₁)")

    pR2 = plot(x_vals, log_abs(series_R2[first(orders)]); label=labels[1], linewidth=2, color=colors[1])
    for (idx, k) in enumerate(orders)
        if idx == 1; continue; end
        plot!(pR2, x_vals, log_abs(series_R2[k]); label=labels[idx], linewidth=3, color=colors[min(idx, length(colors))])
    end
    xlabel!(pR2, "Wealth Share (x)")
    ylabel!(pR2, "log10 |R₂|")
    title!(pR2, "Residuals by order (R₂)")

    combined = plot(pR1, pR2, layout=(2,1), size=(800, 1000))
    plot!(combined, plot_title="Residuals by perturbation order (ε=$(ε), L=$(L))\n(ψ=$(p.ψ), σ=$(p.σ), γ=$(p.γ), κ̂=$(p.κhat), γ̂₁=$(p.γhat1), γ̂₂=$(p.γhat2))")
    return combined
end

function compute_residuals_closed_form(xgrid, ε, pcf; use_composite::Bool, δ_scale::Float64=1.0, y_floor::Float64=1e-8)
    σ = pcf.σ; μ = pcf.μ; ψ = pcf.ψ; γ = pcf.γ
    # Helper to sample series
    sample(x) = use_composite ? PerturbationClosedForm.composite_series_at_x(x, ε, pcf; δ_scale=δ_scale, y_floor=y_floor) : PerturbationClosedForm.series_at_x(x, ε, pcf)
    R1raw = Float64[]; R2raw = Float64[]
    xmin = first(xgrid); xmax = last(xgrid)
    for x in xgrid
        # Robust finite differences for y derivatives
        h_center = min(1e-3, 0.25*min(x - xmin, xmax - x))
        sm = sample(x)
        yx = 0.0
        yxx = 0.0
        if h_center > 1e-8
            h = h_center
            xl = x - h
            xr = x + h
            sl = sample(xl)
            sr = sample(xr)
            yx  = (sr.y - sl.y) / (2*h)
            yxx = (sr.y - 2*sm.y + sl.y) / (h^2)
        else
            # One-sided near edges
            # Decide direction by proximity to boundary
            left_room = x - xmin
            right_room = xmax - x
            if right_room >= left_room && right_room > 1e-8
                # forward differences
                h = min(1e-3, 0.25*right_room)
                s1 = sample(x + h)
                s2 = sample(x + 2h)
                yx  = (-3*sm.y + 4*s1.y - s2.y) / (2*h)
                yxx = (sm.y - 2*s1.y + s2.y) / (h^2)
            elseif left_room > 1e-8
                # backward differences
                h = min(1e-3, 0.25*left_room)
                s1 = sample(x - h)
                s2 = sample(x - 2h)
                yx  = (3*sm.y - 4*s1.y + s2.y) / (2*h)
                yxx = (sm.y - 2*s1.y + s2.y) / (h^2)
            else
                # Degenerate fallback
                h = 1e-4
                s1 = sample(clamp(x + h, xmin, xmax))
                s2 = sample(clamp(x - h, xmin, xmax))
                yx  = (s1.y - s2.y) / (2*h)
                yxx = (s1.y - 2*sm.y + s2.y) / (h^2)
            end
        end
        # Recompute σx, α's, π via same fixed-point closure used in Taylor residuals
        function close_sigmax_float(y, yδ, x)
            σx = 0.0
            α1 = 1.0
            α2 = 1.0
            for _ in 1:8
                px_over_p = - yδ / max(y, 1e-12)
                σp = px_over_p * σx
                σR = σ + σp
                γ1_loc = γ * (1 + pcf.γhat1 * ε)
                γ2_loc = γ * (1 + pcf.γhat2 * ε)
                γ̄ = PerturbationLocalTaylor.γbar(x, γ1_loc, γ2_loc)
                π  = γ̄ * (σR^2)
                α1 = π / (γ1_loc * (σR^2))
                α2 = π / (γ2_loc * (σR^2))
                denom = 1 - x*(α1 - 1)*px_over_p
                σx = ( x*(α1 - 1) * σ ) / max(denom, 1e-12)
            end
            return σx, α1, α2
        end

        σx, α1, α2 = close_sigmax_float(sm.y, yx, x)
        σy = yx * σx
        # x-drift and y-drift
        γ1 = γ * (1 + pcf.γhat1 * ε)
        γ2 = γ * (1 + pcf.γhat2 * ε)
        px_over_p = - yx / max(sm.y, 1e-12)
        σp = px_over_p * σx
        σR = σ + σp
        γ̄ = PerturbationLocalTaylor.γbar(x, γ1, γ2)
        π  = γ̄ * (σR^2)
        μx = x * ( (π - σR^2)*(α1 - 1) + (pcf.κhat)*(pcf.ω1 - x)/max(x, 1e-12) )
        μy = yx * μx + 0.5 * yxx * (σx^2)
        rhs_c1 = ψ*pcf.ρ + (1 - ψ) * ( sm.y + μ - μy + (σy^2) - σ*σy + π*(α1 - 1) - 0.5*γ1*(σR^2)*(α1^2) )
        rhs_c2 = ψ*pcf.ρ + (1 - ψ) * ( sm.y + μ - μy + (σy^2) - σ*σy + π*(α2 - 1) - 0.5*γ2*(σR^2)*(α2^2) )
        # Relative residuals with guard for small denominators
        denom1 = abs(sm.c1) > 1e-12 ? sm.c1 : (sm.c1 >= 0 ? 1e-12 : -1e-12)
        denom2 = abs(sm.c2) > 1e-12 ? sm.c2 : (sm.c2 >= 0 ? 1e-12 : -1e-12)
        push!(R1raw, 1.0 - rhs_c1 / denom1)
        push!(R2raw, 1.0 - rhs_c2 / denom2)
    end
    return (; R1=R1raw, R2=R2raw)
end

function generate_residuals_composite_plot(; p=PerturbationLocalTaylor.Params(), ε=1.0, x_vals=range(0.01, 0.95, length=50), overlay_outer::Bool=true, δ_scale::Float64=1.0, y_floor::Float64=1e-8)
    p_cf = PerturbationClosedForm.Params(ρ=p.ρ, ψ=p.ψ, μ=p.μ, σ=p.σ, γ=p.γ, γhat1=p.γhat1, γhat2=p.γhat2, κhat=p.κhat, ω1=p.ω1)
    res_outer = compute_residuals_closed_form(collect(x_vals), ε, p_cf; use_composite=false, δ_scale=δ_scale, y_floor=y_floor)
    res_comp  = compute_residuals_closed_form(collect(x_vals), ε, p_cf; use_composite=true,  δ_scale=δ_scale, y_floor=y_floor)

    log_abs(v) = log10.(max.(abs.(v), 1e-16))

    pR1 = plot(x_vals, log_abs(res_comp.R1), label="|R₁| (Composite)", linewidth=3, color=:blue, linestyle=:dash)
    if overlay_outer
        plot!(pR1, x_vals, log_abs(res_outer.R1), label="|R₁| (Outer)", linewidth=3, color=:darkblue, linestyle=:solid)
    end
    xlabel!(pR1, "Wealth Share (x)")
    ylabel!(pR1, "log10 |R₁|")
    title!(pR1, "Residuals — Composite" * (overlay_outer ? " vs Outer" : "") * " (γ̂₁=$(p.γhat1), γ̂₂=$(p.γhat2))")

    pR2 = plot(x_vals, log_abs(res_comp.R2), label="|R₂| (Composite)", linewidth=3, color=:red, linestyle=:dash)
    if overlay_outer
        plot!(pR2, x_vals, log_abs(res_outer.R2), label="|R₂| (Outer)", linewidth=3, color=:darkred, linestyle=:solid)
    end
    xlabel!(pR2, "Wealth Share (x)")
    ylabel!(pR2, "log10 |R₂|")
    title!(pR2, "Residuals — Composite" * (overlay_outer ? " vs Outer" : "") * " (γ̂₁=$(p.γhat1), γ̂₂=$(p.γhat2))")

    combined = plot(pR1, pR2, layout=(2,1), size=(800, 1000))
    plot!(combined, plot_title="Residuals (Closed-form/Composite)\n(ψ=$(p_cf.ψ), σ=$(p_cf.σ), κ̂=$(p_cf.κhat))")
    return combined
end

function generate_composite_plot(; p=PerturbationLocalTaylor.Params(), ε=1.0, x_vals=range(0.01, 0.95, length=50), δ_scale::Float64=1.0, y_floor::Float64=1e-8)
    p_cf = PerturbationClosedForm.Params(
        ρ=p.ρ, ψ=p.ψ, μ=p.μ, σ=p.σ, γ=p.γ,
        γhat1=p.γhat1, γhat2=p.γhat2, κhat=p.κhat, ω1=p.ω1
    )

    c1_outer, c2_outer = Float64[], Float64[]
    c1_comp,  c2_comp  = Float64[], Float64[]
    for x in x_vals
        o = PerturbationClosedForm.series_at_x(x, ε, p_cf)
        c = PerturbationClosedForm.composite_series_at_x(x, ε, p_cf; δ_scale=δ_scale, y_floor=y_floor)
        push!(c1_outer, o.c1); push!(c2_outer, o.c2)
        push!(c1_comp,  c.c1); push!(c2_comp,  c.c2)
    end

    q1 = plot(x_vals, c1_outer, label="c₁ Outer (2nd)", linewidth=3, color=:darkblue, linestyle=:solid)
    plot!(q1, x_vals, c1_comp,  label="c₁ Composite", linewidth=3, color=:blue, linestyle=:dash)
    xlabel!(q1, "Wealth Share (x)")
    ylabel!(q1, "c₁ Consumption Rate")
    title!(q1, "c₁: Composite vs Outer (γ̂₁=$(p.γhat1), γ̂₂=$(p.γhat2); δ_scale=$(δ_scale))")

    q2 = plot(x_vals, c2_outer, label="c₂ Outer (2nd)", linewidth=3, color=:darkred, linestyle=:solid)
    plot!(q2, x_vals, c2_comp,  label="c₂ Composite", linewidth=3, color=:red, linestyle=:dash)
    xlabel!(q2, "Wealth Share (x)")
    ylabel!(q2, "c₂ Consumption Rate")
    title!(q2, "c₂: Composite vs Outer (γ̂₁=$(p.γhat1), γ̂₂=$(p.γhat2); δ_scale=$(δ_scale))")

    combined = plot(q1, q2, layout=(2,1), size=(800, 1000))
    plot!(combined, plot_title="Composite vs Outer (Closed-form)\n(ψ=$(p_cf.ψ), σ=$(p_cf.σ), κ̂=$(p_cf.κhat); δ_scale=$(δ_scale))")
    return combined
end

function compute_sigmax_from_taylor(sol, x::Float64, ε::Float64)
    p = sol.p
    vals = taylor_value_and_derivatives(sol, x, ε)
    y, yδ = vals.y, vals.yδ
    γ1, γ2, _ = PerturbationLocalTaylor.γs_and_κ(p, ε)
    σ = p.σ
    function close_sigmax_float(y, yδ, x)
        σx = 0.0
        α1 = 1.0
        for _ in 1:8
            px_over_p = - yδ / max(y, 1e-12)
            σp = px_over_p * σx
            σR = σ + σp
            γ̄ = PerturbationLocalTaylor.γbar(x, γ1, γ2)
            π  = γ̄ * (σR^2)
            α1 = π / (γ1 * (σR^2))
            denom = 1 - x*(α1 - 1)*px_over_p
            σx = ( x*(α1 - 1) * σ ) / max(denom, 1e-12)
        end
        return σx
    end
    return close_sigmax_float(y, yδ, x)
end

function generate_sigmax_orders_plot(; p=PerturbationLocalTaylor.Params(), xbar=0.5, L=10, ε=1.0,
                                     orders=0:5,
                                     gammahat1::Union{Nothing,Float64}=nothing,
                                     gammahat2::Union{Nothing,Float64}=nothing,
                                     kappahat::Union{Nothing,Float64}=nothing)
    p_eff = if gammahat1 === nothing && gammahat2 === nothing && kappahat === nothing
        p
    else
        PerturbationLocalTaylor.Params(
            ρ=p.ρ, ψ=p.ψ, μ=p.μ, σ=p.σ, γ=p.γ,
            γhat1 = (gammahat1 === nothing ? p.γhat1 : gammahat1),
            γhat2 = (gammahat2 === nothing ? p.γhat2 : gammahat2),
            κhat  = (kappahat === nothing ? p.κhat : kappahat),
            ω1=p.ω1
        )
    end

    x_vals = range(0.01, 0.95, length=50)
    sols = Dict{Int, Any}()
    for k in orders
        sols[k] = PerturbationLocalTaylor.solve_local(K=k, L=L, xbar=xbar, p=p_eff)
    end

    colors = [:lightgray, :gray, :blue, :darkblue, :purple, :indigo]
    labels = ["order $(k)" for k in orders]

    # Build plot
    first_k = first(orders)
    σx_first = [compute_sigmax_from_taylor(sols[first_k], x, ε) for x in x_vals]
    psx = plot(x_vals, σx_first; label=labels[1], linewidth=2, color=colors[1])
    for (idx, k) in enumerate(orders)
        if idx == 1; continue; end
        σx_vals = [compute_sigmax_from_taylor(sols[k], x, ε) for x in x_vals]
        plot!(psx, x_vals, σx_vals; label=labels[idx], linewidth=3, color=colors[min(idx, length(colors))])
    end
    xlabel!(psx, "Wealth Share (x)")
    ylabel!(psx, "σₓ")
    title!(psx, "σₓ: Taylor Series Convergence (Orders $(first(orders))–$(last(orders)))")
    return psx
end

function generate_cx_orders_plot(; p=PerturbationLocalTaylor.Params(), xbar=0.5, L=10, ε=1.0,
                                 orders=0:5,
                                 gammahat1::Union{Nothing,Float64}=nothing,
                                 gammahat2::Union{Nothing,Float64}=nothing,
                                 kappahat::Union{Nothing,Float64}=nothing)
    p_eff = if gammahat1 === nothing && gammahat2 === nothing && kappahat === nothing
        p
    else
        PerturbationLocalTaylor.Params(
            ρ=p.ρ, ψ=p.ψ, μ=p.μ, σ=p.σ, γ=p.γ,
            γhat1 = (gammahat1 === nothing ? p.γhat1 : gammahat1),
            γhat2 = (gammahat2 === nothing ? p.γhat2 : gammahat2),
            κhat  = (kappahat === nothing ? p.κhat : kappahat),
            ω1=p.ω1
        )
    end

    x_vals = range(0.01, 0.95, length=50)
    sols = Dict{Int, Any}()
    for k in orders
        sols[k] = PerturbationLocalTaylor.solve_local(K=k, L=L, xbar=xbar, p=p_eff)
    end

    colors = [:lightgray, :gray, :blue, :darkblue, :purple, :indigo]
    labels = ["order $(k)" for k in orders]

    # Build c1x plot
    first_k = first(orders)
    c1x_first = [taylor_value_and_derivatives(sols[first_k], x, ε).c1x for x in x_vals]
    pc1x = plot(x_vals, c1x_first; label=labels[1], linewidth=2, color=colors[1])
    for (idx, k) in enumerate(orders)
        if idx == 1; continue; end
        c1x_vals = [taylor_value_and_derivatives(sols[k], x, ε).c1x for x in x_vals]
        plot!(pc1x, x_vals, c1x_vals; label=labels[idx], linewidth=3, color=colors[min(idx, length(colors))])
    end
    xlabel!(pc1x, "Wealth Share (x)")
    ylabel!(pc1x, "c₁ₓ")
    title!(pc1x, "c₁ₓ: Taylor Series Convergence (Orders $(first(orders))–$(last(orders)))")

    # Build c2x plot
    c2x_first = [taylor_value_and_derivatives(sols[first_k], x, ε).c2x for x in x_vals]
    pc2x = plot(x_vals, c2x_first; label=labels[1], linewidth=2, color=colors[1])
    for (idx, k) in enumerate(orders)
        if idx == 1; continue; end
        c2x_vals = [taylor_value_and_derivatives(sols[k], x, ε).c2x for x in x_vals]
        plot!(pc2x, x_vals, c2x_vals; label=labels[idx], linewidth=3, color=colors[min(idx, length(colors))])
    end
    xlabel!(pc2x, "Wealth Share (x)")
    ylabel!(pc2x, "c₂ₓ")
    title!(pc2x, "c₂ₓ: Taylor Series Convergence (Orders $(first(orders))–$(last(orders)))")

    combined = plot(pc1x, pc2x, layout=(2,1), size=(800, 1000))
    plot!(combined, plot_title="cⱼₓ across Taylor orders (ψ=$(p_eff.ψ), σ=$(p_eff.σ), κ̂=$(p_eff.κhat))")
    return combined
end

function generate_weighted_cx_orders_plot(; p=PerturbationLocalTaylor.Params(), xbar=0.5, L=10, ε=1.0,
                                          orders=0:5,
                                          gammahat1::Union{Nothing,Float64}=nothing,
                                          gammahat2::Union{Nothing,Float64}=nothing,
                                          kappahat::Union{Nothing,Float64}=nothing)
    p_eff = if gammahat1 === nothing && gammahat2 === nothing && kappahat === nothing
        p
    else
        PerturbationLocalTaylor.Params(
            ρ=p.ρ, ψ=p.ψ, μ=p.μ, σ=p.σ, γ=p.γ,
            γhat1 = (gammahat1 === nothing ? p.γhat1 : gammahat1),
            γhat2 = (gammahat2 === nothing ? p.γhat2 : gammahat2),
            κhat  = (kappahat === nothing ? p.κhat : kappahat),
            ω1=p.ω1
        )
    end

    x_vals = range(0.01, 0.95, length=50)
    sols = Dict{Int, Any}()
    for k in orders
        sols[k] = PerturbationLocalTaylor.solve_local(K=k, L=L, xbar=xbar, p=p_eff)
    end

    colors = [:lightgray, :gray, :blue, :darkblue, :purple, :indigo]
    labels = ["order $(k)" for k in orders]

    # x*c1x
    first_k = first(orders)
    x_c1x_first = [x * taylor_value_and_derivatives(sols[first_k], x, ε).c1x for x in x_vals]
    px1 = plot(x_vals, x_c1x_first; label=labels[1], linewidth=2, color=colors[1])
    for (idx, k) in enumerate(orders)
        if idx == 1; continue; end
        x_c1x_vals = [x * taylor_value_and_derivatives(sols[k], x, ε).c1x for x in x_vals]
        plot!(px1, x_vals, x_c1x_vals; label=labels[idx], linewidth=3, color=colors[min(idx, length(colors))])
    end
    xlabel!(px1, "Wealth Share (x)")
    ylabel!(px1, "x · c₁ₓ")
    title!(px1, "x · c₁ₓ across Taylor orders (Orders $(first(orders))–$(last(orders)))")

    # x*c2x
    x_c2x_first = [x * taylor_value_and_derivatives(sols[first_k], x, ε).c2x for x in x_vals]
    px2 = plot(x_vals, x_c2x_first; label=labels[1], linewidth=2, color=colors[1])
    for (idx, k) in enumerate(orders)
        if idx == 1; continue; end
        x_c2x_vals = [x * taylor_value_and_derivatives(sols[k], x, ε).c2x for x in x_vals]
        plot!(px2, x_vals, x_c2x_vals; label=labels[idx], linewidth=3, color=colors[min(idx, length(colors))])
    end
    xlabel!(px2, "Wealth Share (x)")
    ylabel!(px2, "x · c₂ₓ")
    title!(px2, "x · c₂ₓ across Taylor orders (Orders $(first(orders))–$(last(orders)))")

    combined = plot(px1, px2, layout=(2,1), size=(800, 1000))
    plot!(combined, plot_title="Weighted derivatives x·c₁ₓ and x·c₂ₓ (ψ=$(p_eff.ψ), σ=$(p_eff.σ), κ̂=$(p_eff.κhat))")
    return combined
end

function generate_weighted_c2x_power_plot(; p=PerturbationLocalTaylor.Params(), xbar=0.5, L=10, ε=1.0,
                                          orders=0:5, alpha::Float64=1.0,
                                          gammahat1::Union{Nothing,Float64}=nothing,
                                          gammahat2::Union{Nothing,Float64}=nothing,
                                          kappahat::Union{Nothing,Float64}=nothing)
    p_eff = if gammahat1 === nothing && gammahat2 === nothing && kappahat === nothing
        p
    else
        PerturbationLocalTaylor.Params(
            ρ=p.ρ, ψ=p.ψ, μ=p.μ, σ=p.σ, γ=p.γ,
            γhat1 = (gammahat1 === nothing ? p.γhat1 : gammahat1),
            γhat2 = (gammahat2 === nothing ? p.γhat2 : gammahat2),
            κhat  = (kappahat === nothing ? p.κhat : kappahat),
            ω1=p.ω1
        )
    end

    x_vals = range(0.01, 0.95, length=50)
    sols = Dict{Int, Any}()
    for k in orders
        sols[k] = PerturbationLocalTaylor.solve_local(K=k, L=L, xbar=xbar, p=p_eff)
    end

    colors = [:lightgray, :gray, :blue, :darkblue, :purple, :indigo]
    labels = ["order $(k)" for k in orders]

    first_k = first(orders)
    w_c2x_first = [(x^alpha) * taylor_value_and_derivatives(sols[first_k], x, ε).c2x for x in x_vals]
    pwr = plot(x_vals, w_c2x_first; label=labels[1], linewidth=2, color=colors[1])
    for (idx, k) in enumerate(orders)
        if idx == 1; continue; end
        w_c2x_vals = [(x^alpha) * taylor_value_and_derivatives(sols[k], x, ε).c2x for x in x_vals]
        plot!(pwr, x_vals, w_c2x_vals; label=labels[idx], linewidth=3, color=colors[min(idx, length(colors))])
    end
    xlabel!(pwr, "Wealth Share (x)")
    ylabel!(pwr, "x^α · c₂ₓ")
    title!(pwr, "x^α · c₂ₓ across Taylor orders (α=$(alpha))")
    return pwr
end

function generate_yx_orders_plot(; p=PerturbationLocalTaylor.Params(), xbar=0.5, L=10, ε=1.0,
                                 orders=0:5,
                                 gammahat1::Union{Nothing,Float64}=nothing,
                                 gammahat2::Union{Nothing,Float64}=nothing,
                                 kappahat::Union{Nothing,Float64}=nothing)
    p_eff = if gammahat1 === nothing && gammahat2 === nothing && kappahat === nothing
        p
    else
        PerturbationLocalTaylor.Params(
            ρ=p.ρ, ψ=p.ψ, μ=p.μ, σ=p.σ, γ=p.γ,
            γhat1 = (gammahat1 === nothing ? p.γhat1 : gammahat1),
            γhat2 = (gammahat2 === nothing ? p.γhat2 : gammahat2),
            κhat  = (kappahat === nothing ? p.κhat : kappahat),
            ω1=p.ω1
        )
    end

    x_vals = range(0.01, 0.95, length=50)
    sols = Dict{Int, Any}()
    for k in orders
        sols[k] = PerturbationLocalTaylor.solve_local(K=k, L=L, xbar=xbar, p=p_eff)
    end

    colors = [:lightgray, :gray, :blue, :darkblue, :purple, :indigo]
    labels = ["order $(k)" for k in orders]

    first_k = first(orders)
    yx_first = [taylor_value_and_derivatives(sols[first_k], x, ε).yδ for x in x_vals]
    pyx = plot(x_vals, yx_first; label=labels[1], linewidth=2, color=colors[1])
    for (idx, k) in enumerate(orders)
        if idx == 1; continue; end
        yx_vals = [taylor_value_and_derivatives(sols[k], x, ε).yδ for x in x_vals]
        plot!(pyx, x_vals, yx_vals; label=labels[idx], linewidth=3, color=colors[min(idx, length(colors))])
    end
    xlabel!(pyx, "Wealth Share (x)")
    ylabel!(pyx, "yₓ")
    title!(pyx, "yₓ: Taylor Series Convergence (Orders $(first(orders))–$(last(orders)))")
    return pyx
end

function generate_B_orders_plot(; p=PerturbationLocalTaylor.Params(), xbar=0.5, L=10, ε=1.0,
                                orders=0:5,
                                gammahat1::Union{Nothing,Float64}=nothing,
                                gammahat2::Union{Nothing,Float64}=nothing,
                                kappahat::Union{Nothing,Float64}=nothing)
    p_eff = if gammahat1 === nothing && gammahat2 === nothing && kappahat === nothing
        p
    else
        PerturbationLocalTaylor.Params(
            ρ=p.ρ, ψ=p.ψ, μ=p.μ, σ=p.σ, γ=p.γ,
            γhat1 = (gammahat1 === nothing ? p.γhat1 : gammahat1),
            γhat2 = (gammahat2 === nothing ? p.γhat2 : gammahat2),
            κhat  = (kappahat === nothing ? p.κhat : kappahat),
            ω1=p.ω1
        )
    end

    x_vals = range(0.01, 0.95, length=50)
    sols = Dict{Int, Any}()
    for k in orders
        sols[k] = PerturbationLocalTaylor.solve_local(K=k, L=L, xbar=xbar, p=p_eff)
    end

    colors = [:lightgray, :gray, :blue, :darkblue, :purple, :indigo]
    labels = ["order $(k)" for k in orders]

    first_k = first(orders)
    B_first = [ x * (let vals = taylor_value_and_derivatives(sols[first_k], x, ε); vals.yδ / max(vals.y, 1e-12) end) for x in x_vals ]
    pB = plot(x_vals, B_first; label=labels[1], linewidth=2, color=colors[1])
    for (idx, k) in enumerate(orders)
        if idx == 1; continue; end
        B_vals = [ x * (let vals = taylor_value_and_derivatives(sols[k], x, ε); vals.yδ / max(vals.y, 1e-12) end) for x in x_vals ]
        plot!(pB, x_vals, B_vals; label=labels[idx], linewidth=3, color=colors[min(idx, length(colors))])
    end
    xlabel!(pB, "Wealth Share (x)")
    ylabel!(pB, "B(x) = x · yₓ / y")
    title!(pB, "B(x): Taylor Series Convergence (Orders $(first(orders))–$(last(orders)))")
    return pB
end

function generate_B1B2_orders_plot(; p=PerturbationLocalTaylor.Params(), xbar=0.5, L=10, ε=1.0,
                                   orders=0:5,
                                   gammahat1::Union{Nothing,Float64}=nothing,
                                   gammahat2::Union{Nothing,Float64}=nothing,
                                   kappahat::Union{Nothing,Float64}=nothing)
    p_eff = if gammahat1 === nothing && gammahat2 === nothing && kappahat === nothing
        p
    else
        PerturbationLocalTaylor.Params(
            ρ=p.ρ, ψ=p.ψ, μ=p.μ, σ=p.σ, γ=p.γ,
            γhat1 = (gammahat1 === nothing ? p.γhat1 : gammahat1),
            γhat2 = (gammahat2 === nothing ? p.γhat2 : gammahat2),
            κhat  = (kappahat === nothing ? p.κhat : kappahat),
            ω1=p.ω1
        )
    end

    x_vals = range(0.01, 0.95, length=50)
    sols = Dict{Int, Any}()
    for k in orders
        sols[k] = PerturbationLocalTaylor.solve_local(K=k, L=L, xbar=xbar, p=p_eff)
    end

    colors = [:lightgray, :gray, :blue, :darkblue, :purple, :indigo]
    labels = ["order $(k)" for k in orders]

    # B1(x) = x * (c1 - c2 + x*c1') / y
    first_k = first(orders)
    B1_first = [ x * (let vals = taylor_value_and_derivatives(sols[first_k], x, ε); (vals.c1 - vals.c2 + x*vals.c1x) / max(vals.y, 1e-12) end) for x in x_vals ]
    pB1 = plot(x_vals, B1_first; label=labels[1], linewidth=2, color=colors[1])
    for (idx, k) in enumerate(orders)
        if idx == 1; continue; end
        B1_vals = [ x * (let vals = taylor_value_and_derivatives(sols[k], x, ε); (vals.c1 - vals.c2 + x*vals.c1x) / max(vals.y, 1e-12) end) for x in x_vals ]
        plot!(pB1, x_vals, B1_vals; label=labels[idx], linewidth=3, color=colors[min(idx, length(colors))])
    end
    xlabel!(pB1, "Wealth Share (x)")
    ylabel!(pB1, "B₁(x)")
    title!(pB1, "B₁(x) = x·(c₁−c₂ + x c₁ₓ)/y")

    # B2(x) = x*(1-x)*c2'/y
    B2_first = [ x*(1-x) * (let vals = taylor_value_and_derivatives(sols[first_k], x, ε); vals.c2x / max(vals.y, 1e-12) end) for x in x_vals ]
    pB2 = plot(x_vals, B2_first; label=labels[1], linewidth=2, color=colors[1])
    for (idx, k) in enumerate(orders)
        if idx == 1; continue; end
        B2_vals = [ x*(1-x) * (let vals = taylor_value_and_derivatives(sols[k], x, ε); vals.c2x / max(vals.y, 1e-12) end) for x in x_vals ]
        plot!(pB2, x_vals, B2_vals; label=labels[idx], linewidth=3, color=colors[min(idx, length(colors))])
    end
    xlabel!(pB2, "Wealth Share (x)")
    ylabel!(pB2, "B₂(x)")
    title!(pB2, "B₂(x) = x(1−x) c₂ₓ/y")

    combined = plot(pB1, pB2, layout=(2,1), size=(800, 1000))
    plot!(combined, plot_title="B₁(x) and B₂(x) across Taylor orders (ψ=$(p_eff.ψ), σ=$(p_eff.σ), κ̂=$(p_eff.κhat))")
    return combined
end

end # module PerturbationPlots

