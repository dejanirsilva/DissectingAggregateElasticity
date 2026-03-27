# Closed-form (analytic) ε-perturbation based on the user's formulas
# Implements Eqs. (17)–(36) from the note (zeroth, first, and second order)
# Outer solution valid for x ∈ (0,1), away from boundaries.

# module PerturbationClosedForm

# export Params, series_at_x, grid_series, to_dataframe
module PerturbationClosedForm

    export Params, series_at_x, grid_series, to_dataframe

    using LinearAlgebra

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

    # Convenience helpers
    γbar(x, γ1, γ2) = 1.0 / (x/γ1 + (1-x)/γ2)

    # Zeroth order (Eq. 17; Eq. 18)
    function c0(p::Params)
        return p.ρ + (1/p.ψ - 1.0) * (p.μ - 0.5*p.γ*p.σ^2)
    end

    # -----------------------
    # First-order terms (Eqs. 19–24)
    # -----------------------
    function order1(x, p::Params)
        x1 = x; x2 = 1.0 - x
        σ, γ = p.σ, p.γ
        ystar = c0(p)

        # π1 (Eq. 19)
        π1 = γ*σ^2 * (x1*p.γhat1 + x2*p.γhat2)

        # αj,1 (Eq. 20)
        α1_1 = π1/(γ*σ^2) - p.γhat1
        α2_1 = π1/(γ*σ^2) - p.γhat2

        # y1 (Eq. 21)
        y1 = (1.0 - 1.0/p.ψ) * (γ*σ^2/2.0) * (x1*p.γhat1 + x2*p.γhat2)

        # c_j,1 (Eq. 22)
        common = (1.0 - p.ψ) * (γ*σ^2/2.0)
        bracket = (1.0 - 1.0/p.ψ)*(x1*p.γhat1 + x2*p.γhat2)
        c1_1 = common * (bracket - p.γhat1)
        c2_1 = common * (bracket - p.γhat2)

        # σx,1 (Eq. 23)
        σx_1 = x*(1.0 - x)*(p.γhat2 - p.γhat1) * σ

        # μx,1 (Eq. 24)
        μx_1 = x*(1.0 - x)*(γ - 1.0)*σ^2*(p.γhat2 - p.γhat1) + p.κhat*(p.ω1 - x)

        return (π1=π1, α1_1=α1_1, α2_1=α2_1, y1=y1, c1_1=c1_1, c2_1=c2_1, σx_1=σx_1, μx_1=μx_1, y0=ystar)
    end

    # -----------------------
    # Second-order terms (Eqs. 25–36)
    # -----------------------
    function order2(x, p::Params, o1)
        x1 = x; x2 = 1.0 - x
        σ, γ = p.σ, p.γ
        y0 = o1.y0

        # σy,2 (Eq. 25)
        Δγhat = (p.γhat2 - p.γhat1)
        σy_2 = -(1.0 - 1.0/p.ψ) * (γ*σ^2) / (2.0*y0) * (Δγhat^2) * σ

        # μy,2 (Eq. 26)
        μy_2 = -(1.0 - 1.0/p.ψ) * (γ*σ^2) / (2.0*y0) * (Δγhat) * ( x*(1.0 - x)*(γ - 1.0)*σ^2*(Δγhat) + p.κhat*(p.ω1 - x) )

        # μc_j,2 and σc_j,2 (Eq. 27) — not used directly below but kept for completeness
        μc1_2 = -(p.ψ - 1.0) * μy_2
        μc2_2 = μc1_2
        σc1_2 = -(p.ψ - 1.0) * σy_2
        σc2_2 = σc1_2

        # ς_j,2 (Eq. 28)
        ϛ1_2 = -(1.0 - 1.0/γ) * (σy_2/σ)
        ϛ2_2 = ϛ1_2

        # γm,1 and γm,2 (Eq. 29)
        γm_1 = γ * (x1*p.γhat1 + x2*p.γhat2)
        γm_2 = -γ * ( x1*p.γhat1^2 + x2*p.γhat2^2 - (x1*p.γhat1 + x2*p.γhat2)^2 )

        # π2 (Eq. 30)
        π2 = γ*σ^2 * ( (γm_2/γ) - (1.0 + 1.0/γ)*(σy_2/σ) )

        # αj,2 (Eq. 31)
        π0 = γ*σ^2
        α1_2 = (γm_2/γ) + p.γhat1^2 - p.γhat1 * (o1.π1/π0)
        α2_2 = (γm_2/γ) + p.γhat2^2 - p.γhat2 * (o1.π1/π0)

        # ξ_j,2 (Eq. 32)
        ξ1_2 = -(p.ψ - 1.0) * ( μy_2 + (1.0 - γ)*σy_2*σ )
        ξ2_2 = ξ1_2

        # y2 (Eq. 33)
        sum1 = x1*(p.γhat1*o1.α1_1) + x2*(p.γhat2*o1.α2_1)
        sum2 = x1*(o1.α1_1^2) + x2*(o1.α2_1^2)
        y2 = (1.0 - 1.0/p.ψ) * γ*σ^2 * ( sum1 + 0.5*sum2 )

        # c_j,2 (Eq. 34)
        c1_2 = (1.0 - p.ψ) * ( y2 + o1.π1*o1.α1_1 - γ*σ^2*p.γhat1*o1.α1_1 - 0.5*γ*σ^2*o1.α1_1^2 )
        c2_2 = (1.0 - p.ψ) * ( y2 + o1.π1*o1.α2_1 - γ*σ^2*p.γhat2*o1.α2_1 - 0.5*γ*σ^2*o1.α2_1^2 )

        # σx,2 and μx,2 (Eqs. 35–36)
        σx_2 = x * α1_2 * σ
        μx_2 = x * ( o1.π1*o1.α1_1 + (γ - 1.0)*σ^2*α1_2 )

        return (σy_2=σy_2, μy_2=μy_2, μc1_2=μc1_2, μc2_2=μc2_2, σc1_2=σc1_2, σc2_2=σc2_2,
                ϛ1_2=ϛ1_2, ϛ2_2=ϛ2_2, γm_1=γm_1, γm_2=γm_2, π2=π2,
                α1_2=α1_2, α2_2=α2_2, ξ1_2=ξ1_2, ξ2_2=ξ2_2, y2=y2, c1_2=c1_2, c2_2=c2_2,
                σx_2=σx_2, μx_2=μx_2)
    end

    # -----------------------
    # Assemble series at a point x for a given ε (through O(ε^2))
    # c_j(x;ε) = c0 + c_{j,1}(x) ε + c_{j,2}(x) ε^2
    # y(x;ε)   = y0 + y1(x) ε + y2(x) ε^2
    # Also returns α_j, π, σx, μx up to matching order.
    # -----------------------
    function series_at_x(x::Float64, ε::Float64, p::Params)
        # 0th order
        y0 = c0(p)

        # 1st order bundle
        o1 = order1(x, p)

        # 2nd order bundle
        o2 = order2(x, p, o1)

        # Series evaluation
        c1 = y0 + o1.c1_1*ε + o2.c1_2*ε^2
        c2 = y0 + o1.c2_1*ε + o2.c2_2*ε^2
        y  = y0 + o1.y1*ε  + o2.y2*ε^2

        # Risk premium and α's up to O(ε^2)
        π0 = p.γ*p.σ^2
        π  = π0 + o1.π1*ε + o2.π2*ε^2
        α1 = 1.0 + o1.α1_1*ε + o2.α1_2*ε^2
        α2 = 1.0 + o1.α2_1*ε + o2.α2_2*ε^2

        # x SDE terms
        σx = o1.σx_1*ε + o2.σx_2*ε^2
        μx = o1.μx_1*ε + o2.μx_2*ε^2

        return (; c1, c2, y, π, α1, α2, σx, μx,
                y0=y0, y1=o1.y1, y2=o2.y2,
                c1_1=o1.c1_1, c1_2=o2.c1_2, c2_1=o1.c2_1, c2_2=o2.c2_2,
                α1_1=o1.α1_1, α1_2=o2.α1_2, α2_1=o1.α2_1, α2_2=o2.α2_2,
                π1=o1.π1, π2=o2.π2,
                σx_1=o1.σx_1, σx_2=o2.σx_2, μx_1=o1.μx_1, μx_2=o2.μx_2)
    end

    # -----------------------
    # Evaluate on a grid of x
    # -----------------------
    function grid_series(xgrid::AbstractVector{<:Real}, ε::Float64, p::Params)
        out = Vector{NamedTuple}(undef, length(xgrid))
        for (i,x) in enumerate(xgrid)
            out[i] = series_at_x(Float64(x), ε, p)
        end
        return out
    end

    # -----------------------
    # Inner approximation near x ≈ 0 using boundary-layer formulas
    # and symmetric inner approximation near x ≈ 1 by swapping types.
    # Returns (c1,c2,y) including up to the terms implied by the inner
    # expansion through O(ε^2) where y2 and c2,2 scale with x/ε.
    # -----------------------
    function inner_series_left(x::Float64, ε::Float64, p::Params)
        y0 = c0(p)
        σ, γ = p.σ, p.γ
        # First-order constants near x → 0
        y1_hat = (1.0 - 1.0/p.ψ) * (γ*σ^2/2.0) * p.γhat2
        c2_1_hat = y1_hat
        c1_1_hat = y1_hat - (p.ψ - 1.0) * (γ*σ^2/2.0) * (p.γhat2 - p.γhat1)
        # Second-order inner growth terms (proportional to x/ε)
        Δc1_1 = c1_1_hat - c2_1_hat
        y2_hat = (1.0/p.ψ) * Δc1_1 * (x/ε)
        A = (1.0/p.ψ - 1.0) * Δc1_1
        c2_2_hat = A * (x/ε)
        # Assemble series
        y  = y0 + ε*y1_hat + ε^2*y2_hat
        c2 = y0 + ε*c2_1_hat + ε^2*c2_2_hat
        # Recover c1 from identity y = x c1 + (1-x) c2
        # Guard against x=0 with a small epsilon in denominator
        denom = max(x, 1e-8)
        c1 = (y - (1.0 - x)*c2) / denom
        # Inner controls near left boundary (first nontrivial orders)
        Δ = (p.γhat2 - p.γhat1)
        π0 = γ*σ^2
        π1 = γ*σ^2 * p.γhat2
        π2 = -γ*σ^2 * Δ * (x/ε)
        π  = π0 + ε*π1 + ε^2*π2
        α1 = 1.0 + ε*Δ
        α2 = 1.0 + ε*0.0
        σx = x * Δ * σ
        μx = x * ( (γ - 1.0)*σ^2*Δ - p.κhat )
        return (; c1, c2, y, π, α1, α2, σx, μx)
    end

    function inner_series_right(x::Float64, ε::Float64, p::Params)
        # Symmetry: define x̃ = 1 - x and swap types 1 ↔ 2
        x̃ = 1.0 - x
        p_swapped = Params(ρ=p.ρ, ψ=p.ψ, μ=p.μ, σ=p.σ, γ=p.γ,
                           γhat1=p.γhat2, γhat2=p.γhat1, κhat=p.κhat, ω1=1.0 - p.ω1)
        inner = inner_series_left(x̃, ε, p_swapped)
        # Map back by swapping roles: at right boundary, agent 1 dominates
        # y is invariant, c1↔c2
        # Controls: swap α's, keep π same, reflect σx, μx to right boundary
        Δr = (p.γhat1 - p.γhat2)
        σx_r = (1.0 - x) * Δr * p.σ
        μx_r = (1.0 - x) * ( (p.γ - 1.0)*p.σ^2*Δr - p.κhat )
        return (; c1=inner.c2, c2=inner.c1, y=inner.y,
                 π=inner.π, α1=inner.α2, α2=inner.α1, σx=σx_r, μx=μx_r)
    end

    # -----------------------
    # Composite approximation blending inner (left/right) and outer solutions.
    # Weights: wL = exp(-δ x / ε), wR = exp(-δ (1-x) / ε), wO = 1 - wL - wR (clamped to [0,1]).
    # δ chosen from boundary-layer scale δ = y*/(κ̂ ω1) with guards for κ̂≈0.
    # -----------------------
    function composite_series_at_x(x::Float64, ε::Float64, p::Params; δ_scale::Float64=1.0, y_floor::Float64=1e-8, clamp_weights::Bool=true)
        outer = series_at_x(x, ε, p)
        # Compute δ with guard against division by zero
        y0 = max(outer.y0, y_floor)
        if p.κhat <= 1e-12 || p.ω1 <= 1e-12 || p.ω1 >= 1.0 - 1e-12
            # No mean reversion or degenerate ω1: fall back to outer
            return outer
        end
        # Optional tuning factor for layer thickness
        δ = δ_scale * y0 / (p.κhat * p.ω1)
        if !(isfinite(δ)) || δ <= 0
            return outer
        end
        # Weights
        expoL = -δ * x / max(ε, 1e-8)
        expoR = -δ * (1.0 - x) / max(ε, 1e-8)
        if clamp_weights
            expoL = clamp(expoL, -50.0, 50.0)
            expoR = clamp(expoR, -50.0, 50.0)
        end
        wL = exp(expoL)
        wR = exp(expoR)
        wO = 1.0 - (wL + wR)
        # Clamp to [0,1]
        if wO < 0.0
            wO = 0.0
            s = wL + wR
            if s > 0
                wL /= s; wR /= s
            else
                wL = 0.5; wR = 0.5
            end
        end
        # Inner approximations
        innerL = inner_series_left(x, ε, p)
        innerR = inner_series_right(x, ε, p)
        # Blend
        c1 = wL*innerL.c1 + wR*innerR.c1 + wO*outer.c1
        c2 = wL*innerL.c2 + wR*innerR.c2 + wO*outer.c2
        y  = wL*innerL.y  + wR*innerR.y  + wO*outer.y
        π  = wL*innerL.π  + wR*innerR.π  + wO*outer.π
        α1 = wL*innerL.α1 + wR*innerR.α1 + wO*outer.α1
        α2 = wL*innerL.α2 + wR*innerR.α2 + wO*outer.α2
        σx = wL*innerL.σx + wR*innerR.σx + wO*outer.σx
        μx = wL*innerL.μx + wR*innerR.μx + wO*outer.μx
        return (; c1, c2, y,
                 y0=outer.y0, y1=outer.y1, y2=outer.y2,
                 π=π, α1=α1, α2=α2, σx=σx, μx=μx)
    end

    # -----------------------
    # Optional: convert to a simple table (no dependencies)
    # -----------------------
    function to_dataframe(xgrid::AbstractVector{<:Real}, res::Vector{NamedTuple})
        # Minimal table-like structure
        cols = (
            x = collect(Float64.(xgrid)),
            c1 = [r.c1 for r in res],
            c2 = [r.c2 for r in res],
            y  = [r.y  for r in res],
            π  = [r.π  for r in res],
            α1 = [r.α1 for r in res],
            α2 = [r.α2 for r in res],
            σx = [r.σx for r in res],
            μx = [r.μx for r in res]
        )
        return cols
    end

end



# -----------------------
# Example usage (uncomment to run directly)
# -----------------------
# using .PerturbationClosedForm
# p = PerturbationClosedForm.Params()
# ε = 0.2
# xs = range(0.05, 0.95, length=11)
# res = PerturbationClosedForm.grid_series(xs, ε, p)
# tbl = PerturbationClosedForm.to_dataframe(xs, res)
# for i in eachindex(xs)
#     @show xs[i], tbl.c1[i], tbl.c2[i], tbl.y[i], tbl.π[i], tbl.α1[i], tbl.α2[i]
# end