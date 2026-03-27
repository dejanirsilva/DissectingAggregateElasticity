#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Chebyshev-collocation solver for the heterogeneous-agent boundary-layer model.

Unknowns:
  - u(s)  = log y(x), with x = exp(-s)
  - wa(s) = log c_a(x)
  - wb(s) = log c_b(x)
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


# ---------- Chebyshev grid and differentiation ----------

def cheb_lobatto_nodes_and_D(N, S):
    """Chebyshev-Lobatto nodes on [0,S] and first/second diff matrices."""
    if N < 4:
        raise ValueError("Use N >= 4 for stable second derivatives.")
    k = np.arange(N)
    t = -np.cos(np.pi * k / (N - 1))
    s = 0.5 * S * (t + 1)
    c = np.ones(N)
    c[0] = c[-1] = 2.0
    c = c * ((-1.0) ** k)
    X = np.tile(t, (N, 1))
    dX = X - X.T + np.eye(N)
    D = (np.outer(c, 1/c)) / dX
    D -= np.diag(np.sum(D, axis=1))
    dt_ds = 2.0 / S
    D1 = D * dt_ds
    D2 = (D @ D) * (dt_ds ** 2)
    return s, D1, D2


# ---------- Parameters and residual ----------

@dataclass
class Params:
    psi: float
    rho: float
    mu: float
    sigma: float
    gamma_a: float
    gamma_b: float


def residual(v, p: Params, s, x, D1, D2):
    """Euler residuals + market clearing at Chebyshev nodes."""
    N = len(s)
    u, wa, wb = v[0:N], v[N:2*N], v[2*N:3*N]

    y, ca, cb = np.exp(u), np.exp(wa), np.exp(wb)

    us, uss = D1 @ u, D2 @ u
    was, wass = D1 @ wa, D2 @ wa
    wbs, wbss = D1 @ wb, D2 @ wb

    q = -us / x
    yxx_over_y = (uss + us) / (x**2) + (us**2) / (x**2)

    wax = -was / x
    waxx_over = (wass + was) / (x**2) + (was**2) / (x**2)
    wbx = -wbs / x
    wbxx_over = (wbss + wbs) / (x**2) + (wbs**2) / (x**2)

    gbar = 1.0 / (x/p.gamma_a + (1-x)/p.gamma_b)
    αa = gbar / p.gamma_a
    αb = gbar / p.gamma_b

    denom = 1 + x * (αa - 1) * q
    denom = np.where(np.abs(denom) < 1e-12, 1e-12 * np.sign(denom)+1e-12, denom)

    σx = x * (αa - 1) * p.sigma / denom
    σR = p.sigma - q * σx
    σy = q * σx
    π = gbar * σR**2

    μx = x * (αa - 1) * (π - σR**2)
    μy = q * μx + 0.5 * yxx_over_y * σx**2
    μca = wax * μx + 0.5 * waxx_over * σx**2
    μcb = wbx * μx + 0.5 * wbxx_over * σx**2

    Ra = ca - (p.psi*p.rho
               + (1-p.psi)*(y + p.mu + σy**2 - p.sigma*σy
                            + π*(αa-1) - 0.5*p.gamma_a*σR**2 - μy)
               + μca)
    Rb = cb - (p.psi*p.rho
               + (1-p.psi)*(y + p.mu + σy**2 - p.sigma*σy
                            + π*(αb-1) - 0.5*p.gamma_b*σR**2 - μy)
               + μcb)

    Ry = y - ((1-x)*ca + x*cb)
    return np.concatenate([Ra, Rb, Ry])

# %%
# ---------- Newton solver & diagnostics ----------

def newton_solve(p: Params, N=120, S=10.0, maxit=25, tol=1e-7, fd_eps=1e-6, verbose=True):
    s, D1, D2 = cheb_lobatto_nodes_and_D(N, S)
    x = np.exp(-s)

    c0a = p.rho + (1/p.psi - 1)*(p.mu - 0.5*p.gamma_a*p.sigma**2)
    c0b = p.rho + (1/p.psi - 1)*(p.mu - 0.5*p.gamma_b*p.sigma**2)
    ca0 = np.full(N, c0a)
    cb0 = np.full(N, c0b)
    y0 = (1-x)*ca0 + x*cb0
    u0 = np.log(np.maximum(y0, 1e-14))
    v = np.concatenate([u0, np.log(ca0), np.log(cb0)])

    for it in range(1, maxit+1):
        r = residual(v, p, s, x, D1, D2)
        rn = np.linalg.norm(r, np.inf)
        if verbose: print(f"it {it:02d}  ||r||∞ = {rn:.3e}")
        if rn < tol: break

        nvar = v.size
        J = np.zeros((r.size, nvar))
        step = fd_eps * np.maximum(1, np.abs(v))
        for j in range(nvar):
            vv = v.copy(); vv[j] += step[j]
            J[:, j] = (residual(vv, p, s, x, D1, D2) - r) / step[j]
        δ = -np.linalg.solve(J.T @ J + 1e-10*np.eye(nvar), J.T @ r)
        α = 1.0
        while α > 1e-3:
            vv = v + α*δ
            if np.linalg.norm(residual(vv, p, s, x, D1, D2), np.inf) <= rn:
                v = vv; break
            α *= 0.5

    u, wa, wb = v[0:N], v[N:2*N], v[2*N:3*N]
    y, ca, cb = np.exp(u), np.exp(wa), np.exp(wb)

    us, uss = D1 @ u, D2 @ u
    q = -us / x
    Bx = x * q
    Hx = (uss + us) / x**2 + (us**2) / x**2

    gbar = 1 / (x/p.gamma_a + (1-x)/p.gamma_b)
    αa = gbar / p.gamma_a
    denom = 1 + x*(αa-1)*q
    denom = np.where(np.abs(denom)<1e-12, 1e-12*np.sign(denom)+1e-12, denom)
    σx = x*(αa-1)*p.sigma / denom
    σR = p.sigma - q*σx
    σy = q*σx

    return dict(x=x, y=y, ca=ca, cb=cb, Bx=Bx, Hx=Hx, σR=σR, σy=σy)


# ---------- Plotting ----------
# %%
def plot_solution(sol):
    x = sol["x"]
    plt.figure(); plt.plot(x, sol["Bx"]); plt.xscale("log")
    plt.title("B(x)"); plt.xlabel("x"); plt.ylabel("x q(x)")
    plt.figure(); plt.plot(x, sol["Hx"]); plt.xscale("log")
    plt.title("H(x)"); plt.xlabel("x"); plt.ylabel("x^2 y''/y")
    plt.figure(); plt.plot(x, sol["σR"], label="σ_R")
    plt.plot(x, sol["σy"], label="σ_y"); plt.xscale("log"); plt.legend(); plt.title("σ's")
    plt.figure(); plt.plot(x, sol["y"]); plt.xscale("log"); plt.title("Price–dividend")
    plt.figure(); plt.plot(x, sol["ca"], label="c_a"); plt.plot(x, sol["cb"], label="c_b")
    plt.xscale("log"); plt.legend(); plt.title("Consumptions")
    plt.show()


# ---------- Demo ----------

if __name__ == "__main__":
    p = Params(0.5, 0.02, 0.01, 0.20, 6.0, 2.0)
    sol = newton_solve(p)
    plot_solution(sol)
