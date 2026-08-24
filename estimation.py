# SMM estimation of the corrected PTW model: closed-form BGP in (g, ẑ, Ω)
# against the two-power stationary density, the discrete 5-year quartile
# transition operator (discretization_notes.md), and the 12-moment SMM
# objective.  Cancellation-guarded kernels and the discretized post-fit
# verification live in estimation_robust.py.

import dataclasses
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax.scipy.linalg import expm
from model import StructuralParameters, transport_rates
from stationary_solution import stationary_distribution

# W = diag(100, 100, 100, 100, 1, ..., 1)
SMM_WEIGHTS = jnp.concatenate([100.0 * jnp.ones(4), jnp.ones(8)])


# -- integrals of f(z) = θξ₂/(ξ₂−θ)·(z^{−θ−1} − z^{−ξ₂−1})  (eq:moment-formula)
# as plain antiderivatives; off-equilibrium exponent collisions (p = θ, p = ξ₂,
# ξ₂ = θ) evaluate to garbage or NaN and are rejected with the trial step like
# any other infeasible iterate

def dist_I_inf(p_, l, tail, xi2):
    # ∫_l^∞ z^p f dz = θξ₂/(ξ₂−θ)·(l^{p−θ}/(θ−p) − l^{p−ξ₂}/(ξ₂−p)),
    # at l = 1 equal to θξ₂/((θ−p)(ξ₂−p));  divergent at p ≥ θ: finite
    # sentinel, rejected by acceptance
    q = p_ - tail
    q_safe = jnp.where(q < 0.0, q, -1.0)
    val = (tail * xi2 / (xi2 - tail)
           * (l ** q_safe / (-q_safe) - l ** (p_ - xi2) / (xi2 - p_)))
    return jnp.where(q < 0.0, val, 1e10)


def dist_I_1u(p_, u, tail, xi2):
    # ∫_1^u z^p f dz = θξ₂/(ξ₂−θ)·((u^{p−θ}−1)/(p−θ) − (u^{p−ξ₂}−1)/(p−ξ₂))
    return (tail * xi2 / (xi2 - tail)
            * ((u ** (p_ - tail) - 1.0) / (p_ - tail)
               - (u ** (p_ - xi2) - 1.0) / (p_ - xi2)))


def dist_I_scaled_beta(beta, zhat, tail, xi2):
    # ẑ^{−β}·∫_1^ẑ z^β f dz
    #   = θξ₂/(ξ₂−θ)·((ẑ^{−θ} − ẑ^{−β})/(β−θ) − (ẑ^{−ξ₂} − ẑ^{−β})/(β−ξ₂)),
    # every power decayed: no overflow at large β
    zb = zhat ** (-beta)
    return (tail * xi2 / (xi2 - tail)
            * ((zhat ** (-tail) - zb) / (beta - tail)
               - (zhat ** (-xi2) - zb) / (beta - xi2)))


def one_minus_F(z, tail, xi2):
    # 1 − F(z) = (ξ₂·z^{−θ} − θ·z^{−ξ₂})/(ξ₂−θ)
    return (xi2 * z ** (-tail) - tail * z ** (-xi2)) / (xi2 - tail)


# -- closed-form value function: roots −ν, β of (υ²/2)η² + (μ−g)η − (r−g) = 0,
# r − g = ρ + δ at γ = 1;  v = A·z^{σ−1} + c₁·z^{−ν} + c₂·z^β on [1, ẑ],
# v = A(1+(N−1)d^{1−σ})·z^{σ−1} − K/(r−g) + c₃·z^{−ν} on [ẑ, ∞), K = (N−1)κ

def value_function(g, zhat, pi_min, p):
    s = p.sigma - 1.0
    ups2 = p.upsilon ** 2
    rg = p.rho + p.delta
    disc = jnp.sqrt((g - p.mu) ** 2 + 2.0 * ups2 * rg)
    nu = (disc - (g - p.mu)) / ups2
    beta = ((g - p.mu) + disc) / ups2
    # 1/a = ρ̃ at L̇ = 0; a ≤ 0 gets a finite sentinel, rejected by acceptance
    inv_a = rg - s * (p.mu - g + s * ups2 / 2.0)
    a = jnp.where(inv_a > 1e-8, 1.0 / jnp.where(inv_a > 1e-8, inv_a, 1.0),
                  1e8)
    A = a * pi_min
    K = (p.N - 1.0) * p.kappa
    Q = K * (1.0 - a * rg) / rg
    # c̃₂ = K·a·s·ν(ν+s)(υ²/2)/((r−g)(ν+β))   (C¹ pasting; c₂ = c̃₂·ẑ^{−β})
    c2t = K * a * s * nu * (nu + s) * (ups2 / 2.0) / (rg * (nu + beta))
    zb = jnp.exp(-beta * jnp.log(zhat))
    # c₁ = (s·A + β·c̃₂·ẑ^{−β})/ν   (v′(1) = 0)
    c1 = (s * A + beta * c2t * zb) / nu
    # c₃ = c₁ + (Q + c̃₂)·ẑ^ν       (C⁰ pasting)
    c3 = c1 + (Q + c2t) * zhat ** nu
    v1 = A * (1.0 + s / nu) + c2t * zb * (nu + beta) / nu
    return dict(nu=nu, beta=beta, a=a, A=A, K=K, c2t=c2t, c1=c1, c3=c3,
                v1=v1, rg=rg)


# -- the 3-equation BGP system in φ = (g, log(ẑ−1), log Ω)

def bgp_quantities(phi, d, p):
    g, q, w = phi
    z_hat, Omega = 1.0 + jnp.exp(q), jnp.exp(w)
    s = p.sigma - 1.0
    # inherited tails θ, ξ₂ = 2(g−μ)/υ² − θ (eq:F-corr); existence ξ₂ ≥ θ
    # checked at acceptance
    tail = p.theta
    xi2 = 2.0 * (g - p.mu) / p.upsilon ** 2 - p.theta
    # S = θ·(g − μ − θ·υ²/2) = (υ²/2)·θ·ξ₂
    S = tail * (g - p.mu - tail * p.upsilon ** 2 / 2.0)
    one_mF = one_minus_F(z_hat, tail, xi2)
    # L̃ = Ω·[(N−1)(1−F(ẑ))·κ + ζ·(S + δ/χ)]   (BGP entry E = δ)
    L_tilde = Omega * ((p.N - 1.0) * one_mF * p.kappa
                       + p.zeta * (S + p.delta / p.chi))
    Ezs = dist_I_inf(s, 1.0, tail, xi2)
    Jx = dist_I_inf(s, z_hat, tail, xi2)
    # z̄^{σ−1} = Ω·[E[z^{σ−1}] + (N−1)d^{1−σ}·∫_ẑ^∞ z^{σ−1}f]
    zbar_pow = Omega * (Ezs + (p.N - 1.0) * d ** (-s) * Jx)
    # π̄_min = (1 − L̃)/((σ−1)·z̄^{σ−1}); L̃ ≥ 1 evaluates finitely, rejected
    # by acceptance (c > 0)
    pi_min = (1.0 - L_tilde) / (s * zbar_pow)
    vf = value_function(g, z_hat, pi_min, p)
    # E[v] = A·∫₁^ẑ z^{σ−1}f + c₁·∫₁^ẑ z^{−ν}f + c̃₂·ẑ^{−β}∫₁^ẑ z^β f
    #        + A(1+(N−1)d^{1−σ})·∫_ẑ^∞ z^{σ−1}f − K/(r−g)·(1−F) + c₃·∫_ẑ^∞ z^{−ν}f
    int_vf = (vf["A"] * dist_I_1u(s, z_hat, tail, xi2)
              + vf["c1"] * dist_I_1u(-vf["nu"], z_hat, tail, xi2)
              + vf["c2t"] * dist_I_scaled_beta(vf["beta"], z_hat, tail, xi2)
              + vf["A"] * (1.0 + (p.N - 1.0) * d ** (-s)) * Jx
              - vf["K"] / vf["rg"] * one_mF
              + vf["c3"] * dist_I_inf(-vf["nu"], z_hat, tail, xi2))
    # λ_ii = 1/(1 + (N−1)d^{1−σ}·∫_ẑ^∞ z^{σ−1}f / E[z^{σ−1}])
    lam_ii = 1.0 / (1.0 + (p.N - 1.0) * d ** (-s) * Jx / Ezs)
    z_bar = jnp.where(zbar_pow > 0.0,
                      jnp.where(zbar_pow > 0.0, zbar_pow, 1.0) ** (1.0 / s),
                      1.0)
    c = (1.0 - L_tilde) * z_bar
    # value matching    v(1) = E[v] − ζ
    # export threshold  ẑ^{σ−1}·π̄_min = κ·d^{σ−1}   (finite for any sign)
    # free entry        E[v] = ζ/χ
    resid = jnp.array([
        vf["v1"] - (int_vf - p.zeta),
        z_hat ** s * pi_min - p.kappa * d ** s,
        int_vf - p.zeta / p.chi])
    # Ū = (ρ·log c + g)/ρ²
    return dict(resid=resid, g=g, z_hat=z_hat, Omega=Omega, tail=tail,
                xi2=xi2, S=S, L_tilde=L_tilde, pi_min=pi_min, z_bar=z_bar,
                lam_ii=lam_ii, c=c, one_mF=one_mF, Ezs=Ezs, Jx=Jx,
                U_bar=(p.rho * jnp.log(jnp.where(c > 0.0, c, 1.0)) + g)
                / p.rho ** 2, **vf)


# Newton start: reaches the unique admissible root in every tested case
# (port notes) — approaching from small ẑ, Ω keeps iterates admissible
PHI0 = (0.0005, 1.5, 0.3)


def bgp_system(phi, args):
    d, p = args
    return bgp_quantities(phi, d, p)["resid"]


def check_bgp_solution(sol, out, bgp_resid, p):
    # acceptance: Newton converged on a small residual at an economically
    # admissible root; per-condition flags name what failed
    flags = dict(
        converged=(sol.result == optx.RESULTS.successful)
        & (bgp_resid < 1e-7),
        g_pos=out["g"] > 0.0,
        S_pos=out["S"] > 0.0,
        lam_ii_in_unit=(out["lam_ii"] >= 0.0) & (out["lam_ii"] <= 1.0),
        c_pos=out["c"] > 0.0,
        nu_pos=out["nu"] > 0.0,
        a_admissible=(out["a"] > 0.0) & (out["a"] < 1e7),
        tail_moment=out["tail"] > p.sigma - 1.0,
        xi2_ordered=out["xi2"] >= out["tail"])
    ok = jnp.array(True)
    for flag in flags.values():
        ok = ok & flag
    return ok, flags


def solve_bgp(d, p, alg_set, phi_0=None):
    # Newton in φ = (g, log(ẑ−1), log Ω)
    if phi_0 is None:
        g0, zh0, Om0 = PHI0
        phi_0 = jnp.array([g0, jnp.log(zh0 - 1.0), jnp.log(Om0)])
    d, p = jnp.asarray(d), jax.tree_util.tree_map(jnp.asarray, p)
    sol = optx.root_find(
        bgp_system, optx.Newton(rtol=alg_set.bgp_rtol, atol=alg_set.bgp_atol),
        phi_0, args=(d, p), max_steps=alg_set.bgp_max_steps, throw=False)
    out = bgp_quantities(sol.value, d, p)
    bgp_resid = jnp.max(jnp.abs(out["resid"]))
    ok, flags = check_bgp_solution(sol, out, bgp_resid, p)
    return dict(out, phi=sol.value, ok=ok, ok_flags=flags,
                bgp_resid=bgp_resid)


# -- 5-year quartile transition operator: CHAIN_M-point grid on [0, CHAIN_X_BAR]

CHAIN_M, CHAIN_X_BAR, CHAIN_T = 500, 7.0, 5.0

def quartile_weights(f_bar):
    # W[i,k]: fraction of node i's mass in quartile k, straddling cells
    # split fractionally; below-noise nodes assigned whole-cell by cdf
    total = f_bar.sum()
    cdf_hi = jnp.cumsum(f_bar)
    cdf_lo = cdf_hi - f_bar
    q = jnp.concatenate([jnp.array([0.0, 0.25, 0.5, 0.75]) * total,
                         cdf_hi[-1:]])
    overlap = (jnp.minimum(cdf_hi[:, None], q[None, 1:])
               - jnp.maximum(cdf_lo[:, None], q[None, :4]))
    pos = f_bar > 1e-12 * total
    f_safe = jnp.where(pos, f_bar, 1.0)
    W = jnp.clip(overlap, 0.0, None) / f_safe[:, None]
    W_sum = jnp.where(pos, W.sum(axis=1), 1.0)
    W = W / W_sum[:, None]
    bins = jnp.clip(jnp.searchsorted(q[1:4], cdf_hi), 0, 3)
    whole = jax.nn.one_hot(bins, 4, dtype=f_bar.dtype)
    return jnp.where(pos[:, None], W, whole)


@jax.jit
def transition_rows(g, p, alg_set):
    M = CHAIN_M
    x = jnp.linspace(0.0, CHAIN_X_BAR, M)
    dx = x[1] - x[0]
    # tail closure f_{M−1} = r_tail·f_{M−2},  r_tail = e^{−θ·Δx}
    r_tail = jnp.exp(-p.theta * dx)
    args = dict(x=x, hm=jnp.full(M, dx), hp=jnp.full(M, dx), r_tail=r_tail)
    f_bar, _, stationary_resid = stationary_distribution(g, args, p, alg_set)
    down, up = transport_rates(g, args, p)
    dn, up1 = down[0], up[0]
    # generator rows = origin: L[i, i±1] = up/down, top row down only,
    # barrier absorption reinjected as a draw from f̄
    idx = jnp.arange(M)
    L = (jnp.diag(jnp.where(idx < M - 1, -(dn + up1), -dn))
         + jnp.diag(jnp.full(M - 1, up1), 1)
         + jnp.diag(jnp.full(M - 1, dn), -1))
    L = L.at[0, :].add(dn * f_bar)
    # P4 = diag(f̄W)⁻¹·(f̄∘W)ᵀ·e^{TL}·W; rows Q3, Q4 are the targets
    P = expm(L * CHAIN_T)
    W = quartile_weights(f_bar)
    origin = f_bar[:, None] * W
    P4 = (origin.T @ (P @ W)) / origin.sum(axis=0)[:, None]
    return P4[2:, :], stationary_resid


# -- SMM: transforms, moments, residuals
# x = (d, κ, 1/χ, μ, υ); (θ, σ) are pinned in p_base

X_NAMES = ("d", "kappa", "inv_chi", "mu", "upsilon")
# d ↦ log(v−1);  κ, 1/χ, υ ↦ log v;  μ ↦ μ
T_FWD = dict(d=lambda v: jnp.log(v - 1.0), kappa=jnp.log, inv_chi=jnp.log,
             mu=lambda v: v, upsilon=jnp.log)
T_INV = dict(d=lambda y: 1.0 + jnp.exp(y), kappa=jnp.exp, inv_chi=jnp.exp,
             mu=lambda y: y, upsilon=jnp.exp)


def to_transformed(x):
    return jnp.array([T_FWD[n](v) for n, v in zip(X_NAMES, x)])


def from_transformed(y):
    return jnp.array([T_INV[n](v) for n, v in zip(X_NAMES, y)])


def params_from_x(x, p_base):
    d, kappa, inv_chi, mu, upsilon = x
    return d, dataclasses.replace(p_base, kappa=kappa, chi=1.0 / inv_chi,
                                  mu=mu, upsilon=upsilon)


@jax.jit
def model_moments(x, p_base, alg_set):
    d, p = params_from_x(x, p_base)
    eq = solve_bgp(d, p, alg_set)
    # mass_ex = 1 − F(ẑ);  size_ex = E[z^{σ−1}|z>ẑ]/E[z^{σ−1}|z<ẑ]
    # (relative DOMESTIC shipments, the BEJK data convention)
    mass_ex = eq["one_mF"]
    size_ex = ((eq["Jx"] / mass_ex)
               / ((eq["Ezs"] - eq["Jx"]) / (1.0 - mass_ex)))
    rows, stationary_resid = transition_rows(eq["g"], p, alg_set)
    okf = eq["ok"] & (stationary_resid < alg_set.stationary_resid_tol)
    mom = jnp.concatenate([jnp.array([eq["g"], eq["lam_ii"],
                                      mass_ex, size_ex]),
                           rows[0], rows[1]])
    # an inadmissible root or an unconverged chain can still be finite:
    # okf re-NaNs the moment vector
    mom = jnp.where(okf, mom, jnp.nan)
    return mom, eq, stationary_resid


def smm_objective(x, p_base, alg_set, targets):
    # Q = (1/12)·(log T − log m)′·W·(log T − log m)
    mom, _, _ = model_moments(x, p_base, alg_set)
    diff = jnp.log(targets) - jnp.log(mom)
    return (SMM_WEIGHTS * diff * diff).sum() / 12.0


@jax.jit
def smm_fit(y0, p_base, alg_set, targets):
    # BFGS on the SMM objective in the transformed coordinates; a NaN
    # objective at an infeasible trial fails the Armijo test and the
    # backtracking line search shrinks the step
    def fn(y, _):
        return smm_objective(from_transformed(y), p_base, alg_set, targets)
    sol = optx.minimise(fn, optx.BFGS(rtol=1e-8, atol=1e-10), y0,
                        max_steps=500, throw=False)
    return dict(y=sol.value, x=from_transformed(sol.value),
                objective=fn(sol.value, None),
                ok=sol.result == optx.RESULTS.successful,
                steps=sol.stats["num_steps"])


def welfare_gain_ce(eq_base, eq_cf, rho):
    # CE = e^{ρ·ΔŪ} − 1   (log utility)
    return jnp.exp(rho * (eq_cf["U_bar"] - eq_base["U_bar"])) - 1.0


# -- data targets and the SMM multistart

MOMENT_NAMES = ("g", "home_share", "frac_exporters", "rel_size_exporters",
                "Q3_Q1", "Q3_Q2", "Q3_Q3", "Q3_Q4",
                "Q4_Q1", "Q4_Q2", "Q4_Q3", "Q4_Q4")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# calibration-region warm starts: the corrected-baseline neighborhood and the
# published-paper point
X0_BASE = (3.079, 0.0997, 7.221, -0.0322, 0.0480)
X0_PUB = (3.0224928254626, 0.104196324793307, 7.88353731986437,
          -0.0310646242175711, 0.0483011406016648)


def load_targets():
    growth_r = np.loadtxt(os.path.join(DATA, "growth_and_r_moments.csv"))
    firm = np.loadtxt(os.path.join(DATA, "firm_moments.csv"), delimiter=",")
    bejk = np.loadtxt(os.path.join(DATA, "bejk_moments.csv"), delimiter=",")
    trade = float(np.loadtxt(os.path.join(DATA, "trade_moments.csv")))
    entry = float(np.loadtxt(os.path.join(DATA, "entry_moment.csv")))
    r_data, g_target = growth_r
    # additive row renormalization to sum 1 (conditioning on survival)
    firm = firm + ((1.0 - firm.sum(axis=1)) / 4.0)[:, None]
    targets = jnp.array(np.concatenate(([g_target, trade, bejk[0], bejk[1]],
                                        firm[0], firm[1])))
    # ρ = r_data − g_data (the γ = 1 Euler equation) and δ = the entry rate
    # are matched directly, never estimated
    return targets, float(r_data - g_target), entry


def load_estimated_parameters(path):
    # (theta, sigma, rho, delta) pinned, the rest from the saved fit; d_T is
    # the 10% cut in the iceberg margin d - 1
    if not os.path.exists(path):
        raise FileNotFoundError(f"no estimation output at {path}: run "
                                "replication_estimation.py first")
    with open(path) as f:
        fit = json.load(f)
    x = fit["x"]
    return StructuralParameters(
        rho=fit["rho"], delta=fit["delta"], theta=fit["theta"],
        sigma=fit["sigma"], kappa=x["kappa"], chi=1.0 / x["inv_chi"],
        mu=x["mu"], upsilon=x["upsilon"], d_0=x["d"],
        d_T=1.0 + 0.9 * (x["d"] - 1.0))


def estimate(y0s, p_base, alg_set, targets):
    best = None
    for k, y0 in enumerate(y0s):
        # the fit differentiates at its starting point, so an infeasible
        # start cannot be recovered from -- skip it
        obj0 = float(smm_objective(from_transformed(jnp.array(y0)), p_base,
                                   alg_set, targets))
        if not np.isfinite(obj0):
            print(f"    start {k}: infeasible start, skipped")
            continue
        t0 = time.time()
        fit = smm_fit(jnp.array(y0), p_base, alg_set, targets)
        # materialize before timing: dispatch is asynchronous
        obj = float(fit["objective"])
        wall = time.time() - t0
        print(f"    start {k}: objective={obj:.12f}  ok={bool(fit['ok'])}  "
              f"steps={int(fit['steps'])}  wall={wall:.1f}s")
        if bool(fit["ok"]) and (best is None or obj < best[0]):
            best = (obj, fit)
    if best is None:
        raise RuntimeError("no SMM start converged")
    return best[1]
