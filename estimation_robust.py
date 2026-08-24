# Ex-post verification of a saved SMM fit against two independent
# re-implementations of the BGP: a cancellation-guarded closed-form solve on
# divided-difference (expm1) kernels exact through the root merge ξ₂ = θ, and
# the brute-force discretized machinery of stationary_solution (QSD Newton
# density, tridiagonal HJB, grid-sum statics — no closed forms anywhere).
# Reads the json written by replication_estimation.py and never runs the SMM
# optimizer itself.  Run by replicate.sh; writes a verification_*.json summary.

import json
import os

from utilities import configure_process

configure_process()

import jax.numpy as jnp
import numpy as np
import optimistix as optx

from estimation import (MOMENT_NAMES, PHI0, X_NAMES, load_targets,
                        params_from_x, transition_rows)
from model import AlgorithmSettings, StructuralParameters
from stationary_solution import solve_stationary

EPS = float(jnp.finfo(jnp.float64).eps)


# -- divided-difference kernels, exact through the root merge ξ₂ = θ, where
# the density is the critical member f = ξ̄²·log z·z^{−ξ̄−1}, ξ̄ = (g−μ)/υ²,
# S = (g−μ)²/(2υ²); each switches to its series where the series truncation
# crosses the general branch's cancellation error

def em1r(y):
    # (1 − e^{−y})/y,  em1r(0) = 1; the expm1 branch is exact in value but
    # 0/0 at y = 0, and its AD gradient cancels like eps/|y| — matched by
    # the series' y²/8 derivative truncation at |y| = eps^{1/3}
    small = jnp.abs(y) < EPS ** (1 / 3)
    safe = jnp.where(small, 1.0, y)
    return jnp.where(small, 1.0 - y / 2.0 + y * y / 6.0,
                     -jnp.expm1(-safe) / safe)


def h2(y):
    # (y − 1 + e^{−y})/y²,  h2(0) = 1/2; general branch cancels like
    # 3·eps/y², series truncates at y⁴/720 — both ~eps^{2/3} at eps^{1/6}
    small = jnp.abs(y) < EPS ** (1 / 6)
    safe = jnp.where(small, 1.0, y)
    return jnp.where(small, 0.5 - y / 6.0 + y * y / 24.0 - y ** 3 / 120.0,
                     (safe - 1.0 + jnp.exp(-safe)) / (safe * safe))


def k3(y):
    # (y²/2 − 1 + e^{−y}(1+y))/y³,  k3(0) = 1/3; general branch cancels like
    # 4·eps/y³ (≤ 4·eps^{1/2} at the eps^{1/6} switch, series truncation
    # y³/144 = eps^{1/2}/144) — k3 is the q-weighted correction to the h2
    # term inside the collide window, so eps^{1/2}-level error is invisible
    small = jnp.abs(y) < EPS ** (1 / 6)
    safe = jnp.where(small, 1.0, y)
    return jnp.where(small, 1.0 / 3.0 - y / 8.0 + y * y / 30.0,
                     (safe * safe / 2.0 - 1.0
                      + jnp.exp(-safe) * (1.0 + safe)) / safe ** 3)


def collide(q, gap):
    # series where its O(q²) truncation beats the general branch's
    # cancellation ~eps/max(|q|·gap, q²):
    # q²·max(|q|·gap, q²) = |q|³·max(gap, |q|) < eps  (forces |q| < eps^{1/4})
    return jnp.abs(q) ** 3 * jnp.maximum(gap, jnp.abs(q)) < EPS


# -- integrals of f(z) = θξ₂/(ξ₂−θ)·(z^{−θ−1} − z^{−ξ₂−1})  (eq:moment-formula)

def dist_I_inf(p_, l, tail, xi2):
    # ∫_l^∞ z^p f dz = θξ₂·l^q·(1 − q·log l·em1r(gap·log l))/(q(p−ξ₂)),
    # q = p − θ; divergent at q ≥ 0: finite sentinel, rejected by acceptance
    gap = xi2 - tail
    q = p_ - tail
    L = jnp.log(l)
    q_safe = jnp.where(q < 0.0, q, -1.0)
    val = (tail * xi2 * l ** q_safe * (1.0 - q_safe * L * em1r(gap * L))
           / (q_safe * (p_ - xi2)))
    return jnp.where(q < 0.0, val, 1e10)


def dist_I_1u(p_, u, tail, xi2):
    # ∫_1^u z^p f dz = θξ₂·[u^q(qU·em1r(gap·U) − 1) + 1]/(q(p−ξ₂)), U = log u;
    # collision expansions θξ₂[U²h2(±gap·U) + q·U³k3(±gap·U)] at q → 0, p−ξ₂ → 0
    gap = xi2 - tail
    q = p_ - tail
    r = p_ - xi2
    U = jnp.log(u)
    c_q, c_r = collide(q, gap), collide(-r, gap)
    q_safe = jnp.where(c_q, 1.0, q)
    r_safe = jnp.where(c_r, 1.0, r)
    general = (tail * xi2 * (u ** q * (q * U * em1r(gap * U) - 1.0) + 1.0)
               / (q_safe * r_safe))
    exp_q = tail * xi2 * (U * U * h2(gap * U) + q * U ** 3 * k3(gap * U))
    exp_r = tail * xi2 * (U * U * h2(-gap * U) + r * U ** 3 * k3(-gap * U))
    return jnp.where(c_q, exp_q, jnp.where(c_r, exp_r, general))


def dist_I_scaled_beta(beta, zhat, tail, xi2):
    # ẑ^{−β}·∫_1^ẑ z^β f dz = θξ₂·[ẑ^{−θ}((β−θ)h·em1r(gap·h) − 1) + ẑ^{−β}]
    #                         /((β−θ)(β−ξ₂)),  h = log ẑ  (decayed: no overflow)
    gap = xi2 - tail
    qb = beta - tail
    rb = beta - xi2
    h = jnp.log(zhat)
    zb = jnp.exp(-beta * h)
    c_q, c_r = collide(qb, gap), collide(-rb, gap)
    q_safe = jnp.where(c_q, 1.0, qb)
    r_safe = jnp.where(c_r, 1.0, rb)
    general = (tail * xi2 * (zhat ** (-tail) * (qb * h * em1r(gap * h) - 1.0)
                             + zb) / (q_safe * r_safe))
    exp_q = zb * tail * xi2 * (h * h * h2(gap * h) + qb * h ** 3 * k3(gap * h))
    exp_r = zb * tail * xi2 * (h * h * h2(-gap * h) + rb * h ** 3 * k3(-gap * h))
    return jnp.where(c_q, exp_q, jnp.where(c_r, exp_r, general))


def one_minus_F(z, tail, xi2):
    # 1 − F(z) = z^{−θ}·(1 + θ·log z·em1r((ξ₂−θ)·log z))
    x = jnp.log(z)
    return z ** (-tail) * (1.0 + tail * x * em1r((xi2 - tail) * x))


def tail_roots(g, p):
    # inherited tails (θ, ξ₂ = 2(g−μ)/υ² − θ); existence ξ₂ ≥ θ checked at
    # acceptance
    return p.theta, 2.0 * (g - p.mu) / p.upsilon ** 2 - p.theta


def adoption_S(g, p, tail):
    # S = θ·(g − μ − θ·υ²/2) = (υ²/2)·θ·ξ₂
    return tail * (g - p.mu - tail * p.upsilon ** 2 / 2.0)


# -- closed-form value function: roots −ν, β of (υ²/2)η² + (μ−g)η − (r−g) = 0,
# r − g = ρ + δ at γ = 1;  v = A·z^{σ−1} + c₁·z^{−ν} + c₂·z^β on [1, ẑ],
# v = A(1+(N−1)d^{1−σ})·z^{σ−1} − K/(r−g) + c₃·z^{−ν} on [ẑ, ∞), K = (N−1)κ

def value_function(g, zhat, pi_min, p):
    s = p.sigma - 1.0
    ups2 = p.upsilon ** 2
    rg = p.rho + p.delta
    disc = jnp.sqrt((g - p.mu) ** 2 + 2.0 * ups2 * rg)
    # ν rationalized: no cancellation as υ → 0
    nu = 2.0 * rg / (disc + (g - p.mu))
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
    tail, xi2 = tail_roots(g, p)
    S = adoption_S(g, p, tail)
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


def bgp_system(phi, args):
    d, p = args
    return bgp_quantities(phi, d, p)["resid"]


def check_bgp_solution(sol, out, bgp_resid, p):
    # acceptance: Newton converged on a small residual at an economically
    # admissible root (growth, adoption flow, consumption, value-function
    # coefficients, and tail ordering all valid)
    return ((sol.result == optx.RESULTS.successful) & (bgp_resid < 1e-7)
            & (out["g"] > 0.0) & (out["S"] > 0.0)
            & (out["lam_ii"] >= 0.0) & (out["lam_ii"] <= 1.0)
            & (out["c"] > 0.0) & (out["nu"] > 0.0) & (out["a"] > 0.0)
            & (out["a"] < 1e7) & (out["tail"] > p.sigma - 1.0)
            & (out["xi2"] >= out["tail"]))


def solve_bgp(d, p, alg_set):
    g0, zh0, Om0 = PHI0
    sol = optx.root_find(
        bgp_system, optx.Newton(rtol=alg_set.bgp_rtol, atol=alg_set.bgp_atol),
        jnp.array([g0, jnp.log(zh0 - 1.0), jnp.log(Om0)]), args=(d, p),
        max_steps=alg_set.bgp_max_steps, throw=False)
    out = bgp_quantities(sol.value, d, p)
    bgp_resid = jnp.max(jnp.abs(out["resid"]))
    ok = check_bgp_solution(sol, out, bgp_resid, p)
    return dict(out, phi=sol.value, ok=ok, bgp_resid=bgp_resid)


def solve_bgp_grid(d, p, alg_set):
    # the same BGP from the discretized machinery: QSD Newton density,
    # tridiagonal HJB, grid-sum statics — no closed-form density, value
    # function, or analytic tail integrals anywhere in the solve
    g0, zh0, Om0 = PHI0
    phi0 = jnp.array([g0, jnp.log(zh0 - 1.0), jnp.log(Om0)])
    eq = solve_stationary(p, d, alg_set, phi0)
    c = eq["c"]
    # acceptance on the discrete diagnostics; θ > σ−1 keeps E[z^{σ−1}] finite
    # in the model (the truncated grid sum is finite regardless), and
    # ρ̃ > 0 keeps the HJB operator diagonally dominant — a ρ̃ ≤ 0 system
    # still solves finitely and would pass every other gate
    s = p.sigma - 1.0
    rho_t = (p.rho + p.delta
             - s * (p.mu - eq["g"] + s * p.upsilon ** 2 / 2.0))
    ok = ((eq["result"] == optx.RESULTS.successful)
          & (eq["bgp_resid"] < alg_set.stationary_tol)
          & (eq["stationary_resid"] < alg_set.stationary_tol)
          & (eq["g"] > 0.0) & (eq["S"] > 0.0) & (c > 0.0)
          & (jnp.min(eq["f"]) >= alg_set.min_f_tol)
          & (eq["lambda_ii"] >= 0.0) & (eq["lambda_ii"] <= 1.0)
          & (p.theta > s) & (rho_t > 0.0))
    return dict(eq, lam_ii=eq["lambda_ii"], ok=ok,
                U_bar=(p.rho * jnp.log(jnp.where(c > 0.0, c, 1.0)) + eq["g"])
                / p.rho ** 2)


# -- the verification: closed forms vs the discretized backend

def value_shape_checks(eq, d, p):
    # economic shape at a reported solution — conditions a zero residual does
    # not imply: v(z) > 0, z·v′(z) ≥ 0, the adoption obstacle v(z) ≥ v(1),
    # v″(1) ≥ 0
    n, span = 200, 6.0
    s = p.sigma - 1.0
    A, c1, c2t, c3 = eq["A"], eq["c1"], eq["c2t"], eq["c3"]
    nu, beta, zhat = eq["nu"], eq["beta"], eq["z_hat"]
    Ax = A * (1.0 + (p.N - 1.0) * d ** (-s))
    lz = jnp.log(zhat)
    z_lo = jnp.exp(jnp.linspace(0.0, lz, n))
    z_hi = jnp.exp(jnp.linspace(lz, lz + span, n))
    # z^β only as (z/ẑ)^β ≤ 1: β can reach hundreds
    ratio_b = jnp.exp(beta * (jnp.log(z_lo) - lz))
    v_lo = A * z_lo ** s + c1 * z_lo ** (-nu) + c2t * ratio_b
    v_hi = Ax * z_hi ** s - eq["K"] / eq["rg"] + c3 * z_hi ** (-nu)
    zvp_lo = s * A * z_lo ** s - nu * c1 * z_lo ** (-nu) + beta * c2t * ratio_b
    zvp_hi = s * Ax * z_hi ** s - nu * c3 * z_hi ** (-nu)
    v_min = jnp.minimum(v_lo.min(), v_hi.min())
    vpp1 = (s * (s - 1.0) * A + nu * (nu + 1.0) * c1
            + beta * (beta - 1.0) * c2t * jnp.exp(-beta * lz))
    return dict(v_min=v_min,
                zvp_min=jnp.minimum(zvp_lo.min(), zvp_hi.min()),
                obstacle_min=v_min - eq["v1"], vpp1=vpp1)


VERIFY_AGGS = ("g", "z_hat", "Omega", "S", "L_tilde", "pi_min", "z_bar",
               "lam_ii", "c", "U_bar", "Ezs", "Jx", "one_mF")


def verify_solution(x, p_base, alg_set, tol=0.03):
    # solve the BGP at x with both backends, compare the 12 SMM moments
    # (each through the transition chain at its own g) and the aggregates —
    # differences sit at the grid-truncation scale of the discretization —
    # and check the closed-form value function's economic shape
    d, p = params_from_x(x, p_base)
    eq_c = solve_bgp(d, p, alg_set)
    eq_g = solve_bgp_grid(d, p, alg_set)
    shape = value_shape_checks(eq_c, d, p)

    def moment_vector(eq):
        mass_ex = eq["one_mF"]
        size_ex = ((eq["Jx"] / mass_ex)
                   / ((eq["Ezs"] - eq["Jx"]) / (1.0 - mass_ex)))
        rows, _ = transition_rows(eq["g"], p, alg_set)
        return np.asarray(jnp.concatenate(
            [jnp.array([eq["g"], eq["lam_ii"], mass_ex, size_ex]),
             rows[0], rows[1]]))

    names = list(MOMENT_NAMES) + list(VERIFY_AGGS)
    closed = np.concatenate([moment_vector(eq_c),
                             [float(eq_c[k]) for k in VERIFY_AGGS]])
    grid = np.concatenate([moment_vector(eq_g),
                           [float(eq_g[k]) for k in VERIFY_AGGS]])
    rel = np.abs(grid - closed) / np.abs(closed)
    print(f"  {'quantity':<20}{'closed form':>16}{'discretized':>16}"
          f"{'rel diff':>12}")
    for name, vc, vg, r in zip(names, closed, grid, rel):
        print(f"  {name:<20}{vc:>16.8f}{vg:>16.8f}{r:>12.2e}")
    shape_min = min(float(v) for v in shape.values())
    print("  value shape: " + "  ".join(f"{k}={float(v):.3e}"
                                        for k, v in shape.items()))
    ok = (bool(eq_c["ok"]) and bool(eq_g["ok"]) and shape_min >= -1e-9
          and bool(np.all(np.isfinite(rel))) and float(np.max(rel)) < tol)
    worst = names[int(np.argmax(rel))]
    print(f"  {'PASS' if ok else 'FAIL'}: worst rel diff "
          f"{float(np.max(rel)):.2e} ({worst}) vs tol {tol:g}, "
          f"ok_closed={bool(eq_c['ok'])} ok_grid={bool(eq_g['ok'])} "
          f"shape_min={shape_min:.2e}")
    return dict(ok=ok, names=names, closed=closed, grid=grid, rel=rel,
                shape=shape, eq_closed=eq_c, eq_grid=eq_g)


def main(theta: float = 4.988976587938262,
         sigma: float = 3.166924135838110,
         est_dir: str = "output", tol: float = 0.03):
    path = os.path.join(est_dir,
                        f"estimation_theta{theta:g}_sigma{sigma:g}.json")
    with open(path) as f:
        fit = json.load(f)
    x_hat = jnp.array([fit["x"][name] for name in X_NAMES])
    _, rho, delta = load_targets()
    p_pin = StructuralParameters(rho=rho, delta=delta, theta=theta,
                                 sigma=sigma)
    print(f"verifying {path}:")
    print("  " + "  ".join(f"{n}={float(v):.9f}"
                           for n, v in zip(X_NAMES, x_hat)))
    result = verify_solution(x_hat, p_pin, AlgorithmSettings(), tol=tol)
    summary = dict(
        theta=theta, sigma=sigma, tol=tol, passed=bool(result["ok"]),
        max_rel_diff=float(np.max(result["rel"])),
        worst_quantity=result["names"][int(np.argmax(result["rel"]))],
        rel_by_quantity={n: float(r) for n, r
                         in zip(result["names"], result["rel"])})
    out_path = os.path.join(
        est_dir, f"verification_theta{theta:g}_sigma{sigma:g}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print(f"-> {out_path}  (max rel diff {summary['max_rel_diff']:.2e})")
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(main)
