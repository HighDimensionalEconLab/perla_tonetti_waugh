# Parameter and settings containers, grids and discrete operators of
# PTW_corrigendum_numerics.tex, and the per-instant model functions.

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


# rho, delta data-pinned; (theta, sigma) literature-pinned; the rest is the
# accepted SMM fit (production drivers load the estimation JSON instead)
@partial(jax.tree_util.register_dataclass,
         data_fields=["rho", "sigma", "theta", "kappa", "chi", "mu", "upsilon",
                      "zeta", "delta", "N", "d_0", "d_T"],
         meta_fields=[])
@dataclass(frozen=True)
class StructuralParameters:
    rho: float = 0.020338044668517
    sigma: float = 3.166924135838110
    theta: float = 4.988976587938262
    kappa: float = 0.0901185740532064
    chi: float = 0.15338102536555157
    mu: float = -0.03247291335115772
    upsilon: float = 0.05633114382539044
    zeta: float = 1.0
    delta: float = 0.02
    N: float = 10.0
    d_0: float = 3.04827083188645
    d_T: float = 2.8434437486978048  # 10% cut in the iceberg margin d - 1


# meta_fields are compile-time structure (changing one recompiles);
# data_fields are traced or host-read and never recompile.
@partial(jax.tree_util.register_dataclass,
         data_fields=["z_segment_1", "z_segment_2", "z_max",
                      "t_segment_1", "t_segment_2",
                      "anderson_depth", "outer_maxit", "T", "outer_tol",
                      "anderson_ridge", "final_consistency_tol",
                      "hjb_resid_tol", "assembled_hjb_resid_tol",
                      "terminal_omega_tol",
                      "kfe_resid_tol", "min_f_tol", "mass_err_tol",
                      "stationary_resid_tol", "stationary_tol", "node_rtol",
                      "node_atol", "bgp_rtol", "bgp_atol"],
         meta_fields=["z_segment_1_points", "z_segment_2_points",
                      "z_segment_3_points", "t_segment_1_points",
                      "t_segment_2_points", "t_segment_3_points",
                      "kfe_passes", "stationary_steps", "node_max_steps",
                      "bgp_max_steps"])
@dataclass(frozen=True)
class AlgorithmSettings:
    # grid defaults are the manuscript's production resolution
    # z grid: piecewise-uniform on [0, z_segment_1, z_segment_2, z_max]
    z_segment_1: float = 0.1
    z_segment_2: float = 1.0
    z_max: float = 5.0
    z_segment_1_points: int = 1440
    z_segment_2_points: int = 1920
    z_segment_3_points: int = 960
    # time grid: piecewise-uniform on [0, t_segment_1, t_segment_2, T], dense
    # early; g and E share it
    t_segment_1: float = 10.0
    t_segment_2: float = 20.0
    t_segment_1_points: int = 161
    t_segment_2_points: int = 81
    t_segment_3_points: int = 221
    kfe_passes: int = 12
    # Newton from the closed-form start converges in ~4 steps everywhere
    # tested; 8 doubles that margin (unconverged chains are NaN-masked)
    stationary_steps: int = 8
    node_max_steps: int = 32
    bgp_max_steps: int = 32
    # host-read solver knobs: data fields even where int or bool
    anderson_depth: int = 10
    outer_maxit: int = 80
    T: float = 75.0
    outer_tol: float = 1e-7
    anderson_ridge: float = 1e-12
    final_consistency_tol: float = 1e-7
    hjb_resid_tol: float = 1e-9
    # Nonlinear HJB roots meet hjb_resid_tol. The fixed-control evaluation
    # additionally contains the accepted outer fixed-point error.
    assembled_hjb_resid_tol: float = 2e-5
    # Maximum absolute log gap between marched and terminal-BGP variety mass.
    terminal_omega_tol: float = 5e-4
    kfe_resid_tol: float = 1e-12
    min_f_tol: float = -1e-10
    mass_err_tol: float = 1e-6
    stationary_resid_tol: float = 1e-13
    # one order above bgp_atol: the Newton stopping tolerance is also the
    # achieved-residual floor, so a gate at bgp_atol rejects converged solves
    stationary_tol: float = 1e-8
    node_rtol: float = 1e-10
    node_atol: float = 1e-10
    bgp_rtol: float = 1e-12
    bgp_atol: float = 1e-9


# -- grids

def z_grid(alg_set):
    a = jnp.linspace(0.0, alg_set.z_segment_1, alg_set.z_segment_1_points)
    b = jnp.linspace(alg_set.z_segment_1, alg_set.z_segment_2,
                     alg_set.z_segment_2_points)
    c = jnp.linspace(alg_set.z_segment_2, alg_set.z_max,
                     alg_set.z_segment_3_points)
    return jnp.concatenate([a, b[1:], c[1:]])


def t_grid(alg_set):
    a = jnp.linspace(0.0, alg_set.t_segment_1, alg_set.t_segment_1_points)
    b = jnp.linspace(alg_set.t_segment_1, alg_set.t_segment_2,
                     alg_set.t_segment_2_points)
    c = jnp.linspace(alg_set.t_segment_2, alg_set.T,
                     alg_set.t_segment_3_points)
    return jnp.concatenate([a, b[1:], c[1:]])


# -- operators on the extended grid: z_ex[0] = 0 is the adoption barrier

def hjb_operators(z_ex, s):
    h_ex = jnp.diff(z_ex)
    hm, hp = h_ex[:-1], h_ex[1:]
    # ghosts: v₀ = Ξ₁·v₁ (smooth pasting), v_{P+1} = v_P (far-field Neumann)
    Xi_1 = 1.0 / (1.0 - s * h_ex[0])   # Ξ₁ = 1/(1 − (σ−1)h₀)
    # (L₁v)_i = (v_i − v_{i−1})/h⁻_i   (upwind for ν < 0)
    L1_diag = jnp.concatenate([((1.0 - Xi_1) / hm[0])[None], 1.0 / hm[1:]])
    L1_sub = -1.0 / hm[1:]
    # (L₂v)_i = 2v_{i−1}/(h⁻(h⁻+h⁺)) − 2v_i/(h⁻h⁺) + 2v_{i+1}/(h⁺(h⁻+h⁺))
    L2_sub = 2.0 / (hm[1:] * (hm[1:] + hp[1:]))
    L2_sup = 2.0 / (hp[:-1] * (hm[:-1] + hp[:-1]))
    core = -2.0 / (hm * hp)
    L2_diag = jnp.concatenate(
        [(core[0] + Xi_1 * 2.0 / (hm[0] * (hm[0] + hp[0])))[None],
         core[1:-1],
         (core[-1] + 2.0 / (hp[-1] * (hm[-1] + hp[-1])))[None]])
    return dict(z=z_ex[1:-1], Xi_1=Xi_1,
                L1_sub=L1_sub, L1_diag=L1_diag,
                L2_sub=L2_sub, L2_diag=L2_diag, L2_sup=L2_sup)


def kfe_operators(z_ex, theta):
    # h⁻_i = x_i − x_{i−1},  h⁺_i = x_{i+1} − x_i  (h⁺_P = h⁻_P)
    x = z_ex[1:]
    hm = jnp.diff(z_ex)
    hp = jnp.concatenate([jnp.diff(x), hm[-1:]])
    w_cell = 0.5 * (hm + hp)   # w_i = (h⁻_i + h⁺_i)/2
    # r_tail = e^{−θ·h⁻_P}·(w_P/w_{P−1})   (eq:tailrow)
    r_tail = jnp.exp(-theta * (x[-1] - x[-2])) * w_cell[-1] / w_cell[-2]
    return dict(x=x, hm=hm, hp=hp, r_tail=r_tail)


# -- per-instant model functions

def transport_rates(g, args, p):
    # down_i = b/(w_i·(e^{b·h⁻_i/D} − 1)),  up_i = −b/(w_i·(e^{−b·h⁺_i/D} − 1))
    # (eq:sg),  b = μ − g,  D = υ²/2;  down₀·f₀ is the barrier absorption.
    # NaN at b = 0 exactly, rejected with the trial step like any other
    # inadmissible iterate (admissibility requires g > μ)
    hm, hp = args["hm"], args["hp"]
    b = p.mu - g
    D = p.upsilon ** 2 / 2.0
    w = 0.5 * (hm + hp)
    # up[-1] never enters: the tail row replaces row P
    down = b / (w * jnp.expm1(b * hm / D))
    up = -b / (w * jnp.expm1(-b * hp / D))
    return down, up


def tail_weight(x, lower_x):
    # mass fraction above lower_x, the straddling cell split linearly
    seg = jnp.clip((x[1:] - lower_x) / (x[1:] - x[:-1]), 0.0, 1.0)
    return jnp.concatenate([seg, (x[-1:] >= lower_x).astype(x.dtype)])


def omega_weights(f, x, s):
    # ω_i = f_i·e^{(σ−1)x_i}; the top node (at the HJB ghost) folds onto v_P
    w = f * jnp.exp(s * x)
    return jnp.concatenate([w[:-2], w[-2:-1] + w[-1:]])


def statics(z_hat, E, args, p):
    x, f, Omega, d = args["x"], args["f"], args["Omega"], args["d"]
    s = p.sigma - 1.0
    above = tail_weight(x, jnp.log(z_hat))
    weighted = f * jnp.exp(s * x)
    # Ezs = Σ f·e^{(σ−1)x};  Jx = Σ f·e^{(σ−1)x}·1{x ≥ log ẑ};  1−F(ẑ) = Σ f·1{…}
    Ezs = weighted.sum()
    Jx = weighted @ above
    one_mF = f @ above
    # export moment: (N−1)·d^{1−σ}·Jx
    export_moment = (p.N - 1) * d ** (-s) * Jx
    # L̃ = Ω·[(N−1)(1−F(ẑ))·κ + ζ·(S + E/χ)]
    L_tilde = Omega * ((p.N - 1) * one_mF * p.kappa
                       + p.zeta * (args["S"] + E / p.chi))
    # z̄^{σ−1} = Ω·[Σ f·e^{(σ−1)x} + export moment]
    zbar_pow = Omega * (Ezs + export_moment)
    # π̄_min = (1 − L̃)/((σ−1)·z̄^{σ−1})
    return dict(L_tilde=L_tilde, pi_min=(1.0 - L_tilde) / (s * zbar_pow),
                z_bar=zbar_pow ** (1.0 / s),
                lambda_ii=1.0 / (1.0 + export_moment / Ezs),
                Ezs=Ezs, Jx=Jx, one_mF=one_mF)


def profits(pi_min, z_hat, args, p):
    z, d = args["z"], args["d"]
    s = p.sigma - 1.0
    # π̃ = π̄_min·(1 + (N−1)·d^{1−σ}·1{x ≥ log ẑ})
    #     − (N−1)·κ·e^{−(σ−1)x}·1{x ≥ log ẑ}
    # the indicator's ẑ-jump cancels at the export-threshold root (AD-sound)
    exports = (z >= jnp.log(z_hat)).astype(z.dtype)
    return (pi_min * (1.0 + (p.N - 1) * d ** (-s) * exports)
            - (p.N - 1) * p.kappa * jnp.exp(-s * z) * exports)


# separate from hjb_operators: the grid stencils are fixed, these bands
# carry the (g, L̇)-dependent coefficients and rebuild per iterate
def hjb_diagonals(g, Ldot, args, p):
    # A = ρ̃·I − ν·L₁ − D·L₂   (eq:hjb)
    s = p.sigma - 1.0
    D = p.upsilon ** 2 / 2.0
    # ρ̃ = ρ + δ + d/dt log(1−L̃) − (σ−1)(μ − g + (σ−1)·υ²/2)
    rho_tilde = (p.rho + p.delta + Ldot) - s * (p.mu - g + s * D)
    # ν = μ − g + (σ−1)·υ²
    nu = p.mu - g + s * p.upsilon ** 2
    return (-nu * args["L1_sub"] - D * args["L2_sub"],
            rho_tilde - nu * args["L1_diag"] - D * args["L2_diag"],
            -D * args["L2_sup"])
