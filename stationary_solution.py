# Stationary BGP of the corrected PTW model at trade cost d, built on the
# quasi-stationary density of the absorbed firm chain.

import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
from model import (hjb_diagonals, hjb_operators, kfe_operators, omega_weights,
                   profits, statics, transport_rates, z_grid)


def stationary_distribution(g, args, p, alg_set):
    # quasi-stationary density of the absorbed chain:
    # (M + S·I)f = 0 rows 0..P−1,   f_P = r_tail·f_{P−1},   Σf = 1
    x, r_tail = args["x"], args["r_tail"]
    upsilon2 = p.upsilon ** 2
    n = x.shape[0]

    # Scharfetter-Gummel up/down jump intensities of the discretized chain
    down, up = transport_rates(g, args, p)

    def residual(f, S):
        # (Mf)_i = up_{i−1}·f_{i−1} + down_{i+1}·f_{i+1} − (down_i + up_i)·f_i
        Mf = (-(down + up) * f
              + jnp.concatenate([down[1:] * f[1:], jnp.zeros_like(f[:1])])
              + jnp.concatenate([jnp.zeros_like(f[:1]), up[:-1] * f[:-1]]))
        return jnp.concatenate([(Mf + S * f)[:n - 1],
                                f[n - 1:] - r_tail * f[n - 2:n - 1],
                                (f.sum() - 1.0)[None]])

    # start: f ∝ (e^{−θx} − e^{−ξ₂x})·w while the inherited tail is the
    # slower power, else e^{−θx}·w with S floored at (g−μ)²/(4υ²)
    w_cell = 0.5 * (args["hm"] + args["hp"])
    S_closed = p.theta * (g - p.mu - p.theta * upsilon2 / 2.0)
    xi2 = 2.0 * (g - p.mu) / upsilon2 - p.theta
    two_power = (S_closed > 0.0) & (xi2 > p.theta)
    f0 = jnp.where(two_power,
                   (jnp.exp(-p.theta * x) - jnp.exp(-xi2 * x)) * w_cell,
                   jnp.exp(-p.theta * x) * w_cell)
    f0 = f0 / f0.sum()
    S0 = jnp.where(two_power, S_closed,
                   jnp.maximum(S_closed, 0.25 * (g - p.mu) ** 2 / upsilon2))

    def newton(state, _):
        f, S = state
        G = residual(f, S)
        # freeze converged and non-finite iterates by masking to the identity/
        # zero solve -- AD through lineax throws on non-finite systems
        G_max = jnp.max(jnp.abs(G))
        take = (G_max >= alg_set.stationary_resid_tol) & jnp.isfinite(G_max)

        # bordered Newton: [A, ∂G/∂S; 𝟙ᵀ, 0][df; dS] = G, ∂G/∂S = (f; 0);
        # Schur: two tridiagonal solves and one scalar
        diag = jnp.concatenate([(-(down + up) + S)[:-1], jnp.ones_like(f[:1])])
        operator = lx.TridiagonalLinearOperator(
            jnp.where(take, diag, 1.0),
            jnp.where(take, jnp.concatenate([up[:-2], -r_tail[None]]), 0.0),
            jnp.where(take, down[1:], 0.0))
        # y₁ = A⁻¹G_f, y₂ = A⁻¹(f;0); dS = (Σy₁−G_mass)/Σy₂, df = y₁−dS·y₂
        y1 = lx.linear_solve(operator, jnp.where(take, G[:n], 0.0),
                             lx.Tridiagonal(), throw=False).value
        y2 = lx.linear_solve(
            operator,
            jnp.where(take, jnp.concatenate([f[:-1], jnp.zeros_like(f[:1])]),
                      0.0),
            lx.Tridiagonal(), throw=False).value
        dS = (y1.sum() - jnp.where(take, G[n], 0.0)) / jnp.where(
            take, y2.sum(), 1.0)
        return (jnp.where(take, f - (y1 - dS * y2), f),
                jnp.where(take, S - dS, S)), None

    # fixed-length scan, not while_loop: reverse-mode AD cannot cross one
    (f, S), _ = jax.lax.scan(newton, (f0, S0),
                             length=alg_set.stationary_steps)
    return f, S, jnp.max(jnp.abs(residual(f, S)))


# stationary BGP: Newton on θ = (g, log(ẑ−1), log Ω), each iterate nesting
# the stationary-density and HJB solves inside the equilibrium residuals
@jax.jit
def solve_stationary(p, d, alg_set, theta0):
    z_ex = z_grid(alg_set)
    s = p.sigma - 1.0
    kfe_ops = kfe_operators(z_ex, p.theta)
    hjb_ops = hjb_operators(z_ex, s)
    x = kfe_ops["x"]
    dist_args = dict(x=x, hm=kfe_ops["hm"], hp=kfe_ops["hp"],
                     r_tail=kfe_ops["r_tail"])

    def bgp(theta):
        g, q, w = theta
        # ẑ = 1 + e^q, Ω = e^w: every iterate admissible
        z_hat, Omega = 1.0 + jnp.exp(q), jnp.exp(w)
        f, S, stationary_resid = stationary_distribution(g, dist_args, p,
                                                         alg_set)
        st = statics(z_hat, p.delta,
                     dict(x=x, d=d, f=f, Omega=Omega, S=S), p)
        sub, diag, sup = hjb_diagonals(g, 0.0, hjb_ops, p)   # A bands, L̇=0
        pi = profits(st["pi_min"], z_hat, dict(z=hjb_ops["z"], d=d), p)
        # stationary HJB:  A·ṽ = π̃
        v = lx.linear_solve(lx.TridiagonalLinearOperator(diag, sub, sup), pi,
                            lx.Tridiagonal(), throw=False).value
        omega = omega_weights(f, x, s)
        # value matching    Ξ₁·v₁ − ω·v + ζ = 0        (eq:algebraic)
        # export threshold  ẑ^{σ−1} − κ·d^{σ−1}/π̄_min = 0
        # free entry        Ξ₁·v₁ − ζ·(1−χ)/χ = 0
        resid = jnp.array([
            hjb_ops["Xi_1"] * v[0] - omega @ v + p.zeta,
            z_hat ** s - p.kappa * d ** s / st["pi_min"],
            hjb_ops["Xi_1"] * v[0] - p.zeta * (1.0 - p.chi) / p.chi])
        return dict(st, g=g, z_hat=z_hat, Omega=Omega, v=v, f=f, S=S,
                    c=(1.0 - st["L_tilde"]) * st["z_bar"],
                    stationary_resid=stationary_resid, resid=resid)

    # bgp_atol must sit above the value-matching cancellation floor
    # (~P·eps·|v|·cond) or the solver runs to the step cap at the root
    sol = optx.root_find(lambda theta, _: bgp(theta)["resid"],
                         optx.Newton(rtol=alg_set.bgp_rtol,
                                     atol=alg_set.bgp_atol),
                         theta0, max_steps=alg_set.bgp_max_steps, throw=False)
    out = bgp(sol.value)   # re-evaluated at the accepted root
    out["bgp_resid"] = jnp.max(jnp.abs(out.pop("resid")))
    out["result"] = sol.result
    out["phi"] = sol.value
    return out
