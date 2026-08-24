# The corrected PTW transition (PTW_corrigendum_numerics.tex): forward KFE,
# backward HJB with per-node level imposition, outer fixed point on (g, E).

import math

import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
from model import (AlgorithmSettings, StructuralParameters, hjb_diagonals,
                   hjb_operators, kfe_operators, omega_weights, profits,
                   statics, t_grid, transport_rates, z_grid)
from stationary_solution import solve_stationary
from utilities import configure_process, fixed_point


# Forward KFE (eq:kfe): implicit Euler on the Scharfetter-Gummel chain, g at
# the arrival node of each step.
def march_forward(f_0, S_0, ts, g_path, args, p, alg_set):
    def step(f, xk):
        g, dt = xk
        down, up = transport_rates(g, args, p)
        # (I − Δt·(M + s·I))f⁺ = f rows 0..P−1,   f⁺_P = r_tail·f⁺_{P−1},
        # s = down₀·f⁺₀/Σf⁺;  exact tridiagonal solve conditional on s,
        # alternated with re-evaluating s
        base = 1.0 + dt * (down + up)
        lower = jnp.concatenate([-dt * up[:-2], -args["r_tail"][None]])
        upper = -dt * down[1:]
        rhs = jnp.concatenate([f[:-1], jnp.zeros_like(f[:1])])

        def alternate(f_new, _):
            s = down[0] * f_new[0] / f_new.sum()
            diagonal = jnp.concatenate([(base - dt * s)[:-1],
                                        jnp.ones_like(f[:1])])
            operator = lx.TridiagonalLinearOperator(diagonal, lower, upper)
            f_next = lx.linear_solve(operator, rhs, lx.Tridiagonal(),
                                     throw=False).value
            return f_next, jnp.max(jnp.abs(f_next - f_new))

        # the implicit system is linear given the scalar s, so a few fixed
        # passes converge it (last resid reported in the diagnostics)
        f_new, resid = jax.lax.scan(alternate, f, length=alg_set.kfe_passes)
        return f_new, (f_new, down[0] * f_new[0] / f_new.sum(), resid[-1])

    # one implicit-Euler step per time interval
    _, (f_path, S_path, resid) = jax.lax.scan(step, f_0,
                                              (g_path[1:], jnp.diff(ts)))
    f_path = jnp.concatenate([f_0[None, :], f_path])
    S_path = jnp.concatenate([S_0[None], S_path])
    diag = dict(kfe_resid=jnp.max(resid), min_f=jnp.min(f_path),
                mass_err=jnp.max(jnp.abs(f_path.sum(axis=1) - 1.0)))
    return f_path, S_path, diag


# Backward HJB (eq:hjb): a 3-D root solve per node picks
# θ_k = (g, log(ẑ−1), E) so the stepped value satisfies eq:algebraic.
def node_residual(theta, args, p):
    g, q, E = theta
    dt = args["dt"]
    z_hat = 1.0 + jnp.exp(q)
    st = statics(z_hat, E, args, p)
    # L̇_k = [log(1−L̃_{k+1}) − log(1−L̃(θ_k))]/Δt_k
    L_dot = (jnp.log(1.0 - args["L_next"])
             - jnp.log(1.0 - st["L_tilde"])) / dt
    lower, diagonal, upper = hjb_diagonals(g, L_dot, args, p)
    # (I + Δt_k·A(θ_k))·v_k = v_{k+1} + Δt_k·π̃(θ_k)
    operator = lx.TridiagonalLinearOperator(
        1.0 + dt * diagonal, dt * lower, dt * upper)
    rhs = args["v_next"] + dt * profits(st["pi_min"], z_hat, args, p)
    v = lx.linear_solve(operator, rhs, lx.Tridiagonal(), throw=False).value
    # value matching    Ξ₁·v₁ − ω·v + ζ = 0        (eq:algebraic)
    # export threshold  ẑ^{σ−1} − κ·d^{σ−1}/π̄_min = 0
    # free entry        Ξ₁·v₁ − ζ·(1−χ)/χ = 0
    s = p.sigma - 1.0
    resid = jnp.array([
        args["Xi_1"] * v[0] - args["omega"] @ v + p.zeta,
        z_hat ** s - p.kappa * args["d"] ** s / st["pi_min"],
        args["Xi_1"] * v[0] - p.zeta * (1.0 - p.chi) / p.chi])
    return resid, (v, st)


def march_backward(ts, f_path, S_path, Omega_grid, v_T, L_T, theta_T, args, p,
                   alg_set):
    x = args["x"]
    solver = optx.Newton(rtol=alg_set.node_rtol, atol=alg_set.node_atol)

    # per backward node: root-find θ_k against node_residual, warm-started
    # from the node just solved
    def node(carry, xk):
        v_next, L_next, theta_warm = carry
        f_k, S_k, Omega_k, dt_k = xk
        data = dict(args, f=f_k, S=S_k, Omega=Omega_k, dt=dt_k,
                    v_next=v_next, L_next=L_next,
                    omega=omega_weights(f_k, x, p.sigma - 1.0))
        sol = optx.root_find(lambda theta, a: node_residual(theta, a, p),
                             solver, theta_warm, args=data, has_aux=True,
                             max_steps=alg_set.node_max_steps, throw=False)
        # re-evaluated at the accepted root
        resid, (v_k, st) = node_residual(sol.value, data, p)
        out = dict(theta=sol.value, v=v_k, L_tilde=st["L_tilde"],
                   z_bar=st["z_bar"], lambda_ii=st["lambda_ii"],
                   resid=jnp.max(jnp.abs(resid)),
                   ok=sol.result == optx.RESULTS.successful)
        return (v_k, st["L_tilde"], sol.value), out

    xs = (f_path[:-1], S_path[:-1], Omega_grid[:-1], jnp.diff(ts))
    _, out = jax.lax.scan(node, (v_T, L_T, theta_T), xs, reverse=True)
    return out


def march_backward_evaluate(ts, theta_path, f_path, S_path, Omega_grid, v_T,
                            L_T, args, p):
    """Evaluate the backward equations at fixed controls on fixed states."""
    x = args["x"]

    def node(carry, xk):
        v_next, L_next = carry
        theta_k, f_k, S_k, Omega_k, dt_k = xk
        data = dict(args, f=f_k, S=S_k, Omega=Omega_k, dt=dt_k,
                    v_next=v_next, L_next=L_next,
                    omega=omega_weights(f_k, x, p.sigma - 1.0))
        resid, (v_k, st) = node_residual(theta_k, data, p)
        out = dict(theta=theta_k, v=v_k, L_tilde=st["L_tilde"],
                   z_bar=st["z_bar"], lambda_ii=st["lambda_ii"],
                   resid=resid)
        return (v_k, st["L_tilde"]), out

    xs = (theta_path, f_path[:-1], S_path[:-1], Omega_grid[:-1],
          jnp.diff(ts))
    _, out = jax.lax.scan(node, (v_T, L_T), xs, reverse=True)
    return out


# Outer fixed point on x = (g, E), both on the shared time grid;
# Omega is derived from E, never iterated (eq:Omega-from-E).

def Omega_path(E, ts, Omega_0, delta):
    # Ω(t) = Ω₀·exp∫₀ᵗ(E−δ), exact for piecewise-linear E
    seg = 0.5 * ((E - delta)[:-1] + (E - delta)[1:]) * jnp.diff(ts)
    return Omega_0 * jnp.exp(jnp.concatenate([jnp.zeros_like(seg[:1]),
                                              jnp.cumsum(seg)]))


# one full sweep of the outer iteration: KFE forward on g_path, HJB back on
# the resulting paths
@jax.jit
def transition_dynamics_iteration(g_path, E_path, boundary, p, alg_set):
    z_ex = z_grid(alg_set)
    kfe_ops = kfe_operators(z_ex, p.theta)
    kfe_args = dict(hm=kfe_ops["hm"], hp=kfe_ops["hp"],
                    r_tail=kfe_ops["r_tail"])
    # the cut to d_T at t = 0 is permanent and unanticipated
    hjb_args = dict(hjb_operators(z_ex, p.sigma - 1.0), x=kfe_ops["x"],
                    d=p.d_T)
    ts = boundary["ts"]
    Omega_grid = Omega_path(E_path, ts, boundary["Omega_0"], p.delta)
    f_path, S_path, kfe_diag = march_forward(boundary["f_0"], boundary["S_0"],
                                             ts, g_path, kfe_args, p, alg_set)
    out = march_backward(ts, f_path, S_path, Omega_grid, boundary["v_T"],
                         boundary["L_T"], boundary["theta_T"], hjb_args, p,
                         alg_set)
    g_new = jnp.append(out["theta"][:, 0], boundary["theta_T"][0])
    E_new = jnp.append(out["theta"][:, 2], boundary["theta_T"][2])
    diag = dict(kfe_diag, hjb_resid=jnp.max(jnp.abs(out["resid"])),
                hjb_ok=jnp.all(out["ok"]), min_E=jnp.min(E_new))
    return g_new, E_new, dict(out, f=f_path, S=S_path), diag


# fixed-point map on the flattened x = (g, E); the level mode is unstable
# under plain iteration and is carried by the Anderson extrapolation
def transition_dynamics_map(x, boundary, p, alg_set):
    K = boundary["ts"].shape[0]
    g_new, E_grid, _, _ = transition_dynamics_iteration(x[:K], x[K:],
                                                        boundary, p, alg_set)
    return jnp.concatenate([g_new, E_grid])


def check_transition_solution(diag, alg_set):
    # acceptance stated as pass conditions so NaNs fail
    gates = {key: float(diag[key]) for key in
             ("hjb_resid", "hjb_root_resid", "kfe_resid", "min_f",
              "mass_err", "min_E", "final_err_g", "final_err_E",
              "terminal_omega")}
    if not (bool(diag["hjb_ok"])
            and gates["hjb_root_resid"] <= alg_set.hjb_resid_tol
            and gates["hjb_resid"] <= alg_set.assembled_hjb_resid_tol
            and gates["kfe_resid"] <= alg_set.kfe_resid_tol
            and gates["min_f"] >= alg_set.min_f_tol
            and gates["mass_err"] <= alg_set.mass_err_tol
            and gates["min_E"] >= 0.0
            and gates["final_err_g"] <= alg_set.final_consistency_tol
            and gates["final_err_E"] <= alg_set.final_consistency_tol
            and abs(gates["terminal_omega"])
            <= alg_set.terminal_omega_tol):
        raise RuntimeError(f"accepted path fails acceptance gates: {gates}")
    return gates


def solve_transition(boundary, p, alg_set, verbose=False):
    ts = boundary["ts"]
    K = ts.shape[0]
    target = jnp.log(boundary["Omega_T"] / boundary["Omega_0"])

    def level_shift(E_vec):
        # c* = [log(Ω_T/Ω₀) − ∫₀ᵀ(E−δ)]/T   (eq:E-level-condition)
        return (target - jnp.trapezoid(E_vec - p.delta, ts)) / ts[-1]

    E_0 = jnp.full(K, p.delta)
    x0 = jnp.concatenate([jnp.full(K, boundary["theta_T"][0]),
                          E_0 + level_shift(E_0)])
    x_star, outer_iters, _ = fixed_point(
        transition_dynamics_map, x0, (boundary, p, alg_set),
        depth=alg_set.anderson_depth, tol=alg_set.outer_tol,
        maxit=alg_set.outer_maxit, ridge=alg_set.anderson_ridge,
        verbose=verbose)
    g_path, E_path = x_star[:K], x_star[K:]

    # One undamped consistency iteration supplies the controls used for the
    # reported path.
    g_new, E_new, out_root, diag_root = transition_dynamics_iteration(
        g_path, E_path, boundary, p, alg_set)

    # states re-marched under the returned controls, so the returned f
    # solves the KFE at the returned g and Omega integrates the returned E
    kfe_ops = kfe_operators(z_grid(alg_set), p.theta)
    f_path, S_path, kfe_diag = march_forward(
        boundary["f_0"], boundary["S_0"], ts, g_new,
        dict(hm=kfe_ops["hm"], hp=kfe_ops["hp"], r_tail=kfe_ops["r_tail"]),
        p, alg_set)
    Omega_grid = Omega_path(E_new, ts, boundary["Omega_0"], p.delta)

    # Evaluate the HJB once more on exactly those re-marched states, holding
    # the reported controls fixed.  Its algebraic residual is therefore the
    # residual of the assembled returned arrays, rather than of an adjacent
    # nonlinear sweep.
    hjb_args = dict(hjb_operators(z_grid(alg_set), p.sigma - 1.0),
                    x=kfe_ops["x"], d=p.d_T)
    out = march_backward_evaluate(
        ts, out_root["theta"], f_path, S_path, Omega_grid, boundary["v_T"],
        boundary["L_T"], hjb_args, p)
    final_err_g = jnp.max(jnp.abs(g_new - g_path))
    final_err_E = jnp.max(jnp.abs(E_new - E_path))
    terminal_omega = jnp.log(Omega_grid[-1] / boundary["Omega_T"])
    diag = dict(kfe_diag, hjb_resid=jnp.max(jnp.abs(out["resid"])),
                hjb_root_resid=diag_root["hjb_resid"],
                hjb_ok=diag_root["hjb_ok"], min_E=jnp.min(E_new),
                final_err_g=final_err_g, final_err_E=final_err_E,
                terminal_omega=terminal_omega)
    gates = check_transition_solution(diag, alg_set)

    theta_T = boundary["theta_T"]
    # the appended terminal row combines the marched transition states at
    # t = T with the terminal-BGP controls theta_T
    return dict(
        t=ts, g=g_new, E=E_new,
        z_hat=jnp.append(1.0 + jnp.exp(out["theta"][:, 1]),
                         1.0 + jnp.exp(theta_T[1])),
        L_tilde=jnp.append(out["L_tilde"], boundary["L_T"]),
        z_bar=jnp.append(out["z_bar"], boundary["z_bar_T"]),
        lambda_ii=jnp.append(out["lambda_ii"], boundary["lambda_ii_T"]),
        f=f_path, S=S_path, v0=out["v"][0], Omega=Omega_grid,
        outer_iters=outer_iters,
        final_err_g=float(final_err_g),
        final_err_E=float(final_err_E),
        c_star=float(level_shift(E_new)),
        terminal_omega=float(terminal_omega),
        **{key: value for key, value in gates.items()
           if key not in ("final_err_g", "final_err_E", "terminal_omega")})


# U(0) = ∫₀^∞ e^{−ρt}[log M + log c]dt   (eq:U-dynamics)
def welfare_u0(t, log_M, c, g_T, rho):
    flow = log_M + jnp.log(c)
    U0 = jnp.trapezoid(jnp.exp(-rho * t) * flow, t)
    # BGP continuation: e^{−ρT}·[(log M_T + log c_T)/ρ + g_T/ρ²]
    return U0 + jnp.exp(-rho * t[-1]) * (flow[-1] / rho + g_T / rho ** 2)


def solve_transition_dynamics(
    params: StructuralParameters = StructuralParameters(),
    alg_set: AlgorithmSettings = AlgorithmSettings(),
    verbose: bool = False,
) -> dict:
    stat_0 = solve_stationary(params, params.d_0, alg_set,
                              jnp.array([0.008, 0.0, math.log(0.68)]))
    stat_T = solve_stationary(params, params.d_T, alg_set,
                              jnp.array([1.2 * stat_0["g"],
                                         jnp.log(0.95 * stat_0["z_hat"] - 1.0),
                                         jnp.log(0.95 * stat_0["Omega"])]))
    for tag, stat in (("initial", stat_0), ("terminal", stat_T)):
        resids = {k: float(stat[k]) for k in ("stationary_resid",
                                              "bgp_resid")}
        if not (bool(stat["result"] == optx.RESULTS.successful)
                and max(resids.values()) <= alg_set.stationary_tol):
            raise RuntimeError(f"{tag} BGP solve failed: {resids}")

    ts = t_grid(alg_set)
    # BVP boundary data: initial (d_0) density and mass, terminal (d_T)
    # values and BGP point
    boundary = dict(ts=ts, f_0=stat_0["f"], S_0=stat_0["S"],
                    Omega_0=stat_0["Omega"], Omega_T=stat_T["Omega"],
                    v_T=stat_T["v"], L_T=stat_T["L_tilde"],
                    theta_T=jnp.array([stat_T["g"],
                                       jnp.log(stat_T["z_hat"] - 1.0),
                                       params.delta]),
                    z_bar_T=stat_T["z_bar"], lambda_ii_T=stat_T["lambda_ii"])
    paths = solve_transition(boundary, params, alg_set, verbose=verbose)

    rho = params.rho
    c = (1.0 - paths["L_tilde"]) * paths["z_bar"]
    # reported impact selection g(0) = g(0⁻) (sec:impact); its accumulated
    # log M feeds welfare and the figures, while paths["g"][0] keeps the raw
    # first cell of the discrete solution
    g_W = paths["g"].at[0].set(stat_0["g"])
    t = paths["t"]
    log_M = jnp.concatenate([jnp.zeros_like(t[:1]),
                             jnp.cumsum(0.5 * (g_W[1:] + g_W[:-1])
                                        * jnp.diff(t))])
    U0 = welfare_u0(t, log_M, c, g_W[-1], rho)
    # BGP: Ū = (ρ·log c + g)/ρ²;  CE = e^{ρ·ΔU} − 1
    U_bar_0 = (rho * jnp.log(stat_0["c"]) + stat_0["g"]) / rho ** 2
    U_bar_T = (rho * jnp.log(stat_T["c"]) + stat_T["g"]) / rho ** 2
    welfare = dict(U0=float(U0), g0_W=float(stat_0["g"]),
                   CE_transition=float(jnp.exp(rho * (U0 - U_bar_0)) - 1.0),
                   CE_ss_to_ss=float(jnp.exp(rho * (U_bar_T - U_bar_0)) - 1.0))
    diagnostics = dict(
        {key: paths[key] for key in
         ("outer_iters", "final_err_g", "final_err_E", "c_star",
          "terminal_omega", "hjb_resid", "kfe_resid",
          "hjb_root_resid", "min_f", "mass_err", "min_E")},
        stationary_0_resid=float(stat_0["bgp_resid"]),
        stationary_T_resid=float(stat_T["bgp_resid"]))

    if verbose:
        for tag, stat in (("stationary_0", stat_0), ("stationary_T", stat_T)):
            print(f"{tag}: " + "  ".join(
                f"{key}={float(stat[key]):.6f}" for key in
                ("g", "z_hat", "Omega", "L_tilde", "c", "S")))
        print("welfare: " + "  ".join(
            f"{key}={value:.6%}" if key.startswith("CE") else
            f"{key}={value:.6f}" for key, value in welfare.items()))
        print(f"transition: outer_iters={diagnostics['outer_iters']}  "
              f"final_err_g={diagnostics['final_err_g']:.2e}  "
              f"hjb_resid={diagnostics['hjb_resid']:.2e}  "
              f"kfe_resid={diagnostics['kfe_resid']:.2e}")

    return dict(
        stationary_0=stat_0, stationary_T=stat_T,
        **{key: paths[key] for key in
           ("t", "g", "z_hat", "Omega", "E", "L_tilde",
            "z_bar", "lambda_ii", "S", "f", "v0")},
        c=c, g_reporting=g_W, log_M=log_M, welfare=welfare,
        diagnostics=diagnostics)


if __name__ == "__main__":
    import jsonargparse

    configure_process()
    jsonargparse.CLI(solve_transition_dynamics)
