# The corrigendum's tables as \input-ready fragments, from the two retained
# calculations plus the BGP/SMM machinery of estimation.py: the eq-(60)
# welfare decomposition (AD partials of the feasibility map, equilibrium
# responses by the implicit function theorem at the closed-form BGP root)
# and the published-path welfare re-aggregation on the frozen cache.
# Corrected/Data numbers are computed; Published columns are the typeset
# literals of data/published_tables.json, reproducing the published paper's
# tables.  Writes output/{table_*,decomposition_equation}.tex and
# inline_results.json, which records the rounded prose inputs.

import json
import os

from utilities import configure_process, load_transition_artifact

configure_process()

import jax
import jax.numpy as jnp
import numpy as np

import estimation as est
from model import AlgorithmSettings

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# -- eq. (60) of the published paper: dŪ/dd through the feasibility map
# f_c(Ω, ẑ, g; d), the consumption a feasible allocation delivers given the
# stationary distribution the KFE pins at g (so g enters through adoption
# labor S(g) AND the distribution shape ξ₂(g))

def f_c(Omega, z_hat, g, d, p):
    # plain kernels: this calibration sits far from the root merge ξ₂ = θ
    # (ξ₂ − θ ≈ 26); near the merge the divided-difference kernels of
    # estimation_robust.py are required instead
    s = p.sigma - 1.0
    tail = p.theta
    xi2 = 2.0 * (g - p.mu) / p.upsilon ** 2 - p.theta
    S = tail * (g - p.mu - tail * p.upsilon ** 2 / 2.0)
    # L̃ = Ω·[(N−1)(1−F(ẑ))·κ + ζ·(S(g) + δ/χ)]   (BGP entry E = δ)
    L_tilde = Omega * ((p.N - 1.0) * est.one_minus_F(z_hat, tail, xi2)
                       * p.kappa + p.zeta * (S + p.delta / p.chi))
    # z̄^{σ−1} = Ω·[E[z^{σ−1}] + (N−1)d^{1−σ}·∫_ẑ^∞ z^{σ−1}f]
    zbar_pow = Omega * (est.dist_I_inf(s, 1.0, tail, xi2)
                        + (p.N - 1.0) * d ** (-s)
                        * est.dist_I_inf(s, z_hat, tail, xi2))
    return (1.0 - L_tilde) * zbar_pow ** (1.0 / s)


def decompose(d_0, p, alg_set):
    # a non-converged Newton with throw=False still returns finite,
    # plausible derivatives, and the sum-vs-total gate below cannot see it
    # (both sides evaluate at the same point) — so gate admissibility first
    eq = est.solve_bgp(d_0, p, alg_set)
    if not bool(eq["ok"]):
        bad = [k for k, v in eq["ok_flags"].items() if not bool(v)]
        raise RuntimeError(f"decomposition: BGP root inadmissible: {bad}")
    d_0 = jnp.asarray(d_0)
    phi = eq["phi"]
    g, z_hat, Omega, c = eq["g"], eq["z_hat"], eq["Omega"], eq["c"]
    # the sum-vs-total gate is invariant to a level error in f_c (U₁ ∝ 1/c
    # cancels it against the partials), so pin the map to the equilibrium c
    if abs(float(f_c(Omega, z_hat, g, d_0, p) - c)) > 1e-12 * abs(float(c)):
        raise RuntimeError("decomposition: f_c does not reproduce the BGP c")

    # implicit function theorem at the root: φ′(d) = −(∂R/∂φ)⁻¹·∂R/∂d
    J = jax.jacfwd(lambda ph: est.bgp_quantities(ph, d_0, p)["resid"])(phi)
    b = jax.jacfwd(lambda dd: est.bgp_quantities(phi, dd, p)["resid"])(d_0)
    cond_J = float(np.linalg.cond(np.asarray(J)))
    if cond_J > 1e8:
        raise RuntimeError(f"decomposition: cond(∂R/∂φ) = {cond_J:.1e}")
    phi_p = -jnp.linalg.solve(J, b)

    # tangent check ‖R(φ*+hφ′, d+h)‖ = O(h²): validates every component of
    # φ′ — the export-margin envelope makes the decomposition itself
    # insensitive to ẑ′, so the sum-vs-total gate alone cannot
    def r_norm(h):
        return float(jnp.linalg.norm(
            est.bgp_quantities(phi + h * phi_p, d_0 + h, p)["resid"]))
    r1, r2 = r_norm(1e-3), r_norm(1e-4)
    if not (r1 < 1e-5 and 30.0 < r1 / r2 < 300.0):
        raise RuntimeError(f"decomposition: tangent not O(h²): "
                           f"r(1e-3)={r1:.2e}, r(1e-4)={r2:.2e}")

    # natural coordinates: ẑ = 1 + e^q, Ω = e^w
    g_p = phi_p[0]
    zhat_p = (z_hat - 1.0) * phi_p[1]
    Omega_p = Omega * phi_p[2]
    U1, U2 = 1.0 / (p.rho * c), 1.0 / p.rho ** 2
    fc_Om, fc_zh, fc_g, fc_d = (
        jax.grad(f_c, argnums=i)(Omega, z_hat, g, d_0, p) for i in range(4))
    terms = {
        "direct consumption": float(U1 * fc_d),
        "variety": float(U1 * fc_Om * Omega_p),
        "export threshold": float(U1 * fc_zh * zhat_p),
        "growth via consumption": float(U1 * fc_g * g_p),
        "direct growth": float(U2 * g_p)}

    # separately wired total: AD through the Newton solve itself (optimistix
    # implicit adjoint) — the same IFT identity on independent plumbing, so
    # it checks the Jacobian assembly and coordinate chain above
    eq_warm = est.solve_bgp(d_0, p, alg_set, phi_0=phi)
    if not bool(eq_warm["ok"]):
        raise RuntimeError("decomposition: warm-started BGP solve failed")
    total = float(jax.grad(
        lambda dd: est.solve_bgp(dd, p, alg_set, phi_0=phi)["U_bar"])(d_0))
    reldiff = abs(sum(terms.values()) - total) / abs(total)
    if reldiff > 1e-10:
        raise RuntimeError(f"decomposition: terms sum to "
                           f"{sum(terms.values()):.9f} but the separately "
                           f"wired total is {total:.9f} "
                           f"(reldiff {reldiff:.1e})")
    shares = {k: 100.0 * v / total for k, v in terms.items()}
    if abs(shares["export threshold"]) > 1e-8:
        raise RuntimeError("decomposition: export-threshold share "
                           f"{shares['export threshold']:.2e} is not the "
                           "envelope zero")

    # inefficiency ρ(U₁∂f_c/∂g + U₂): CE% of welfare per 1pp of growth
    # reallocated to the direct channel;  local ACR: CE per unit dd
    inefficiency = float(p.rho * (U1 * fc_g + U2))
    semi_elasticity = float(d_0 * g_p)
    def lam_fn(ph, dd):
        return est.bgp_quantities(ph, dd, p)["lam_ii"]
    dlam = (jax.grad(lam_fn, argnums=0)(phi, d_0) @ phi_p
            + jax.grad(lam_fn, argnums=1)(phi, d_0))
    acr_local = -(1.0 / p.theta) * dlam / eq["lam_ii"]
    direct_over_acr = 100.0 * float(p.rho * U1 * fc_d / acr_local)

    print(f"eq-(60) decomposition at the corrected calibration "
          f"(d = {float(d_0):.6f}):")
    print(f"  separately wired total dU/dd = {total:+.6f};  sum of terms = "
          f"{sum(terms.values()):+.6f}  (reldiff {reldiff:.1e})")
    print(f"  cond(dR/dphi) = {cond_J:.1f};  tangent O(h^2): "
          f"r(1e-3) = {r1:.2e}, r(1e-4) = {r2:.2e}")
    print(f"  {'term':>24s} {'level':>12s} {'% of total':>12s}")
    for k in terms:
        print(f"  {k:>24s} {terms[k]:+12.5f} {shares[k]:12.2f}")
    print(f"  inefficiency rho*(U1 dfc/dg + U2) = {inefficiency:.2f}")
    print(f"  growth semi-elasticity d*g'(d) = {semi_elasticity:+.4f}")
    print(f"  direct consumption / local ACR = {direct_over_acr:.1f}%")
    # persist the prose scalars alongside the shares (free-growth benchmark
    # rho*U2 = 1/rho)
    return dict(terms=terms, shares=shares, total=total,
                inefficiency=inefficiency,
                free_growth_benchmark=float(1.0 / p.rho),
                semi_elasticity=semi_elasticity,
                direct_consumption_over_acr_percent=direct_over_acr)


# -- the published-path welfare aggregation comparison: the published
# package's terminal continuation substitutes T·g(T) for the accumulated
# log M(T); re-aggregate its own frozen cached path under both

def published_reaggregation(rho, delta):
    path = os.path.join(DATA, "published_transition.csv")
    with open(path, encoding="utf-8") as fh:
        names = fh.readline().strip().split(",")
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    col = {name: raw[:, names.index(name)]
           for name in ("t", "g", "c", "log_M", "U", "r")}
    # the t = −1 row is the pre-shock BGP normalization (M = 1 there)
    pre = np.flatnonzero(col["t"] == -1.0)[0]
    if abs(col["r"][pre] - col["g"][pre] - delta - rho) > 1e-10:
        raise RuntimeError("published cache: r - g - delta != rho")
    U_bar_pre = col["U"][pre]
    U_bar_closed = (rho * np.log(col["c"][pre]) + col["g"][pre]) / rho ** 2
    if abs(U_bar_pre - U_bar_closed) > 1e-12 * abs(U_bar_pre):
        raise RuntimeError("published cache: pre-shock U is not the BGP U")

    # the path proper: t >= 0 only — log M is anchored at the shock, and the
    # pre-shock rows carry negative log M that must not enter the integral
    m = col["t"] >= 0.0
    t, g, c = col["t"][m], col["g"][m], col["c"][m]
    logM, U_csv = col["log_M"][m], col["U"][m]
    if logM[0] != 0.0:
        raise RuntimeError("published cache: log M(0) != 0")
    T, g_T = t[-1], g[-1]

    # primary: exact continuation swap on the cache's own U(0) and log M(T),
    # U_pub(0) − U_corr(0) = e^{−ρT}(T·g_T − log M(T))/ρ — this keeps the
    # cache's integral untouched and changes only the continuation slot
    gap_U = np.exp(-rho * T) * (T * g_T - logM[-1]) / rho
    ce = lambda U0: np.exp(rho * (U0 - U_bar_pre)) - 1.0
    ce_pub = ce(U_csv[0])
    ce_corr = ce(U_csv[0] - gap_U)

    # cross-checks: the cache's log M is the trapezoid of its own g, and a
    # trapezoid rebuild of U(0) reproduces the cache's integrator closely
    logM_trap = np.concatenate(
        [[0.0], np.cumsum(0.5 * (g[:-1] + g[1:]) * np.diff(t))])
    logM_err = np.max(np.abs(logM - logM_trap))
    if logM_err > 1e-4:
        raise RuntimeError(f"published cache: log M column deviates from "
                           f"the trapezoid of g by {logM_err:.1e}")
    integrand = np.exp(-rho * t) * (logM_trap + np.log(c))
    integral = np.sum(0.5 * (integrand[:-1] + integrand[1:]) * np.diff(t))
    cont = lambda logM_T: np.exp(-rho * T) * (
        (logM_T + np.log(c[-1])) / rho + g_T / rho ** 2)
    U_trap_pub = integral + cont(T * g_T)
    U_trap_corr = integral + cont(logM_trap[-1])
    U_err = abs(U_trap_pub - U_csv[0])
    if U_err > 5e-3:
        raise RuntimeError(f"published cache: trapezoid U(0) deviates from "
                           f"the cache by {U_err:.1e}")
    gap_trap = np.exp(-rho * T) * (T * g_T - logM_trap[-1]) / rho
    identity_err = abs((U_trap_pub - U_trap_corr) - gap_trap)
    if identity_err > 1e-12:
        raise RuntimeError("published-path continuation-gap identity failed")

    print(f"published-path welfare aggregation (frozen cache, T = {T:g}):")
    print(f"  log M(T) = {logM[-1]:.6f}  vs  T*g(T) = {T * g_T:.6f}  "
          f"(continuation gap in U: {gap_U:.6f})")
    print(f"  CE, published aggregation = {100 * ce_pub:.4f}%")
    print(f"  CE, corrected aggregation = {100 * ce_corr:.4f}%   "
          f"(gap {100 * (ce_pub - ce_corr):.4f}pp)")
    print(f"  cross-checks: |log M - trapezoid| = {logM_err:.1e};  "
          f"|U_trap(0) - U_cache(0)| = {U_err:.1e};  "
          f"identity error = {identity_err:.1e}")
    return dict(ce_pub_pct=100 * ce_pub, ce_corr_pct=100 * ce_corr)


def inline_results(eq_0, eq_T, p, alg_set, record, arrays, dec, agg):
    """Collect every calibration-dependent number used in running prose."""
    acr_log_pct = (100.0 / p.theta
                   * np.log(float(eq_0["lam_ii"] / eq_T["lam_ii"])))
    acr_ce_pct = 100.0 * np.expm1(acr_log_pct / 100.0)

    d_aut = 1.0 + 2.9 * (p.d_0 - 1.0)
    eq_prev = eq_0
    for d in np.linspace(p.d_0, d_aut, 31)[1:]:
        eq_prev = est.solve_bgp(float(d), p, alg_set, phi_0=eq_prev["phi"])
        if not bool(eq_prev["ok"]):
            raise RuntimeError(f"autarky continuation failed at d={d:g}")
    eq_aut = eq_prev
    loss_to_aut_pct = 100.0 * float(
        est.welfare_gain_ce(eq_0, eq_aut, p.rho))
    gain_from_aut_pct = 100.0 * float(
        est.welfare_gain_ce(eq_aut, eq_0, p.rho))
    acr_aut_log_pct = (100.0 / p.theta
                       * np.log(float(eq_aut["lam_ii"] / eq_0["lam_ii"])))

    t = arrays["t"]
    i_q = int(np.argmin(np.abs(t - 0.25)))
    eligible = np.flatnonzero(t >= 0.25)
    i_trough = int(eligible[np.argmin(arrays["g_reporting"][eligible])])
    stat_0, stat_T = record["stationary_0"], record["stationary_T"]

    L_a_0 = p.zeta * stat_0["Omega"] * stat_0["S"]
    L_E_0 = p.zeta * stat_0["Omega"] * p.delta / p.chi
    L_x_0 = stat_0["L_tilde"] - L_a_0 - L_E_0
    L_a_q = p.zeta * arrays["Omega"][i_q] * arrays["S"][i_q]
    L_E_q = (p.zeta * arrays["Omega"][i_q] * arrays["E"][i_q]
             / p.chi)
    L_x_q = arrays["L_tilde"][i_q] - L_a_q - L_E_q

    log_M_raw = np.concatenate([
        np.zeros(1),
        np.cumsum(0.5 * (arrays["g"][:-1] + arrays["g"][1:])
                  * np.diff(t))])
    flow_raw = log_M_raw + np.log(arrays["c"])
    U0_raw = (np.trapezoid(np.exp(-p.rho * t) * flow_raw, t)
              + np.exp(-p.rho * t[-1])
              * (flow_raw[-1] / p.rho
                 + arrays["g"][-1] / p.rho ** 2))
    U_bar_0 = (p.rho * np.log(stat_0["c"]) + stat_0["g"]) / p.rho ** 2
    ce_raw_pct = 100.0 * np.expm1(p.rho * (U0_raw - U_bar_0))
    ce_selected_pct = 100.0 * record["welfare"]["CE_transition"]

    return {
        "acr_10_percent_trade_cut": {
            "log_percent": float(acr_log_pct),
            "ce_percent": float(acr_ce_pct),
            "steady_state_gain_over_log_acr": float(
                100.0 * est.welfare_gain_ce(eq_0, eq_T, p.rho)
                / acr_log_pct),
            "steady_state_gain_over_ce_acr": float(
                100.0 * est.welfare_gain_ce(eq_0, eq_T, p.rho)
                / acr_ce_pct),
        },
        "autarky": {
            "d": float(d_aut),
            "imports_gdp_percent": float(100.0 * (1.0 - eq_aut["lam_ii"])),
            "growth_percent": float(100.0 * eq_aut["g"]),
            "welfare_loss_from_baseline_percent": float(loss_to_aut_pct),
            "welfare_gain_to_baseline_percent": float(gain_from_aut_pct),
            "acr_log_percent": float(acr_aut_log_pct),
            "loss_over_10_percent_cut_gain": float(
                -loss_to_aut_pct
                / (100.0 * est.welfare_gain_ce(eq_0, eq_T, p.rho))),
        },
        "transition": {
            "ce_selected_percent": float(ce_selected_pct),
            "ce_raw_first_cell_percent": float(ce_raw_pct),
            "raw_minus_selected_percentage_points": float(
                ce_raw_pct - ce_selected_pct),
            "growth_initial_percent": float(100.0 * stat_0["g"]),
            "growth_raw_first_cell_percent": float(100.0 * arrays["g"][0]),
            "growth_first_quarter_percent": float(
                100.0 * arrays["g_reporting"][i_q]),
            "growth_trough_percent": float(
                100.0 * arrays["g_reporting"][i_trough]),
            "growth_trough_year": float(t[i_trough]),
            "growth_terminal_percent": float(100.0 * stat_T["g"]),
            "adoption_flow_initial": float(stat_0["S"]),
            "adoption_flow_terminal": float(stat_T["S"]),
            "domestic_variety_terminal_over_initial": float(
                stat_T["Omega"] / stat_0["Omega"]),
            "first_quarter_labor_change_percentage_points": {
                "adoption": float(100.0 * (L_a_q - L_a_0)),
                "entry": float(100.0 * (L_E_q - L_E_0)),
                "export_fixed_cost": float(100.0 * (L_x_q - L_x_0)),
            },
            "normalized_consumption_change_percent": {
                "impact": float(100.0 * (arrays["c"][0] / stat_0["c"] - 1.0)),
                "first_quarter": float(
                    100.0 * (arrays["c"][i_q] / stat_0["c"] - 1.0)),
            },
        },
        "welfare_decomposition": dec,
        "published_path_aggregation": agg,
    }


# -- \input-ready fragments: complete table environments as the corrigendum
# typesets them, Published panels from data/published_tables.json, Data and
# Corrected numbers computed

def table_params(fit, p, pub):
    x = fit["x"]
    text = "\n".join([
        r"\begin{table}[!ht]",
        r"\refstepcounter{table} \vspace{0.2cm}",
        r"\label{ta:params}",
        r"\footnotesize",
        r"\setlength {\tabcolsep}{3.5mm}",
        r"\renewcommand{\arraystretch}{2.0}",
        r"\begin{center}",
        r"\begin{tabular}{l c c}",
        r"\multicolumn{3}{c}{\textbf{\normalsize Table \ref{ta:params}: "
        r"Calibration: Parameters and Values, Published and Corrected}}\\",
        r"\hline",
        r"\hline",
        r"Parameter & Published & Corrected \\",
        r"\hline",
        rf"Technology Adoption Cost, $\zeta$ & {pub['zeta']} & "
        rf"{p.zeta:.1f} \\",
        rf"Number of Countries, $N$ & {pub['N']} & {p.N:.0f} \\",
        r"\hline",
        rf"Discount Rate, $\rho$ & {pub['rho']} & {fit['rho']:.4f} \\",
        rf"Pareto Shape Parameter, $\theta$ & {pub['theta']} & "
        rf"{fit['theta']:.2f} \\",
        rf"Variety Elasticity of Substitution, $\sigma$ & {pub['sigma']} & "
        rf"{fit['sigma']:.2f} \\",
        rf"Drift of GBM Process $\mu$ & {pub['mu']} & {x['mu']:.3f} \\",
        rf"Std.\ Deviation of GBM Process $\upsilon$ & {pub['upsilon']} & "
        rf"{x['upsilon']:.3f} \\",
        rf"Death Rate of Firms $\delta$ & {pub['delta']} & "
        rf"{fit['delta']:.3f} \\",
        rf"Iceberg Trade Cost, $d$ & {pub['d']} & {x['d']:.2f} \\",
        rf"Export Fixed Cost, $\kappa$ & {pub['kappa']} & "
        rf"{x['kappa']:.3f} \\",
        rf"Entry Cost Relative to Adoption Cost $1/\chi$ & "
        rf"{pub['inv_chi']} & {x['inv_chi']:.2f} \\",
        r"\hline",
        r"\end{tabular}",
        r"\\[0.75ex]",
        r"\parbox{5.8in}{\footnotesize \textbf{Note:} The Published column "
        r"reproduces the published calibration table. In the Corrected "
        r"column, $\theta$ and $\sigma$ are held at their published "
        r"AER estimates (displayed rounded), and $(d, \kappa, "
        r"1/\chi, \mu, \upsilon)$ are estimated by SMM on the same targets "
        r"and weights as the published paper. The GBM parameter is reported "
        r"as a standard deviation in both columns, correcting the variance "
        r"label in the published table.}",
        r"\end{center}",
        r"\end{table}",
        ""])
    printed = {k: f"{x[k]:.6g}" for k in est.X_NAMES} | {
        "rho": f"{fit['rho']:.6g}", "theta": fit["theta"],
        "sigma": fit["sigma"], "delta": fit["delta"]}
    return text, printed


def table_moments(targets, rho, mom, pub):
    data = dict(r=100.0 * (rho + targets[0]), g=100.0 * targets[0],
                imp=100.0 * (1.0 - targets[1]), frac=100.0 * targets[2],
                rel=targets[3])
    corr = dict(r=100.0 * (rho + mom[0]), g=100.0 * mom[0],
                imp=100.0 * (1.0 - mom[1]), frac=100.0 * mom[2], rel=mom[3])
    text = "\n".join([
        r"\begin{table}[!ht]",
        r"\refstepcounter{table}",
        r"\label{ta:moments}",
        r"\footnotesize",
        r"\setlength {\tabcolsep}{3mm}",
        r"\renewcommand{\arraystretch}{2.25}",
        r"\begin{center}",
        r"\begin{tabular}{l l l l}",
        r"\multicolumn{4}{c}{\textbf{\normalsize Table \ref{ta:moments}: "
        r"Aggregate Moments: Data, Published Model, and Corrected Model}}\\",
        r"\hline",
        r"\hline",
        r"Moment Description & Data & Published & Corrected\\",
        r"\hline",
        rf"U.S. Real Interest Rate & {data['r']:.2f} & {pub['r']} & "
        rf"{corr['r']:.2f} \\",
        rf"U.S. Productivity Growth & {data['g']:.2f} & {pub['g']} & "
        rf"{corr['g']:.2f} \\",
        rf"U.S. Import/GDP & {data['imp']:.2f} & {pub['imp']} & "
        rf"{corr['imp']:.2f} \\",
        rf"Share of Exporting Establishments & {data['frac']:.1f} & "
        rf"{pub['frac']} & {corr['frac']:.1f} \\",
        r"Relative Size of Exporting Establishments & "
        rf"{data['rel']:.1f} & {pub['rel']} & {corr['rel']:.1f} \\",
        r"\hline",
        r"\end{tabular}",
        r"\\[0.75ex]",
        r"\parbox{4.4in}{\footnotesize Moment construction is as in the "
        r"published paper. The real interest rate, productivity growth "
        r"rate, and import/GDP ratio are in percent and averages over the "
        r"1977--2000 time period. The Published column reproduces the "
        r"published paper's model column.}",
        r"\end{center}",
        r"\end{table}",
        ""])
    return text, dict(data=data, corrected=corr)


def table_persistence(mom, entry, delta, pub):
    # published orientation reverses the code quartiles: row/col 1 = largest;
    # the table's top two rows are code quartiles 4 and 3, columns reversed
    row1 = 100.0 * mom[8:12][::-1]
    row2 = 100.0 * mom[4:8][::-1]
    # the Data block and the correlation use the raw firm moments, NOT the
    # additively renormalized SMM targets
    firm_raw = np.loadtxt(os.path.join(DATA, "firm_moments.csv"),
                          delimiter=",")
    d1, d2 = 100.0 * firm_raw[1][::-1], 100.0 * firm_raw[0][::-1]
    corr_dc = np.corrcoef(np.concatenate([d1, d2]),
                          np.concatenate([row1, row2]))[0, 1]
    p1, p2 = pub["rows"]
    text = "\n".join([
        r"\begin{table}[!ht]",
        r"\footnotesize",
        r"\refstepcounter{table}",
        r"\renewcommand{\arraystretch}{2.05}",
        r"\setlength {\tabcolsep}{10mm}",
        r"\begin{center}",
        r"\label{tb:persistence}",
        r"\begin{tabular}[t]{c c}",
        r"\multicolumn{2}{c}{{\normalsize\textbf{Table "
        r"\ref{tb:persistence}: Establishment Size Dynamics, Data, "
        r"Corrected and Published Models}}}",
        r"\\",
        r"\hline",
        r"\hline",
        r" \multicolumn{2}{c}{\textbf{Transition Matrix of Relative Size: "
        r"largest, quartile 1; smallest, quartile 4}}\\",
        r"Data & Corrected Model \\",
        r"\cmidrule(r){1-1} \cmidrule(l){2-2}",
        r"$  \begin{array}{c | cccc}",
        r" &  1   &   2&  3&  4 \\",
        r"\hline",
        rf"1 & {d1[0]:.1f} & {d1[1]:.1f} & {d1[2]:.1f} & {d1[3]:.1f} "
        r"\vspace{-.05cm}\\",
        rf"2 & {d2[0]:.1f} & {d2[1]:.1f} & {d2[2]:.1f} & {d2[3]:.1f} "
        r"\vspace{-.05cm} \\",
        r"\end{array}",
        r"$ & \footnotesize $",
        r"\begin{array}{c | cccc}",
        r" &  1   &   2&  3&  4 \\",
        r"\hline",
        rf"1 & {row1[0]:.1f} & {row1[1]:.1f} & {row1[2]:.1f} & "
        rf"{row1[3]:.1f} \vspace{{-.05cm}}\\",
        rf"2 & {row2[0]:.1f} & {row2[1]:.1f} & {row2[2]:.1f} & "
        rf"{row2[3]:.1f} \vspace{{-.05cm}} \\",
        r"\end{array}",
        r"$ \vspace{.25cm}",
        r"\\",
        rf"Corr(Data, Corrected Model) = {corr_dc:.2f} \\",
        r"\cmidrule(l){2-2}",
        r" & Published Model \\",
        r"\cmidrule(l){2-2}",
        r" & \footnotesize $",
        r"\begin{array}{c | cccc}",
        r" &  1   &   2&  3&  4 \\",
        r"\hline",
        rf"1 & {p1[0]} & {p1[1]} & {p1[2]} & {p1[3]} \vspace{{-.05cm}}\\",
        rf"2 & {p2[0]} & {p2[1]} & {p2[2]} & {p2[3]} \vspace{{-.05cm}} \\",
        r"\end{array}",
        r"$ \vspace{.25cm}",
        r"\\",
        rf"Corr(Data, Published Model) = {pub['corr']} \\",
        r"\hline",
        r"\multicolumn{2}{c}{\textbf{Employment Share of New "
        r"Establishments}}\\",
        rf"Data: \ \ \ \ \ \ {entry:.2f} & Corrected Model: "
        rf"\ \ \ \ \ \ {delta:.2f} \\",
        r"\hline",
        r"\end{tabular}",
        r"\\[0.75ex]",
        r"\parbox{5.0in}{\footnotesize  \textbf{Note:} In each panel, rows "
        r"represent the establishment-size quartile in period $t$; columns "
        r"represent the establishment-size quartile in period $t+5$. Data "
        r"source and construction as in the published paper; the Published "
        r"Model panel reproduces the published paper's model panel.}",
        r"\end{center}",
        r"\end{table}",
        ""])
    printed = dict(row1=np.round(row1, 1).tolist(),
                   row2=np.round(row2, 1).tolist(),
                   corr=f"{corr_dc:.2f}", employment=f"{delta:.2f}")
    return text, printed


def table_results(eq_0, eq_T, ce_ss_pct, ce_transition_pct, agg, pub):
    vals = dict(g_0=100.0 * float(eq_0["g"]), g_T=100.0 * float(eq_T["g"]),
                imp_0=100.0 * (1.0 - float(eq_0["lam_ii"])),
                imp_T=100.0 * (1.0 - float(eq_T["lam_ii"])),
                ce_transition=ce_transition_pct, ce_ss=ce_ss_pct,
                note_pub=agg["ce_pub_pct"], note_corr=agg["ce_corr_pct"])
    text = "\n".join([
        r"\begin{table}[!ht]",
        r"\refstepcounter{table}",
        r"\label{ta:results}",
        r"\setlength {\tabcolsep}{4.5mm}",
        r"\vspace{0.1cm}",
        r"\renewcommand{\arraystretch}{2.0}",
        r"\begin{center}",
        r"\begin{tabular}{l c c c c}",
        r"\multicolumn{5}{c}{\textbf{Table \ref{ta:results}: 10 Percent "
        r"Reduction in Trade Costs: Growth, Trade, and Welfare}}\\",
        r"\hline",
        r"\hline",
        r"& \multicolumn{2}{c}{\small Published} & "
        r"\multicolumn{2}{c}{\small Corrected} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"& \small Baseline BGP & \small New BGP & \small Baseline BGP & "
        r"\small New BGP \\",
        r"\hline",
        rf"\small Growth     & {pub['g_0']} & {pub['g_T']} & "
        rf"{vals['g_0']:.2f} & {vals['g_T']:.2f} \\",
        rf"\small Imports/GDP & {pub['imp_0']} & {pub['imp_T']} & "
        rf"{vals['imp_0']:.1f} & {vals['imp_T']:.1f} \\",
        r"\hline",
        r"\multicolumn{5}{l}{\small Welfare} \\",
        rf"\small Transition Path: & "
        rf"\multicolumn{{2}}{{c}}{{{pub['ce_transition']}}} & "
        rf"\multicolumn{{2}}{{c}}{{{vals['ce_transition']:.1f}}} \\",
        rf"\small SS to SS: & \multicolumn{{2}}{{c}}{{{pub['ce_ss']}}} & "
        rf"\multicolumn{{2}}{{c}}{{{vals['ce_ss']:.1f}}} \\",
        r"\hline",
        r"\end{tabular}",
        r"\\[0.75ex]",
        r"\parbox{5.9in}{\footnotesize  \textbf{Note:} All values are in "
        r"percent. Consumption-equivalent is the permanent percent increase "
        r"in consumption a household requires in the old regime to be "
        r"indifferent between the new and old regimes. The Published "
        r"columns reproduce the published paper's Table 5 values at the "
        r"published calibration, computed under the published package's "
        r"welfare aggregation, whose terminal continuation substitutes "
        r"$T\,g(T)$ for the accumulated $\log M(T)$; the same published "
        r"path under the corrected aggregation gives a transition value of "
        rf"${vals['note_corr']:.1f}$ rather than ${vals['note_pub']:.1f}$ "
        r"percent (details in the numerical appendix). The Corrected "
        r"columns use the corrected model, calibration, and welfare "
        r"aggregation, so part of the difference between the transition "
        r"entries reflects the aggregation correction rather than the "
        r"corrected economics.}",
        r"\end{center}",
        r"\end{table}",
        ""])
    return text, {k: f"{v:.4f}" for k, v in vals.items()}


def decomposition_equation(shares):
    def share_str(v):
        # the export-threshold share is the envelope zero: printed as 0
        return "0" if abs(v) < 5e-3 else f"{v:.2f}"
    s = {k: share_str(v) for k, v in shares.items()}
    text = "\n".join([
        r"\begin{align}",
        r"\underbrace{\frac{\diff \bar{U}(c,g)}{\diff d}}_{\textbf{100\%}} "
        r"\ \ &= \ \ \underbrace{\bar{U}_{1} \frac{\partial f_c}{\partial "
        r"d}}_{\textbf{" + s["direct consumption"] + r"\%}} \ \ + \ \ "
        r"\underbrace{\bar{U}_{1} \frac{\partial f_c}{\partial \Omega} "
        r"\frac{\diff f_{\Omega} }{\diff d}}_{\textbf{"
        + s["variety"] + r"\%}}",
        r"\ \ + \ \ \underbrace{\bar{U}_{1} \frac{\partial f_c}{\partial "
        r"\hat{z}} \frac{\diff f_{\hat{z}} }{\diff d}}_{\textbf{"
        + s["export threshold"] + r"\%}} \ \ + \ \ "
        r"\underbrace{\bar{U}_{1} \frac{\partial f_c}{\partial g} "
        r"\frac{\diff f_{g} }{\diff d}}_{\textbf{"
        + s["growth via consumption"] + r"\%}} \ \ + \ \ "
        r"\underbrace{\bar{U}_{2} \frac{\diff f_g}{\diff d} }_{\textbf{"
        + s["direct growth"] + r"\%}}.",
        r"\label{eq:quant_decomposition}",
        r"\end{align}",
        ""])
    return text, s


def inline_macros(inline, fit, mom, eq_0, eq_T, ce_ss_pct, ce_ss_grid_pct,
                  corr_dc, p):
    # \newcommand's for every corrected-model number in the corrigendum's
    # running prose, so it sources them from this computation rather than by
    # hand.  Published-model comparison numbers and modeling inputs stay
    # literals in the .tex.  Rounding matches the prose each macro replaces.
    x = fit["x"]
    acr = inline["acr_10_percent_trade_cut"]
    aut = inline["autarky"]
    tr = inline["transition"]
    dec = inline["welfare_decomposition"]
    agg = inline["published_path_aggregation"]
    lab = tr["first_quarter_labor_change_percentage_points"]
    cons = tr["normalized_consumption_change_percent"]

    # option-value coefficient at the calibrated BGP, in scientific -> LaTeX
    c2 = float(eq_0["c2t"] * eq_0["z_hat"] ** (-eq_0["beta"]))
    mant, exp = f"{c2:.1e}".split("e")
    c2_tex = rf"{mant} \times 10^{{{int(exp)}}}"

    vals = {
        # --- welfare, gains, ACR (percent) ---
        r"\CEss": f"{ce_ss_pct:.1f}",
        r"\CEtransition": f"{tr['ce_selected_percent']:.1f}",
        r"\CEssGrid": f"{ce_ss_grid_pct:.3f}",
        r"\CEtransitionFull": f"{tr['ce_selected_percent']:.3f}",
        r"\CEpubReagg": f"{agg['ce_corr_pct']:.1f}",
        r"\ReaggGap": f"{agg['ce_pub_pct'] - agg['ce_corr_pct']:.1f}",
        r"\GainAutarky": f"{aut['welfare_gain_to_baseline_percent']:.1f}",
        r"\LossAutarky": f"{-aut['welfare_loss_from_baseline_percent']:.1f}",
        r"\LossOverGain": f"{aut['loss_over_10_percent_cut_gain']:.1f}",
        r"\ACRcorr": f"{acr['ce_percent']:.2f}",
        r"\ACRAutarky": f"{aut['acr_log_percent']:.1f}",
        r"\GainTimesACR": f"{acr['steady_state_gain_over_ce_acr']:.1f}",
        r"\ImportsAutarky": f"{aut['imports_gdp_percent']:.1f}",
        # --- transition growth / adoption / consumption / labor ---
        r"\GrowthChangeCorr": f"{100.0 * float(eq_T['g'] - eq_0['g']):.2f}",
        r"\GrowthFirstQuarter": f"{tr['growth_first_quarter_percent']:.2f}",
        r"\GrowthTrough": f"{tr['growth_trough_percent']:.2f}",
        r"\GrowthTroughYear": f"{tr['growth_trough_year']:.1f}",
        r"\GrowthTerminal": f"{tr['growth_terminal_percent']:.2f}",
        r"\AdoptInit": f"{tr['adoption_flow_initial']:.3f}",
        r"\AdoptTerminal": f"{tr['adoption_flow_terminal']:.3f}",
        r"\ConsImpact": f"{cons['impact']:.1f}",
        r"\ConsFirstQuarter": f"{cons['first_quarter']:.1f}",
        r"\RawMinusSelected":
            f"{tr['raw_minus_selected_percentage_points']:.3f}",
        r"\VarietyTerminalPct":
            f"{100.0 * tr['domestic_variety_terminal_over_initial']:.1f}",
        r"\LaborEntry": f"{-lab['entry']:.2f}",
        r"\LaborExport": f"{lab['export_fixed_cost']:.2f}",
        r"\LaborAdoption": f"{lab['adoption']:.2f}",
        # --- moments and calibrated estimates ---
        r"\FracExpCorr": f"{100.0 * float(mom[2]):.1f}",
        r"\RelSizeCorr": f"{float(mom[3]):.1f}",
        r"\CorrCorr": corr_dc,
        r"\dCorr": f"{x['d']:.2f}",
        r"\kappaCorr": f"{x['kappa']:.3f}",
        r"\invChiCorr": f"{x['inv_chi']:.2f}",
        r"\muCorr": f"{x['mu']:.3f}",
        r"\upsCorr": f"{x['upsilon']:.3f}",
        # --- welfare decomposition ---
        r"\DirectGrowthShareCorr": f"{dec['shares']['direct growth']:.1f}",
        r"\InefficiencyCorr": f"{dec['inefficiency']:.1f}",
        r"\FreeGrowthBench": f"{dec['free_growth_benchmark']:.1f}",
        r"\DirectConsOverACR":
            f"{dec['direct_consumption_over_acr_percent']:.1f}",
        r"\SemiElasticityCorr": f"{dec['semi_elasticity']:.3f}",
        # --- correction-section constants ---
        r"\CtwoApprox": c2_tex,
        r"\UpsSq": f"{p.upsilon ** 2:.4f}",
        r"\GmuOverTheta": f"{(float(eq_0['g']) - p.mu) / p.theta:.4f}",
    }
    text = "\n".join([rf"\newcommand{{{name}}}{{{v}}}"
                      for name, v in vals.items()] + [""])
    return text, vals


def main(estimation: str =
         "output/estimation_theta4.98898_sigma3.16692.json",
         transition: str = "output/transition_results.json",
         out_dir: str = "output"):
    alg_set = AlgorithmSettings()
    p = est.load_estimated_parameters(estimation)
    with open(estimation, encoding="utf-8") as fh:
        fit = json.load(fh)
    targets, rho, entry = est.load_targets()
    targets = np.asarray(targets)

    dec = decompose(p.d_0, p, alg_set)
    agg = published_reaggregation(rho, p.delta)
    with open(os.path.join(DATA, "published_tables.json"),
              encoding="utf-8") as fh:
        pub = json.load(fh)

    x = jnp.array([fit["x"][name] for name in est.X_NAMES])
    mom, eq_0, _ = est.model_moments(x, p, alg_set)
    mom = np.asarray(mom)
    if not np.all(np.isfinite(mom)):
        raise RuntimeError("model moments at the fit are not finite")
    eq_T = est.solve_bgp(p.d_T, p, alg_set)
    if not bool(eq_T["ok"]):
        raise RuntimeError("10% cut BGP solve failed")
    ce_ss_pct = 100.0 * float(est.welfare_gain_ce(eq_0, eq_T, p.rho))
    if not os.path.exists(transition):
        raise FileNotFoundError(f"no transition artifact at {transition}: "
                                "run replication_results.py first")
    transition_record, transition_arrays = load_transition_artifact(
        transition, estimation, p, alg_set, load_arrays=True)
    ce_transition_pct = (100.0
                         * transition_record["welfare"]["CE_transition"])

    inline = inline_results(eq_0, eq_T, p, alg_set, transition_record,
                            transition_arrays, dec, agg)
    fragments = {
        "table_params.tex": table_params(fit, p, pub["ta:params"]),
        "table_moments.tex": table_moments(targets, rho, mom,
                                           pub["ta:moments"]),
        "table_persistence.tex": table_persistence(mom, entry, p.delta,
                                                   pub["tb:persistence"]),
        "table_results.tex": table_results(eq_0, eq_T, ce_ss_pct,
                                           ce_transition_pct, agg,
                                           pub["ta:results"]),
        "decomposition_equation.tex": decomposition_equation(dec["shares"]),
    }
    corr_dc = fragments["table_persistence.tex"][1]["corr"]
    ce_ss_grid_pct = 100.0 * transition_record["welfare"]["CE_ss_to_ss"]
    fragments["inline_macros.tex"] = inline_macros(
        inline, fit, mom, eq_0, eq_T, ce_ss_pct, ce_ss_grid_pct, corr_dc, p)
    os.makedirs(out_dir, exist_ok=True)
    for name, (text, printed) in fragments.items():
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"{name}  {printed}")
    inline_path = os.path.join(out_dir, "inline_results.json")
    with open(inline_path, "w", encoding="utf-8") as fh:
        json.dump(inline, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"inline_results.json  {inline['transition']}")


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(main)
