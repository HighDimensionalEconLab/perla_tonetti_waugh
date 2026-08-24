# The corrigendum's eight figures and an auxiliary BGP counterfactual sweep:
# stationary density, the 10%-cut transition paths, and chi robustness sweeps,
# all at the accepted SMM fit
# loaded from the estimation JSON.  Corrected curves are solved here; the
# "Published" overlays come from the frozen artifacts in data/.
# Writes output/{corr,ovl}_*.pdf.

import os

from utilities import configure_process, load_transition_artifact

configure_process()

import dataclasses

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

import estimation as est
from model import AlgorithmSettings, z_grid

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

PRE_TIMES = np.array([-20.0, -15.0, -10.0, -5.0, -1.0])
# Labor and consumption inherit an unresolved initial layer at the shock.
# Those series are not drawn on [0, T_LAYER); average firm-productivity growth
# uses the same reporting window for comparison across the transition figures.
T_LAYER = 0.25

# scale on 1/chi, label, line color (None = matplotlib default)
CHI_SCALES = ((1.0, "Calibrated", None), (0.9, "Large", "red"),
              (1.1, "Small", "k"))

PUB = dict(color="k", ls="--", lw=3, alpha=0.6, label="Published")

# the frozen CSV's exact header and the ASCII names bound to it by position
PUB_TRANSITION_HEADER = ("t,g,z_hat,Ω,E,v_1,L_tilde,entry_residual,λ_ii,S,"
                         "z_bar,c,π_min,log_M,U,π_rat,L_tilde_a,L_tilde_x,"
                         "L_tilde_E,w,r")
PUB_TRANSITION_COLS = ("t", "g", "z_hat", "Omega", "E", "v_1", "L_tilde",
                       "entry_residual", "lambda_ii", "S", "z_bar", "c",
                       "pi_min", "log_M", "U", "pi_rat", "L_tilde_a",
                       "L_tilde_x", "L_tilde_E", "w", "r")


def load_published_transition():
    # the original replication package's cached solution of the published
    # model: t-sorted, pre-shock rows at t ∈ {−20, −15, −10, −5, −1} (so row
    # 4 is the t = −1 normalization row), labor components precomputed
    path = os.path.join(DATA, "published_transition.csv")
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip()
    if header != PUB_TRANSITION_HEADER:
        raise RuntimeError(f"unexpected header in {path}: {header}")
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    return dict(zip(PUB_TRANSITION_COLS, raw.T))


def corr_style(color):
    return dict(color=color, ls="-", lw=5, alpha=0.78, label="Corrected")


def strip(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, path, summary):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"{os.path.basename(path)}  [{summary}]")


def transition_figures(out, pub, p, alg_set, out_dir):
    stat_0 = out["stationary_0"]
    g_0 = float(stat_0["g"])
    t = np.asarray(out["t"])
    t_full = np.concatenate([PRE_TIMES, t])
    corner = (t_full >= 0.0) & (t_full < T_LAYER)

    def with_pre(series, ss_value):
        # transition path with stationary pre-shock rows prepended
        return np.concatenate([np.full(PRE_TIMES.size, float(ss_value)),
                               np.asarray(series)])

    def no_corner(series):
        return np.where(corner, np.nan, series)

    # Average firm productivity is M(t) E_t[z].  For the corrected KFE,
    # its instantaneous growth rate is
    # mu + upsilon^2/2 + S(t) [1 - 1/E_t[z]].  The published normalized
    # Pareto shape is fixed, so its average-productivity growth equals g(t).
    x = np.asarray(z_grid(alg_set))[1:]
    f = np.asarray(out["f"])
    if f.shape != (t.size, x.size):
        raise RuntimeError("transition density and productivity grid disagree")
    mass_error = float(np.max(np.abs(f.sum(axis=1) - 1.0)))
    if mass_error >= alg_set.mass_err_tol:
        raise RuntimeError(f"transition density mass error is {mass_error:g}")
    mean_relative_productivity = f @ np.exp(x)
    if (not np.all(np.isfinite(mean_relative_productivity))
            or np.any(mean_relative_productivity <= 1.0)):
        raise RuntimeError("invalid mean productivity relative to the threshold")
    mean_growth = (p.mu + 0.5 * p.upsilon ** 2
                   + np.asarray(out["S"])
                   * (1.0 - 1.0 / mean_relative_productivity))
    if not np.all(np.isfinite(mean_growth)):
        raise RuntimeError("nonfinite average firm-productivity growth")
    if not np.isclose(mean_growth[0], g_0, rtol=0.0, atol=5e-7):
        raise RuntimeError("initial average-productivity growth fails BGP check")
    if not np.isclose(mean_growth[-1], float(out["stationary_T"]["g"]),
                      rtol=0.0, atol=5e-7):
        raise RuntimeError("terminal average-productivity growth fails BGP check")

    lam = with_pre(out["lambda_ii"], stat_0["lambda_ii"])
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(t_full, 100.0 * (1.0 - lam), **corr_style("blue"))
    ax.plot(pub["t"], 100.0 * (1.0 - pub["lambda_ii"]), **PUB)
    strip(ax)
    ax.set_ylabel("Imports/GDP", fontsize=14)
    ax.set_xlim(-20, 50)
    ax.set_ylim(9, 15)
    ax.legend(fontsize=14, frameon=False)
    save(fig, os.path.join(out_dir, "ovl_trade.pdf"),
         f"imports/GDP {100 * (1 - lam[0]):.2f}% -> "
         f"{100 * (1 - lam[-1]):.2f}%")

    Omega = with_pre(out["Omega"], stat_0["Omega"])
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(t_full, Omega / Omega[0], **corr_style("red"))
    ax.plot(pub["t"], pub["Omega"] / pub["Omega"][0], **PUB)
    strip(ax)
    ax.set_ylabel("Domestic Variety \n Normalized, Initial S.S. = 1",
                  fontsize=14)
    ax.set_xlim(-20, 50)
    ax.set_ylim(0.94, 1.04)
    ax.legend(fontsize=14, frameon=False)
    save(fig, os.path.join(out_dir, "ovl_omega.pdf"),
         f"Omega_T/Omega_0 = {Omega[-1] / Omega[0]:.4f}")

    # labor outside production L̃ = Ω[(N−1)(1−F(ẑ))κ + ζ(S + E/χ)], split as
    # adoption ζΩS, entry ζΩE/χ, exporter = remainder
    L_tilde = with_pre(out["L_tilde"], stat_0["L_tilde"])
    L_a = p.zeta * Omega * with_pre(out["S"], stat_0["S"])
    L_E = p.zeta * Omega * with_pre(out["E"], p.delta) / p.chi
    L_x = L_tilde - L_a - L_E
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.tight_layout(pad=6)
    fig.text(-0.01, 0.5, "Breakdown of Labor Outside of Production,\n "
             "Percentage Point Change relative to SS",
             va="center", rotation="vertical", fontsize=14)
    panels = (((0, 0), L_x, "L_tilde_x", "black", "-.", 0.70,
               "Exporter Costs", (-1.0, 1.0)),
              ((0, 1), L_a, "L_tilde_a", "black", "-", 0.70,
               "Adoption Costs", (-1.0, 1.0)),
              ((1, 0), L_E, "L_tilde_E", "blue", "-", 0.70,
               "Entry Costs", (-2.5, 1.0)),
              ((1, 1), L_tilde, "L_tilde", "red", "-", 0.80,
               "Total Labor Outside of Production", (-1.5, 1.0)))
    for pos, series, pub_key, color, ls, alpha, title, ylim in panels:
        ax = axes[pos]
        ax.plot(t_full, no_corner(100.0 * (series - series[0])), color=color,
                lw=5, ls=ls, alpha=alpha, label="Corrected")
        ax.plot(pub["t"], 100.0 * (pub[pub_key] - pub[pub_key][0]), **PUB)
        strip(ax)
        ax.set_xlim(-20, 50)
        ax.set_ylim(*ylim)
        ax.set_title(title)
    axes[0, 0].legend(fontsize=11, frameon=False)
    save(fig, os.path.join(out_dir, "ovl_fixed_costs.pdf"),
         f"dL_tilde = {100.0 * (L_tilde[-1] - L_tilde[0]):+.2f}pp")

    # selected log M from the solution; pre-period slope g₀ with M(0) = 1
    log_M = np.concatenate([PRE_TIMES * g_0, np.asarray(out["log_M"])])
    i_ref = PRE_TIMES.size - 1   # the t = −1 row: normalization point
    log_C = np.log(with_pre(out["c"], stat_0["c"])) + log_M
    log_C = log_C - log_C[i_ref] + 1.0
    mean_relative_full = with_pre(mean_relative_productivity,
                                  mean_relative_productivity[0])
    log_prod = log_M + np.log(mean_relative_full)
    log_prod = log_prod - log_prod[i_ref] + 1.0
    log_C_pub = np.log(pub["c"]) + pub["log_M"]
    log_C_pub = log_C_pub - log_C_pub[i_ref] + 1.0
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(t_full, no_corner(log_C), color="red", lw=5,
            label="Log Consumption (Corrected)")
    ax.plot(pub["t"], log_C_pub,
            **dict(PUB, label="Log Consumption (Published)"))
    ax.plot(t_full, log_prod, color="k", ls="-.", alpha=0.5, lw=3,
            label="Log Average Firm Productivity (Corrected)")
    strip(ax)
    ax.set_ylabel("Log Consumption & Average Firm Productivity\n"
                  "Normalized, t-1 = 1", fontsize=14)
    ax.set_xlim(-20, 50)
    ax.legend(fontsize=12, frameon=False, loc="upper left")
    axins = zoomed_inset_axes(ax, 1.5, loc=4, borderpad=2)
    axins.plot(t_full, no_corner(log_C), color="red", lw=5)
    axins.plot(pub["t"], log_C_pub, **PUB)
    axins.plot(t_full, log_prod, color="k", ls="-.", alpha=0.5, lw=3)
    axins.set_xlim(-5, 5)
    axins.set_ylim(0.94, 1.10)
    mark_inset(ax, axins, loc1=3, loc2=2, alpha=0.15)
    strip(axins)
    save(fig, os.path.join(out_dir, "ovl_log_consumption_inset.pdf"),
         f"log C(T) = {log_C[-1]:.3f}")

    mean_growth_full = with_pre(mean_growth, g_0)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(t_full, no_corner(100.0 * mean_growth_full), **corr_style("red"))
    ax.plot(pub["t"], 100.0 * pub["g"], **PUB)
    strip(ax)
    ax.set_ylabel("Average Firm Productivity Growth\nPercent per Year",
                  fontsize=14)
    ax.set_xlim(-20, 50)
    ax.set_ylim(0.6, 1.10)
    ax.legend(fontsize=14, frameon=False)
    save(fig, os.path.join(out_dir, "ovl_growth.pdf"),
         f"average firm productivity growth {100 * g_0:.2f}% -> "
         f"{100 * mean_growth[-1]:.2f}%")


def chi_sweep(kind, scales, p, alg_set):
    # for each chi variant: anchor BGP pair (d_0, d_T) at scale 1, then march
    # outward warm-starting each solve from its neighbor; the first rejected
    # solve (inadmissible root or failed continuation -- how nonexistence
    # manifests at the boundary) leaves NaN from there outward
    data = {}
    for cs, name, _ in CHI_SCALES:
        p_chi = dataclasses.replace(p, chi=p.chi / cs)
        eq_0 = est.solve_bgp(p.d_0, p_chi, alg_set)
        eq_T = est.solve_bgp(p.d_T, p_chi, alg_set)
        if not (bool(eq_0["ok"]) and bool(eq_T["ok"])):
            raise RuntimeError(f"{kind} sweep: anchor BGP pair failed at "
                               f"chi scale {cs}")
        dg = np.full(scales.size, np.nan)
        g0 = np.full(scales.size, np.nan)
        wf = np.full(scales.size, np.nan)
        below = np.flatnonzero(scales < 1.0)[::-1]
        above = np.flatnonzero(scales >= 1.0)
        stops = []
        for march_indices in (below, above):
            phi_0, phi_T = eq_0["phi"], eq_T["phi"]
            for i in march_indices:
                s = scales[i]
                if kind == "gbm":
                    p_s = dataclasses.replace(p_chi, mu=p.mu * s,
                                              upsilon=p.upsilon * s)
                else:
                    p_s = dataclasses.replace(p_chi, delta=p.delta * s)
                out_0 = est.solve_bgp(p.d_0, p_s, alg_set, phi_0=phi_0)
                out_T = est.solve_bgp(p.d_T, p_s, alg_set, phi_0=phi_T)
                if not (out_0["ok"] and out_T["ok"]):
                    # name the admissibility conditions lost at the boundary
                    for tag, eq_s in (("d_0", out_0), ("d_T", out_T)):
                        bad = [key for key, val in eq_s["ok_flags"].items()
                               if not bool(val)]
                        if bad:
                            stops.append(f"scale={s:g} {tag}: "
                                         + ",".join(bad))
                    break
                dg[i] = 100.0 * float(out_T["g"] - out_0["g"])
                g0[i] = 100.0 * float(out_0["g"])
                wf[i] = 100.0 * float(est.welfare_gain_ce(out_0, out_T,
                                                          p.rho))
                phi_0, phi_T = out_0["phi"], out_T["phi"]
        print(f"  {kind} chi_scale={cs:g} ({name}): "
              f"{int(np.isfinite(dg).sum())}/{scales.size} points feasible"
              + ("; stopped at " + "; ".join(stops) if stops else ""))
        data[cs] = dict(dg=dg, g0=g0, welfare=wf)
    return data


def chi_figure(data, kind, scales, path, p, base):
    # 2x2 layout, three active panels plus a legend panel, with dashed
    # crosshairs at the calibrated point
    if kind == "gbm":
        x, x_base = p.upsilon * scales, p.upsilon
        xlab = "\n GBM Std. Deviation (Drift Scaled Proportionally)"
        xlim = (0.004, 0.07)
        xticks = np.arange(0.01, 0.071, 0.01)
        ylims = dict(dg=(0.05, 0.30), g0=(0.0, 3.30), welfare=(4, 13))
    else:
        x, x_base = p.delta * scales, p.delta
        xlab = "\n Exit Shock Parameter"
        xlim = (0.01, 0.04)
        xticks = (0.01, 0.02, 0.03, 0.04)
        ylims = dict(dg=(0.05, 0.60), g0=(0.0, 3.6), welfare=(2, 28))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.tight_layout(pad=6)

    def panel(pos, field, ylab, xlabel=None, with_labels=False):
        ax = axes[pos]
        for cs, name, color in CHI_SCALES:
            kwargs = dict(lw=4)
            if color is not None:
                kwargs["color"] = color
            if with_labels:
                kwargs["label"] = f"{name} $\\chi$ = {p.chi / cs:.3f}"
            ax.plot(x, data[cs][field], **kwargs)
        ax.set_xticks(xticks)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylims[field])
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylab, fontsize=12)
        strip(ax)
        ax.vlines(x_base, ylims[field][0], base[field], color="k", ls="--",
                  lw=3)
        ax.hlines(base[field], xlim[0], x_base, color="k", ls="--", lw=3,
                  label="Calibrated Values")
        return ax

    panel((0, 0), "dg",
          "\n Change in Productivity Growth \n Percentage Points",
          xlabel=xlab)
    ax = panel((0, 1), "g0", "\n Initial SS Productivity Growth, Percent",
               xlabel=xlab, with_labels=True)
    ax.legend(bbox_to_anchor=(0.0, -1.25, 1.0, 0.102), frameon=False,
              fontsize=14, loc=4)
    panel((1, 0), "welfare", "\n Welfare Gain, Percent", xlabel=xlab)
    axes[1, 1].axis("off")
    save(fig, path, f"crosshair dg={base['dg']:.3f}pp g0={base['g0']:.3f}% "
                    f"welfare={base['welfare']:.2f}%")


# displayed x-windows of chi_figure, so the prose ranges are read off exactly
# the plotted curves
FIG_XLIM = {"gbm": (0.004, 0.07), "delta": (0.01, 0.04)}


def write_figure_macros(sweeps, scales, p, out_dir):
    # \newcommand's for the firm-dynamics figure ranges quoted in prose:
    # min/max of each curve over its displayed x-window
    def bounds(kind, cs, field):
        x = (p.upsilon if kind == "gbm" else p.delta) * scales
        lo_x, hi_x = FIG_XLIM[kind]
        v = np.asarray(sweeps[kind][cs][field])
        m = (x >= lo_x) & (x <= hi_x) & np.isfinite(v)
        return float(v[m].min()), float(v[m].max())

    def pair(kind, cs, field, fmt):
        lo, hi = bounds(kind, cs, field)
        return rf"${format(lo, fmt)}$--${format(hi, fmt)}$"

    dlo, dhi = bounds("delta", 1.0, "dg")
    vals = {
        r"\DgRangeLargeChi": pair("gbm", 0.9, "dg", ".2f"),
        r"\DgRangeCalibChi": pair("gbm", 1.0, "dg", ".2f"),
        r"\DgRangeSmallChi": pair("gbm", 1.1, "dg", ".2f"),
        r"\WelfRangeLargeChi": pair("gbm", 0.9, "welfare", ".1f"),
        r"\WelfRangeCalibChi": pair("gbm", 1.0, "welfare", ".1f"),
        r"\WelfRangeSmallChi": pair("gbm", 1.1, "welfare", ".1f"),
        r"\DgDeltaCalibLo": f"{dlo:.2f}",
        r"\DgDeltaCalibHi": f"{dhi:.2f}",
    }
    path = os.path.join(out_dir, "figure_macros.tex")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(rf"\newcommand{{{k}}}{{{v}}}"
                           for k, v in vals.items()) + "\n")
    print(f"figure_macros.tex  {vals}")


def main(estimation: str =
         "output/estimation_theta4.98898_sigma3.16692.json",
         transition: str = "output/transition_results.json",
         out_dir: str = "output"):
    p = est.load_estimated_parameters(estimation)
    alg_set = AlgorithmSettings()
    os.makedirs(out_dir, exist_ok=True)
    # Validate the accepted transition before overwriting any figure.
    transition_record, transition_arrays = load_transition_artifact(
        transition, estimation, p, alg_set, load_arrays=True)
    transition_out = dict(
        transition_arrays,
        stationary_0=transition_record["stationary_0"],
        stationary_T=transition_record["stationary_T"],
        welfare=transition_record["welfare"],
        diagnostics=transition_record["diagnostics"])

    # -- fig 1: corrected stationary density against the truncated Pareto
    eq = est.solve_bgp(p.d_0, p, alg_set)
    if not bool(eq["ok"]):
        raise RuntimeError("baseline BGP solve failed")
    tail, xi2 = float(eq["tail"]), float(eq["xi2"])
    z = np.linspace(1.0, 3.0, 601)
    f_corr = (tail * xi2 / (xi2 - tail)
              * (z ** (-tail - 1.0) - z ** (-xi2 - 1.0)))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(z, f_corr, color="red", lw=5, label="Corrected: $f(1)=0$")
    ax.plot(z, tail * z ** (-tail - 1.0), color="k", ls="-.", alpha=0.5,
            lw=3, label="Truncated Pareto (published)")
    strip(ax)
    ax.set_xlabel("Productivity Relative to Adoption Threshold, $z$",
                  fontsize=14)
    ax.set_ylabel("Stationary Density", fontsize=14)
    ax.set_xlim(1.0, 3.0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=14, frameon=False)
    save(fig, os.path.join(out_dir, "corr_density.pdf"),
         f"theta={tail:.3f} xi2={xi2:.3f}")

    # -- fig 2: growth and steady-state welfare against the size of the
    # trade-cost cut (cold BGP solves; a failed solve leaves a NaN gap);
    # published curve from the frozen legacy-model response
    cuts = np.linspace(0.0, 0.30, 13)
    dg = np.full(cuts.size, np.nan)
    gain = np.full(cuts.size, np.nan)
    for i, cut in enumerate(cuts):
        eq_cf = est.solve_bgp(1.0 + (p.d_0 - 1.0) * (1.0 - cut), p, alg_set)
        if bool(eq_cf["ok"]):
            dg[i] = 100.0 * float(eq_cf["g"] - eq["g"])
            gain[i] = 100.0 * float(est.welfare_gain_ce(eq, eq_cf, p.rho))
    pub_cf = np.loadtxt(os.path.join(DATA, "published_counterfactual.csv"),
                        delimiter=",", skiprows=1)   # cut, dg_pp, welfare_pct
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, y, y_pub, ylab in (
            (axes[0], dg, pub_cf[:, 1],
             "Change in Growth Rate,\n Percentage Points"),
            (axes[1], gain, pub_cf[:, 2],
             "Welfare Gain,\n Consumption Equivalent Percent")):
        ax.plot(100.0 * cuts, y, **corr_style("red"))
        ax.plot(100.0 * pub_cf[:, 0], y_pub, **PUB)
        strip(ax)
        ax.set_xlabel("Reduction in Trade Costs, Percent", fontsize=14)
        ax.set_ylabel(ylab, fontsize=14)
    axes[0].legend(fontsize=12, frameon=False)
    fig.tight_layout(pad=3)
    i10 = int(np.argmin(np.abs(cuts - 0.10)))
    save(fig, os.path.join(out_dir, "ovl_counterfactual.pdf"),
         f"10% cut: dg={dg[i10]:+.3f}pp gain={gain[i10]:+.2f}%")

    # -- figs 3-7: the 10%-cut transition experiment
    out = transition_out
    w = out["welfare"]
    print(f"transition: g {100 * float(out['stationary_0']['g']):.3f}% -> "
          f"{100 * float(out['stationary_T']['g']):.3f}%  "
          f"CE_transition={100 * w['CE_transition']:.2f}%  "
          f"CE_ss={100 * w['CE_ss_to_ss']:.2f}%")
    transition_figures(out, load_published_transition(), p, alg_set, out_dir)

    # -- figs 8-9: chi robustness sweeps around the calibration
    eq_T = est.solve_bgp(p.d_T, p, alg_set)
    if not bool(eq_T["ok"]):
        raise RuntimeError("10% cut BGP solve failed")
    base = dict(dg=100.0 * float(eq_T["g"] - eq["g"]),
                g0=100.0 * float(eq["g"]),
                welfare=100.0 * float(est.welfare_gain_ce(eq, eq_T, p.rho)))
    scales = np.linspace(0.1, 2.0, 150)
    sweeps = {}
    for kind, name in (("gbm", "corr_gbm_chi.pdf"),
                       ("delta", "corr_delta_chi.pdf")):
        sweeps[kind] = chi_sweep(kind, scales, p, alg_set)
        chi_figure(sweeps[kind], kind, scales,
                   os.path.join(out_dir, name), p, base)
    write_figure_macros(sweeps, scales, p, out_dir)


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(main)
