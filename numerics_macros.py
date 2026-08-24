# \newcommand's for every computed number in PTW_corrigendum_numerics.tex,
# sourced from the estimation, transition, grid-convergence, verification, and
# inline-results artifacts.  Grid sizes, tolerances, and other settings stay
# literals in the .tex.  Rounding matches the prose/table each macro replaces.
# Runs last in replicate.sh: needs the artifacts the earlier steps write.

import json
import math
import os


def sci(v, digits=2):
    # LaTeX scientific notation m \times 10^{e}
    mant, exp = f"{abs(v):.{digits}e}".split("e")
    sign = "-" if v < 0 else ""
    return rf"{sign}{mant} \times 10^{{{int(exp)}}}"


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main(estimation: str =
         "output/estimation_theta4.98898_sigma3.16692.json",
         transition: str = "output/transition_results.json",
         grid: str = "output/bgp_grid_convergence.json",
         verification: str =
         "output/verification_theta4.98898_sigma3.16692.json",
         inline: str = "output/inline_results.json",
         out_dir: str = "output"):
    est = load(estimation)
    tr = load(transition)
    gj = load(grid)
    ver = load(verification)
    inl = load(inline)
    x, diag, wel = est["x"], tr["diagnostics"], tr["welfare"]
    rows = {int(r["states"]): r for r in gj["grid_rows"]}
    an = gj["analytic"]
    xbar = tr["alg_set"]["z_max"]

    # derived at the fitted analytic BGP
    g, mu, ups, th = est["g"], x["mu"], x["upsilon"], est["theta"]
    inherited_S = th * (g - mu - th * ups ** 2 / 2.0)
    critical_S = (g - mu) ** 2 / (2.0 * ups ** 2)
    tail_decay = math.exp(-est["xi2"] * xbar)

    def brow(n):  # (growth, adoption, home-share) macros for a grid row
        r = rows[n]
        return (f"{r['g']:.8f}", f"{r['S']:.8f}", f"{r['lambda_ii']:.8f}",
                sci(r["qsd_resid"]))

    ga, sa, la, ra = brow(1077)
    gb, sb, lb, rb = brow(2157)
    gc, sc, lc, rc = brow(4317)

    vals = {
        # --- fitted SMM point (numerics precision) ---
        r"\dFit": f"{x['d']:.7f}",
        r"\kappaFit": f"{x['kappa']:.7f}",
        r"\invchiFit": f"{x['inv_chi']:.7f}",
        r"\muFit": f"{x['mu']:.7f}",
        r"\upsFit": f"{x['upsilon']:.7f}",
        r"\XiTwo": f"{est['xi2']:.4f}",
        r"\XiTwoMinusTheta": f"{est['xi2'] - est['theta']:.4f}",
        r"\XiTwoTailDecay": sci(tail_decay, 1),
        # --- inherited vs critical adoption at fixed analytic g ---
        r"\InheritedAdoption": f"{inherited_S:.6f}",
        r"\CriticalAdoption": f"{critical_S:.6f}",
        # --- discrete-BGP convergence table (states 1077 / 2157 / 4317 / an) ---
        r"\BgpGA": ga, r"\BgpSA": sa, r"\BgpLA": la, r"\BgpRA": ra,
        r"\BgpGB": gb, r"\BgpSB": sb, r"\BgpLB": lb, r"\BgpRB": rb,
        r"\BgpGC": gc, r"\BgpSC": sc, r"\BgpLC": lc, r"\BgpRC": rc,
        r"\BgpGN": f"{an['g']:.8f}", r"\BgpSN": f"{an['S']:.8f}",
        r"\BgpLN": f"{an['lambda_ii']:.8f}",
        # --- verifier ---
        r"\VerifierMaxPct": f"{100.0 * ver['max_rel_diff']:.2f}",
        # --- transition acceptance gates (production values) ---
        r"\GateNodeRoot": sci(diag["hjb_root_resid"]),
        r"\GateAssembled": sci(diag["hjb_resid"]),
        r"\GateKfe": sci(diag["kfe_resid"]),
        r"\GateMass": sci(diag["mass_err"]),
        r"\GateMinF": sci(diag["min_f"]),
        r"\GateMinE": f"{diag['min_E']:.5f}",
        r"\GateFinalG": sci(diag["final_err_g"]),
        r"\GateFinalE": sci(diag["final_err_E"]),
        r"\GateTermOmega": sci(abs(diag["terminal_omega"])),
        # --- outer fixed point ---
        r"\OuterIters": str(int(tr["summary"]["outer_iters"])),
        # --- impact reporting selection ---
        r"\ImpactGrowthSel": f"{tr['summary']['g_0']:.8f}",
        r"\ImpactGrowthRaw": f"{tr['summary']['g_first_cell']:.8f}",
        r"\CETransitionFive": f"{100.0 * wel['CE_transition']:.5f}",
        r"\CERawFive": f"{inl['transition']['ce_raw_first_cell_percent']:.5f}",
        r"\RawMinusSelFive":
            f"{inl['transition']['raw_minus_selected_percentage_points']:.5f}",
        # --- validation welfare and residuals ---
        r"\CEssGridFive": f"{100.0 * wel['CE_ss_to_ss']:.5f}",
        r"\AnalyticGainFive": f"{100.0 * est['welfare_gain_10pct_ss']:.5f}",
        r"\GainRoundedPct": f"{100.0 * wel['CE_ss_to_ss']:.1f}",
        r"\StatResidZero": sci(diag["stationary_0_resid"]),
        r"\StatResidT": sci(diag["stationary_T_resid"]),
    }

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "numerics_macros.tex")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(rf"\newcommand{{{k}}}{{{v}}}"
                           for k, v in vals.items()) + "\n")
    for k, v in vals.items():
        print(f"  {k:<20} {v}")
    print(f"-> {path}")


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(main)
