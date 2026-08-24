# Discrete-BGP-to-analytic convergence table (numerics appendix Table 1):
# the discrete stationary backend at the initial BGP on successively refined
# grids, against the analytic backend.  Run by replicate.sh; writes
# bgp_grid_convergence.json, consumed by numerics_macros.py.

import dataclasses
import json
import math
import os

from utilities import configure_process

configure_process()

import jax.numpy as jnp

import estimation as est
from model import AlgorithmSettings, z_grid
from stationary_solution import solve_stationary

# (z_segment_1_points, z_segment_2_points, z_segment_3_points): successive
# halvings of the production grid, giving 1077, 2157, 4317 density states
GRIDS = ((360, 480, 240), (720, 960, 480), (1440, 1920, 960))
THETA0 = jnp.array([0.008, 0.0, math.log(0.68)])


def main(estimation: str =
         "output/estimation_theta4.98898_sigma3.16692.json",
         out_dir: str = "output"):
    p = est.load_estimated_parameters(estimation)
    rows = []
    for n1, n2, n3 in GRIDS:
        alg_set = dataclasses.replace(
            AlgorithmSettings(), z_segment_1_points=n1,
            z_segment_2_points=n2, z_segment_3_points=n3)
        states = int(z_grid(alg_set).shape[0] - 1)
        out = solve_stationary(p, p.d_0, alg_set, THETA0)
        rows.append(dict(states=states, g=float(out["g"]), S=float(out["S"]),
                         lambda_ii=float(out["lambda_ii"]),
                         qsd_resid=float(out["stationary_resid"])))
    eq = est.solve_bgp(p.d_0, p, AlgorithmSettings())
    analytic = dict(states="analytic", g=float(eq["g"]), S=float(eq["S"]),
                    lambda_ii=float(eq["lam_ii"]), qsd_resid=None)

    print(f"{'states':>10}{'g':>14}{'S':>14}{'lambda_ii':>14}"
          f"{'qsd_resid':>12}")
    for r in rows + [analytic]:
        qr = "---" if r["qsd_resid"] is None else f"{r['qsd_resid']:.2e}"
        print(f"{str(r['states']):>10}{r['g']:14.8f}{r['S']:14.8f}"
              f"{r['lambda_ii']:14.8f}{qr:>12}")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "bgp_grid_convergence.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(grid_rows=rows, analytic=analytic), fh, indent=2)
        fh.write("\n")
    print(f"-> {path}")


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(main)
