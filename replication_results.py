# Replication results: solve the corrected transition at the accepted SMM
# fit on the production grid and report the BGP endpoints, welfare, and
# solver statistics, with an auditable result artifact (arrays,
# diagnostics, settings, provenance).

import dataclasses
import json
import os
import platform
import subprocess
import sys
import time
from importlib import metadata

from utilities import (configure_process, load_transition_artifact,
                       sha256_file)

configure_process()

import numpy as np

import estimation as est
from model import AlgorithmSettings
from transition_dynamics import solve_transition_dynamics


def main(estimation: str =
         "output/estimation_theta4.98898_sigma3.16692.json",
         alg_set: AlgorithmSettings = AlgorithmSettings(),
         out_dir: str = "output"):
    params = est.load_estimated_parameters(estimation)
    t0 = time.time()
    out = solve_transition_dynamics(params, alg_set)
    wall = time.time() - t0
    stat_0, stat_T = out["stationary_0"], out["stationary_T"]
    summary = {
        "g_0": float(stat_0["g"]), "g_T": float(stat_T["g"]),
        # first cell of the discrete g path, not the reported impact rate
        "g_first_cell": float(out["g"][0]),
        "z_hat_0": float(stat_0["z_hat"]), "z_hat_T": float(stat_T["z_hat"]),
        "Omega_0": float(stat_0["Omega"]), "Omega_T": float(stat_T["Omega"]),
        "L_tilde_0": float(stat_0["L_tilde"]),
        "L_tilde_T": float(stat_T["L_tilde"]),
        "lambda_ii_0": float(stat_0["lambda_ii"]),
        "lambda_ii_T": float(stat_T["lambda_ii"]),
        "c_0": float(stat_0["c"]), "c_T": float(stat_T["c"]),
        "S_0": float(stat_0["S"]), "S_T": float(stat_T["S"]),
        "CE_transition %": 100.0 * out["welfare"]["CE_transition"],
        "CE_ss_to_ss %": 100.0 * out["welfare"]["CE_ss_to_ss"],
        "outer_iters": out["diagnostics"]["outer_iters"],
        "hjb_resid": out["diagnostics"]["hjb_resid"],
        "kfe_resid": out["diagnostics"]["kfe_resid"],
        "wall_s": wall,
    }
    for key, value in summary.items():
        print(f"{key:>15}  {value:.6g}")

    os.makedirs(out_dir, exist_ok=True)
    arrays_path = os.path.join(out_dir, "transition_arrays.npz")
    arrays_tmp = arrays_path + ".tmp"
    array_keys = ("t", "g", "g_reporting", "log_M", "z_hat", "Omega", "E",
                  "L_tilde", "z_bar", "lambda_ii", "c", "S", "f", "v0")
    with open(arrays_tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            **{key: np.asarray(out[key]) for key in array_keys})
    os.replace(arrays_tmp, arrays_path)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    record = dict(
        schema_version=1,
        summary=summary, welfare=out["welfare"],
        diagnostics=out["diagnostics"],
        stationary_0={key: float(stat_0[key]) for key in
                      ("g", "z_hat", "Omega", "L_tilde", "z_bar",
                       "lambda_ii", "c", "S")},
        stationary_T={key: float(stat_T[key]) for key in
                      ("g", "z_hat", "Omega", "L_tilde", "z_bar",
                       "lambda_ii", "c", "S")},
        params=dataclasses.asdict(params),
        alg_set=dataclasses.asdict(alg_set),
        estimation=dict(path=estimation, sha256=sha256_file(estimation)),
        arrays=dict(path=os.path.basename(arrays_path),
                    sha256=sha256_file(arrays_path), keys=list(array_keys)),
        reporting_selection=dict(
            impact_growth="g_reporting[0] equals pre-shock BGP growth",
            raw_growth="g[0] is the discrete first-cell diagnostic",
            accumulated_productivity="log_M integrates g_reporting"),
        argv=sys.argv,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        commit=subprocess.run(
            ["git", "-C", src_dir, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip(),
        dirty=subprocess.run(
            ["git", "-C", src_dir, "status", "--porcelain"],
            capture_output=True, text=True, check=True).stdout.strip() != "",
        python=platform.python_version(), platform=platform.platform(),
        versions={name: metadata.version(name)
                  for name in ("jax", "optimistix", "lineax", "numpy")},
        source_sha256={name: sha256_file(os.path.join(src_dir, name))
                       for name in sorted(os.listdir(src_dir))
                       if name.endswith(".py")})
    record_path = os.path.join(out_dir, "transition_results.json")
    record_tmp = record_path + ".tmp"
    with open(record_tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)
    os.replace(record_tmp, record_path)
    # A standalone results run succeeds only after its closed payload passes
    # the same integrity and reporting-trace checks used by downstream code.
    load_transition_artifact(record_path, estimation, params, alg_set,
                             load_arrays=True)
    print(f"result artifact -> "
          f"{os.path.join(out_dir, 'transition_results.json')} "
          f"+ transition_arrays.npz")
    return summary


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(main)
