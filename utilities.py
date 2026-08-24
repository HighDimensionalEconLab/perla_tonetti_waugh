# Process configuration, artifact validation, and the Anderson fixed-point
# solver x* = G(x*, *args).

import dataclasses
import hashlib
import json
import os

import jax
import jax.numpy as jnp
import numpy as np


def configure_process(cache_path=".jax_cache"):
    # must run before the first compilation; the low thresholds cache our
    # sub-second compiles
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_compilation_cache_dir", cache_path)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_transition_artifact(manifest_path, estimation_path, params, alg_set,
                             load_arrays=False):
    """Load a transition record after verifying its inputs and array payload."""
    with open(manifest_path, encoding="utf-8") as fh:
        record = json.load(fh)
    if record.get("schema_version") != 1:
        raise RuntimeError("unsupported transition artifact schema")
    expected_estimation = sha256_file(estimation_path)
    if record.get("estimation", {}).get("sha256") != expected_estimation:
        raise RuntimeError("transition artifact was generated from a different "
                           "estimation file")
    if record.get("params") != dataclasses.asdict(params):
        raise RuntimeError("transition artifact parameter vector does not match "
                           "the requested estimation")
    if record.get("alg_set") != dataclasses.asdict(alg_set):
        raise RuntimeError("transition artifact was not generated with the "
                           "requested algorithm settings")
    endpoint_keys = {"g", "z_hat", "Omega", "L_tilde", "z_bar",
                     "lambda_ii", "c", "S"}
    for endpoint in ("stationary_0", "stationary_T"):
        if not endpoint_keys.issubset(record.get(endpoint, {})):
            raise RuntimeError(f"transition manifest is missing {endpoint} "
                               "reporting fields")

    arrays_record = record.get("arrays", {})
    arrays_name = arrays_record.get("path")
    arrays_hash = arrays_record.get("sha256")
    if not arrays_name or not arrays_hash:
        raise RuntimeError("transition manifest does not identify and hash its "
                           "array payload")
    if os.path.isabs(arrays_name) or os.path.basename(arrays_name) != arrays_name:
        raise RuntimeError("transition array payload must be a local basename")
    arrays_path = os.path.join(os.path.dirname(manifest_path), arrays_name)
    if not os.path.exists(arrays_path):
        raise FileNotFoundError(f"transition array payload not found: {arrays_path}")
    if sha256_file(arrays_path) != arrays_hash:
        raise RuntimeError("transition array payload does not match its manifest")

    arrays = None
    if load_arrays:
        with np.load(arrays_path, allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        declared = arrays_record.get("keys")
        if declared is None or sorted(arrays) != sorted(declared):
            raise RuntimeError("transition array keys do not match the manifest")
        required = ("t", "g", "g_reporting", "log_M", "z_hat", "Omega",
                    "E", "L_tilde", "z_bar", "lambda_ii", "c", "S", "f")
        if any(key not in arrays for key in required):
            raise RuntimeError("transition array payload is incomplete")
        n_t = arrays["t"].shape[0]
        if any(arrays[key].shape[0] != n_t for key in required[1:]):
            raise RuntimeError("transition arrays do not share a time dimension")
        if arrays["f"].ndim != 2:
            raise RuntimeError("transition density path must be two-dimensional")
        stat_g = record["stationary_0"]["g"]
        if not np.isclose(arrays["g_reporting"][0], stat_g,
                          rtol=0.0, atol=1e-14):
            raise RuntimeError("reporting growth does not start at the initial BGP")
        if not np.allclose(arrays["g_reporting"][1:], arrays["g"][1:],
                           rtol=0.0, atol=1e-14):
            raise RuntimeError("reporting and raw growth differ after impact")
        log_M = np.concatenate([
            np.zeros(1),
            np.cumsum(0.5 * (arrays["g_reporting"][:-1]
                             + arrays["g_reporting"][1:])
                      * np.diff(arrays["t"]))])
        if not np.allclose(arrays["log_M"], log_M, rtol=0.0, atol=1e-13):
            raise RuntimeError("saved log_M does not integrate reporting growth")
    return record, arrays


def fixed_point(G, x0, args, depth, tol, maxit, ridge, verbose=False):
    # Anderson acceleration on f = G(x) - x, extrapolating from the first step
    # over however much history exists: with an empty history the step is
    # plain x <- G(x), so no separate warmup phase runs undamped.  Solvers
    # that instead run a fixed undamped warmup diverge here on the unstable
    # level mode.  optimistix PR #217 adds an equivalent solver upstream;
    # switch to it once merged.
    eye = jnp.eye(depth)

    def step(state):
        k, x, g_prev, f_prev, dF, dG, _ = state
        g = G(x, *args)
        f = g - x
        # rolling buffers of the last `depth` differences of f and of G(x)
        pos = (k - 1) % depth
        dF = dF.at[pos].set(jnp.where(k > 0, f - f_prev, dF[pos]))
        dG = dG.at[pos].set(jnp.where(k > 0, g - g_prev, dG[pos]))
        # gamma = argmin ||f - dF'gamma||, masked to the filled slots: unfilled
        # ones get a unit diagonal so the solve stays nonsingular and returns 0
        live = jnp.arange(depth) < jnp.minimum(k, depth)
        Fm = jnp.where(live[:, None], dF, 0.0)
        A = jnp.where(live[:, None] & live[None, :], Fm @ Fm.T + ridge * eye,
                      eye)
        gamma = jnp.linalg.solve(A, jnp.where(live, Fm @ f, 0.0))
        err = jnp.linalg.norm(f)
        if verbose:
            jax.debug.print("  anderson {k}: ||G(x)-x|| = {e:.6e}", k=k, e=err)
        # x <- G(x) - dG'gamma
        return (k + 1, g - gamma @ jnp.where(live[:, None], dG, 0.0), g, f,
                dF, dG, err)

    zeros = jnp.zeros((depth,) + x0.shape)
    state = (0, x0, x0, jnp.zeros_like(x0), zeros, zeros, jnp.inf)
    iters, _, g, _, _, _, err = jax.lax.while_loop(
        lambda s: (s[6] > tol) & (s[0] < maxit), step, state)
    error = float(err)
    if not error < tol:
        raise RuntimeError(f"fixed_point: ||G(x)-x|| = {error:.2e} > {tol:g} "
                           f"after {maxit} iterations")
    return g, int(iters), error
