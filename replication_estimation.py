# SMM estimation conditional on pinned (theta, sigma).  The CLI takes those
# published AER values and re-estimates (d, kappa, 1/chi, mu, upsilon),
# reporting the fit and the 10% trade-cut welfare gain across steady states.

import json
import os

from utilities import configure_process

configure_process()

import jax.numpy as jnp
import numpy as np

import estimation as est
from model import AlgorithmSettings, StructuralParameters


def main(theta: float = 4.988976587938262,
         sigma: float = 3.166924135838110,
         out_dir: str = "output"):
    targets, rho, delta = est.load_targets()
    p_pin = StructuralParameters(rho=rho, delta=delta, theta=theta,
                                 sigma=sigma)
    alg_set = AlgorithmSettings()

    print(f"SMM at theta={theta:g}, sigma={sigma:g}:")
    rng = np.random.default_rng(20260729)
    y0 = np.asarray(est.to_transformed(jnp.array(est.X0_BASE)))
    y0s = [y0, np.asarray(est.to_transformed(jnp.array(est.X0_PUB)))]
    y0s += [y0 + rng.uniform(-0.15, 0.15, y0.size) for _ in range(3)]
    fit = est.estimate(y0s, p_pin, alg_set, targets)

    x_hat = fit["x"]
    mom, eq, stationary_resid = est.model_moments(x_hat, p_pin, alg_set)
    d_hat, p_hat = est.params_from_x(np.asarray(x_hat).tolist(), p_pin)

    # 10% cut in the iceberg margin d - 1
    d_cf = 0.90 * (d_hat - 1.0) + 1.0
    eq_cf = est.solve_bgp(d_cf, p_hat, alg_set)
    if not bool(eq_cf["ok"]):
        raise RuntimeError("10% cut counterfactual BGP solve failed")
    welfare = float(est.welfare_gain_ce(eq, eq_cf, p_hat.rho))

    print(f"  d={d_hat:.12f}  objective={float(fit['objective']):.12f}  "
          f"g={float(eq['g']):.6f}  welfare_10pct_ss={welfare:.4%}")

    out = dict(
        theta=theta, sigma=sigma, rho=rho, delta=delta,
        x=dict(zip(est.X_NAMES, map(float, np.asarray(x_hat)))),
        objective=float(fit["objective"]),
        moments=dict(zip(est.MOMENT_NAMES, map(float, np.asarray(mom)))),
        targets=dict(zip(est.MOMENT_NAMES, map(float, np.asarray(targets)))),
        g=float(eq["g"]), z_hat=float(eq["z_hat"]), Omega=float(eq["Omega"]),
        tail=float(eq["tail"]), xi2=float(eq["xi2"]),
        size_tail=float(eq["tail"]) / (sigma - 1.0),
        welfare_gain_10pct_ss=welfare,
        stationary_resid=float(stationary_resid),
        bgp_resid=float(eq["bgp_resid"]))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir,
                        f"estimation_theta{theta:g}_sigma{sigma:g}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"  saved {path}")
    return out


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(main)
