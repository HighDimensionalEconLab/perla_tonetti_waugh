# Perla, Tonetti, and Waugh: Corrigendum Replication

JAX replication of the corrected model in Perla, Tonetti, and Waugh (AER):
SMM calibration, balanced-growth-path counterfactuals, transition dynamics,
and every table and figure the corrigendum includes. The repository is
self-contained: the published-model overlays and SMM data targets are frozen
CSV artifacts in `data/`.

See [Corrigendum: Equilibrium Technology Diffusion, Trade, and Growth](https://christophertonetti.com/files/papers/PerlaTonettiWaugh_Corrigendum.pdf) for a description of the new quantitative results and the numerical algorithm.


## Setup

Install [uv](https://docs.astral.sh/uv/):
- `curl -LsSf https://astral.sh/uv/install.sh | sh` on MacOS or Linux
- `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` on Windows

`uv run` installs the pinned dependencies on first use; `uv sync` installs
them explicitly from `uv.lock`.

## Full replication

Replicate all results and generated figures, tables with

```bash
bash replicate.sh
```

Results are saved `output/`. Timings on an
Apple-silicon laptop are about 1 minute. Individual drivers, from the
repository root:

```bash
uv run python replication_estimation.py     # SMM at the published AER theta and sigma: fit and welfare
uv run python estimation_robust.py          # independent closed-form-versus-grid verification of that fit
uv run python replication_results.py        # transition dynamics for the 10% trade-cost cut: welfare and result artifact
uv run python replication_tables.py         # the corrigendum's tables as \input-ready fragments, the eq-(60) welfare decomposition, and the published-path welfare re-aggregation
uv run python replication_figures.py        # manuscript figures and an auxiliary counterfactual sweep
uv run python bgp_grid_convergence.py       # discrete-BGP-to-analytic convergence table across grid refinements
uv run python numerics_macros.py            # numerics-appendix macros sourced from the artifacts above
```

The estimation writes `estimation_theta4.98898_sigma3.16692.json`; the
downstream drivers load their parameters from it.

## Outputs

A full run leaves in `output/`:

- `estimation_theta4.98898_sigma3.16692.json` — the accepted SMM fit
- `verification_theta4.98898_sigma3.16692.json` — the independent verification summary
- `transition_results.json` and `transition_arrays.npz` — the auditable transition result artifact
- `bgp_grid_convergence.json` — the grid-convergence table inputs
- `inline_results.json` — the rounded numbers used in the corrigendum's prose
- `table_params.tex`, `table_moments.tex`, `table_persistence.tex`,
  `table_results.tex`, `decomposition_equation.tex` — `\input`-ready table fragments
- `inline_macros.tex`, `figure_macros.tex`, `numerics_macros.tex` — `\input`-ready macro files
- the eight manuscript figures `corr_density.pdf`, `ovl_trade.pdf`,
  `ovl_omega.pdf`, `ovl_fixed_costs.pdf`, `ovl_log_consumption_inset.pdf`,
  `ovl_growth.pdf`, `corr_gbm_chi.pdf`, `corr_delta_chi.pdf`, plus the
  auxiliary `ovl_counterfactual.pdf`

## Variations

Every driver exposes its options on the CLI (`--help` for the full list):

```bash
# results-only transition solve on a coarser preview grid (the table and
# figure drivers intentionally reject it because they require the production
# AlgorithmSettings; the defaults are the production grids)
uv run python replication_results.py \
  --alg_set.z_segment_1_points 720 --alg_set.z_segment_2_points 960 \
  --alg_set.z_segment_3_points 480
```
