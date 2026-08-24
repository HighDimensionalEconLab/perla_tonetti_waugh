#!/usr/bin/env bash
# Full replication: SMM estimation, transition dynamics, the corrigendum's
# tables, eight manuscript figures, one auxiliary figure, the discrete-BGP
# convergence table, and numerics-appendix macros.  Everything lands in
# output/ (gitignored).
# numerics_macros.py runs last: it sources every computed number in the
# numerics appendix from the artifacts the earlier steps write.
set -euo pipefail
cd "$(dirname "$0")"
uv run python replication_estimation.py
uv run python estimation_robust.py
uv run python replication_results.py
uv run python replication_tables.py
uv run python replication_figures.py
uv run python bgp_grid_convergence.py
uv run python numerics_macros.py
