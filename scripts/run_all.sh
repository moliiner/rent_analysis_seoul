#!/usr/bin/env bash
# Runs the test suite and then every modeling notebook in order (00 .. 09).
# Usage:  bash scripts/run_all.sh            run everything
#         DRY_RUN=1 bash scripts/run_all.sh  only print the commands
# Notebooks are executed in place (outputs are overwritten). Each model run is appended to
# reports/metrics/{track}__{model_id}.json; every table reads the last run of each model.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
DRY_RUN="${DRY_RUN:-0}"

NOTEBOOKS=(
  notebooks/00_modeling_setup.ipynb
  notebooks/01_baselines.ipynb
  notebooks/02_linear_models.ipynb
  notebooks/03_random_forest.ipynb
  notebooks/04_gradient_boosting.ipynb
  notebooks/05_kmeans_clustering.ipynb
  notebooks/06_hierarchical_gmm.ipynb
  notebooks/07_price_index_forecasting.ipynb
  notebooks/08_hybrid_model.ipynb
  notebooks/09_model_comparison_projection.ipynb
)

run() {
  echo ">> $*"
  if [ "$DRY_RUN" != "1" ]; then
    "$@"
  fi
}

run "$PYTHON" -m pytest -q
for notebook in "${NOTEBOOKS[@]}"; do
  run "$PYTHON" -m nbconvert --to notebook --execute --inplace "$notebook" \
      --ExecutePreprocessor.timeout=-1 --log-level WARN
done
echo "Done."
