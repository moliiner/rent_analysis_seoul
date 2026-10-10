# Seoul rental market project. Targets: install, test, notebooks, all, app.
# The notebooks must run in order 00 .. 08, 10, 09 (later notebooks read the runs saved by earlier ones).
PYTHON ?= python

.PHONY: install test notebooks all app

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

# Equivalent to: bash scripts/run_all.sh (without the tests)
notebooks:
	@for nb in \
	  notebooks/00_modeling_setup.ipynb notebooks/01_baselines.ipynb notebooks/02_linear_models.ipynb \
	  notebooks/03_random_forest.ipynb notebooks/04_gradient_boosting.ipynb notebooks/05_kmeans_clustering.ipynb \
	  notebooks/06_hierarchical_gmm.ipynb notebooks/07_price_index_forecasting.ipynb notebooks/08_hybrid_model.ipynb \
	  notebooks/10_selection_robustness.ipynb notebooks/09_model_comparison_projection.ipynb; do \
	  echo ">> $$nb"; \
	  $(PYTHON) -m nbconvert --to notebook --execute --inplace $$nb --ExecutePreprocessor.timeout=-1 --log-level WARN || exit 1; \
	done

all: test notebooks

# Optional explorer (needs: pip install streamlit)
app:
	$(PYTHON) -m streamlit run app/streamlit_app.py
