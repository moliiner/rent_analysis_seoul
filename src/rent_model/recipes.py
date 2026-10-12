"""Model recipes: fit a named model on the rows before a cutoff and predict later rows.

A recipe is the single definition of a candidate used by the rolling-origin pseudo-tests, by the
behavioural extrapolation check and by the final projection. `fit` receives only rows dated before
`cutoff` (it asserts it), so nothing at or after the cutoff can influence the fit. Hyperparameters
are those saved by the notebooks for the base model; a `_median` suffix switches a boosting model to
the median-aligned loss (HistGradientBoostingRegressor loss='quantile', quantile=0.5 on log1p).

Neighbourhood-aware trend models (`hgb_detrended_ld`, `hybrid_hgb_ld`, with `_median` variants) add the
grouped `legal_dong` feature of `features.NeighbourhoodGrouper`; their settings come from the run saved by
notebook 11 or are passed explicitly (they are tuned inside each fit window). A `_recal` suffix adds one
global scale factor, estimated on the last CV fold of the fit window and applied to the predictions.
"""
import json

import numpy as np
import pandas as pd

from . import config
from .baselines import DistrictTypeAreaMedian
from .boosting import BoostedModel, required_columns
from .compare import _base_id, profile_frame, representative_profile
from .features import build_features, build_target
from .splits import window_folds
from .forest import feature_columns as forest_columns, make_model as make_forest
from .forest import to_original
from .hybrid import StructureModel, adjust_target, forecast_adjustment, index_adjustment
from .linear import _to_original as linear_to_original
from .linear import build_pipeline as build_linear_pipeline
from .ts_index import hedonic_index

MEDIAN_QUANTILE = 0.5
BOOSTING_BASES = ('hgb_static', 'hgb_time', 'hgb_detrended', 'hgb_legal_dong')
BOOSTING_VARIANT = {'hgb_static': 'static', 'hgb_time': 'time', 'hgb_detrended': 'detrended', 'hgb_legal_dong': 'legal_dong'}
RECIPE_BASES = ('baseline_district_type_area', 'ridge_static', 'ridge_time', 'ridge_time_rich', 'rf_detrended',
                *BOOSTING_BASES, 'hybrid_ridge', 'hybrid_hgb')
LD_BASES = ('hgb_detrended_ld', 'hybrid_hgb_ld')
RECAL_BASES = ('hgb_detrended_ld_median', 'hybrid_hgb_ld_median')       # models that have a `_recal` version
FIXED_PROFILE = ('Mapo-gu', 'Apartment', 60.0)   # profile used for the behavioural extrapolation check


def recipe_ids():
    """Every model id with a recipe: the base models and the median-loss variants of the boosting ones."""
    medians = [f'{b}_median' for b in (*BOOSTING_BASES, 'hybrid_hgb')]
    return [*RECIPE_BASES, *medians]


def ld_recipe_ids():
    """Neighbourhood-aware trend models: squared and median loss, and the recalibrated median versions."""
    ids = [m for b in LD_BASES for m in (b, f'{b}_median')]
    return [*ids, *[f'{m}_recal' for m in RECAL_BASES]]


def all_recipe_ids():
    return [*recipe_ids(), *ld_recipe_ids()]


def saved_params(track, model_id, metrics_dir=None):
    """Hyperparameters saved by the notebook of the model (empty for the baseline).

    The neighbourhood-aware models look for their own run first (their settings are tuned for them);
    the other models read the run of their base model."""
    base = _base_id(model_id)
    if base.startswith('baseline'):
        return {}
    metrics_dir = config.METRICS_DIR if metrics_dir is None else metrics_dir
    own = metrics_dir / f'{track}__{model_id.removesuffix("_recal")}.json'
    path = own if base in LD_BASES and own.exists() else metrics_dir / f'{track}__{base}.json'
    meta = json.loads(path.read_text(encoding='utf-8'))[-1]['meta']
    return dict(meta['structure_params'] if base.startswith('hybrid') else meta['params'])


def saved_index_candidate(track, metrics_dir=None):
    """Index forecaster selected before 2025 in notebook 07 (naive or drift)."""
    metrics_dir = config.METRICS_DIR if metrics_dir is None else metrics_dir
    report = json.loads((metrics_dir / f'ts_index_{track}.json').read_text(encoding='utf-8'))
    return report['rolling_origin']['selected']


class ModelRecipe:
    """Fit on rows before `cutoff`; predict on any feature frame (original wones, never negative)."""

    def __init__(self, model_id, track, params=None, index_candidate=None):
        if model_id not in all_recipe_ids():
            raise ValueError(f'no recipe for {model_id}')
        self.model_id, self.track = model_id, track
        self.recal = model_id.endswith('_recal')
        self.core_id = model_id.removesuffix('_recal')
        self.base = _base_id(model_id)
        self.neighbourhood = self.base in LD_BASES
        self.quantile = MEDIAN_QUANTILE if self.core_id.endswith('_median') else None
        self.params = saved_params(track, model_id) if params is None else dict(params)
        self.index_candidate = index_candidate
        self.is_hybrid = self.base.startswith('hybrid')
        if self.is_hybrid and self.index_candidate is None:
            self.index_candidate = saved_index_candidate(track)

    # ---------------------------------------------------------------- fit
    def fit(self, train, cutoff):
        cutoff = pd.Timestamp(cutoff)
        assert train['contract_date'].max() < cutoff, 'a row at or after the cutoff reached the fit'
        self.cutoff = cutoff
        target = config.TRACKS[self.track]['target']
        X = build_features(train, self.track, include_time=True, include_legal_dong=True)
        y = build_target(train, self.track)
        self.floor_log = float(y.min())
        if self.base.startswith('baseline'):
            self.model_ = DistrictTypeAreaMedian().fit(train, target)
        elif self.base.startswith('ridge'):
            self.model_ = build_linear_pipeline('ridge', time=self.base != 'ridge_static', rich=self.base == 'ridge_time_rich',
                                                **self.params).fit(X, y)
        elif self.base == 'rf_detrended':
            self.model_ = make_forest('detrended', **self.params).fit(X[forest_columns(True)], y)
        elif self.base == 'hgb_detrended_ld':
            self.columns_ = required_columns('detrended', neighbourhood=True)
            self.model_ = BoostedModel(variant='detrended', quantile=self.quantile, neighbourhood=True, **self.params).fit(X[self.columns_], y)
        elif self.base in BOOSTING_VARIANT:
            variant = BOOSTING_VARIANT[self.base]
            self.model_ = BoostedModel(variant=variant, quantile=self.quantile, **self.params).fit(X[required_columns(variant)], y)
            self.columns_ = required_columns(variant)
        else:   # hybrid: structure model without time on y_adj (TRAIN-only hedonic index)
            self.index_ = hedonic_index(train, self.track, last_month=cutoff - pd.offsets.MonthBegin(1))
            y_adj = adjust_target(y, train['contract_month'], self.index_)
            if self.base == 'hybrid_ridge':
                self.model_ = StructureModel('ridge', self.params).fit(X, y_adj)
            elif self.base == 'hybrid_hgb_ld':
                self.columns_ = required_columns('static', neighbourhood=True)
                self.model_ = BoostedModel(variant='static', quantile=self.quantile, neighbourhood=True, **self.params).fit(X[self.columns_], y_adj)
            elif self.quantile is None:
                self.model_ = StructureModel('hgb', self.params).fit(X, y_adj)
            else:
                self.model_ = BoostedModel(variant='static', quantile=self.quantile, **self.params).fit(X[required_columns('static')], y_adj)
        self.recal_factor_ = self._estimate_factor(train) if self.recal else 1.0
        return self

    def _estimate_factor(self, train):
        """Global scale factor from the LAST CV fold of the fit window (3-fold expanding window).

        The same model (without the factor) is fit on the rows before the last validation block and predicts
        that block; the factor is sum(actual) / sum(predicted) there. Only rows of the fit window are used:
        nothing from the evaluation window can enter it."""
        train_pos, valid_pos = window_folds(train)[-1]
        valid = train.iloc[valid_pos]
        valid_start = valid['contract_month'].min()
        aux = ModelRecipe(self.core_id, self.track, params=self.params, index_candidate=self.index_candidate)
        aux.fit(train.iloc[train_pos], valid_start)
        pred = aux.predict(build_features(valid, self.track, include_time=True, include_legal_dong=True), valid['contract_month'])
        y = valid[config.TRACKS[self.track]['target']].to_numpy(dtype=float)
        self.recal_info_ = {'valid_start': str(valid_start.date()), 'valid_last': str(valid['contract_date'].max().date()),
                            'n_valid': int(len(valid)), 'n_fit_aux': int(len(train_pos))}
        return float(y.sum() / pred.sum())

    # ------------------------------------------------------------ predict
    def _adjustment(self, months):
        """log(index_t / 100): the fitted index for months inside the fit window, the forecast after it."""
        months = pd.DatetimeIndex(months)
        last = self.index_.index.max()
        inside = months <= last
        adj = np.empty(len(months))
        adj[inside] = index_adjustment(self.index_, months[inside]) if inside.any() else []
        if (~inside).any():
            horizon = (months.max().year - last.year) * 12 + months.max().month - last.month
            forecast = forecast_adjustment(self.index_, self.index_candidate, horizon=horizon)
            adj[~inside] = forecast.reindex(months[~inside]).to_numpy()
        return adj

    def predict_log(self, frame, months=None):
        """Prediction on the log1p scale before back-transformation (not available for the baseline)."""
        if self.base.startswith('baseline'):
            raise NotImplementedError('the baseline predicts original wones directly')
        if self.base.startswith('ridge'):
            return self.model_.predict(frame)
        if self.base == 'rf_detrended':
            return self.model_.predict(frame[forest_columns(True)])
        if self.base == 'hgb_detrended_ld' or self.base in BOOSTING_VARIANT:
            return self.model_.predict(frame[self.columns_])
        if self.base == 'hybrid_hgb_ld':
            structure = self.model_.predict(frame[self.columns_])
        elif self.base == 'hybrid_ridge' or self.quantile is None:
            structure = self.model_.predict_log(frame)
        else:
            structure = self.model_.predict(frame[required_columns('static')])
        return structure + self._adjustment(months)

    def predict(self, frame, months=None):
        if self.base.startswith('baseline'):
            # exp(log(area)) is rounded back to the original area (2 decimals): bucket edges are exact area values
            return self.model_.predict(frame.assign(leased_area_sqm=np.round(np.exp(frame['log_area']), 6)))
        pred_log = self.predict_log(frame, months)
        if self.base.startswith('ridge'):
            return linear_to_original(pred_log, self.floor_log)
        return to_original(pred_log) * (self.recal_factor_ if self.recal else 1.0)


def fixed_profile(df_track):
    """The fixed profile of the behavioural check, with the most frequent neighbourhood of its group."""
    district, building_type, area = FIXED_PROFILE
    profile, _ = representative_profile(df_track, district, building_type, area)
    group = df_track[(df_track['district_name'] == district) & (df_track['building_type'] == building_type)]
    profile['legal_dong'] = (group['district_name'] + ' | ' + group['legal_dong_name']).mode().iloc[0]
    return profile


def projected_path(recipe, df_track, months):
    """Monthly prediction of the fixed profile over `months` (a model fit before the first of them)."""
    frame = profile_frame(fixed_profile(df_track), months)
    return pd.Series(recipe.predict(frame, months), index=pd.DatetimeIndex(months))
