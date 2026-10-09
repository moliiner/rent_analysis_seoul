"""Persist predictions and metrics of one model run (rule 10 of the protocol)."""
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config
from .metrics import all_metrics

_MODEL_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')


def save_run(track, model_id, y_true, y_pred, meta=None, context=None, reports_dir=None):
    """Save one model run for one track.

    Writes reports/predictions/{track}__{model_id}.parquet (overwritten by a rerun of the same
    model) and appends a row to reports/metrics/{track}__{model_id}.json (a list of runs).
    `y_true` and `y_pred` are on the original scale. `context` is an optional DataFrame with
    district_name and contract_date, used for the MAPE by district x quarter and stored with the
    predictions. Returns the two paths.
    """
    if track not in config.TRACKS:
        raise ValueError(f'unknown track: {track}')
    if not _MODEL_ID.match(model_id) or '__' in model_id:
        raise ValueError(f'invalid model_id: {model_id!r}')

    base = config.REPORTS_DIR if reports_dir is None else reports_dir
    pred_dir, metrics_dir = base / 'predictions', base / 'metrics'
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if context is not None:
        context = context.reset_index(drop=True)

    predictions = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    if context is not None:
        predictions = pd.concat([context[['district_name', 'contract_date']], predictions], axis=1)
    pred_path = pred_dir / f'{track}__{model_id}.parquet'
    predictions.to_parquet(pred_path, index=False)

    row = {
        'track': track,
        'model_id': model_id,
        'run_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'random_seed': config.RANDOM_SEED,
        **all_metrics(y_true, y_pred, context=context),
        'meta': meta or {},
    }
    metrics_path = metrics_dir / f'{track}__{model_id}.json'
    rows = json.loads(metrics_path.read_text(encoding='utf-8')) if metrics_path.exists() else []
    rows.append(row)
    metrics_path.write_text(json.dumps(rows, indent=2, default=str), encoding='utf-8')
    return pred_path, metrics_path
