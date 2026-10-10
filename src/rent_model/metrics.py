"""Evaluation metrics (rule 7 of the protocol). Inputs are on the original scale (10k KRW).

Percentage metrics are returned in percent. Metrics that divide by the true value (MdAPE, biases,
MAPE) skip rows with y <= 0.
"""
import numpy as np
import pandas as pd

from .config import RANDOM_SEED


def _arrays(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if y.shape != pred.shape:
        raise ValueError('y and pred must have the same shape')
    return y, pred


def wape(y, pred):
    """Weighted absolute percentage error: sum|y - pred| / sum|y| (percent)."""
    y, pred = _arrays(y, pred)
    return float(np.abs(y - pred).sum() / np.abs(y).sum() * 100)


def mae(y, pred):
    y, pred = _arrays(y, pred)
    return float(np.abs(y - pred).mean())


def _positive(y, pred):
    y, pred = _arrays(y, pred)
    keep = y > 0
    return y[keep], pred[keep]


def mdape(y, pred):
    """Median absolute percentage error (percent)."""
    y, pred = _positive(y, pred)
    return float(np.median(np.abs(pred - y) / y) * 100)


def median_bias_pct(y, pred):
    """Median of the signed percentage error (pred - y) / y (percent)."""
    y, pred = _positive(y, pred)
    return float(np.median((pred - y) / y) * 100)


def aggregate_bias_pct(y, pred):
    """(sum(pred) - sum(y)) / sum(y) (percent)."""
    y, pred = _arrays(y, pred)
    return float((pred.sum() - y.sum()) / y.sum() * 100)


def rmse_log(y, pred):
    """RMSE on the log1p scale."""
    y, pred = _arrays(y, pred)
    return float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(y)) ** 2)))


def r2_log(y, pred):
    """R2 on the log1p scale."""
    y, pred = _arrays(y, pred)
    ly, lp = np.log1p(y), np.log1p(pred)
    return float(1 - ((ly - lp) ** 2).sum() / ((ly - ly.mean()) ** 2).sum())


def mape_by_district_quarter(context, y, pred):
    """Mean absolute percentage error per district x quarter.

    `context` needs the columns district_name and contract_date (same row order as y).
    """
    y, pred = _arrays(y, pred)
    frame = pd.DataFrame({
        'district_name': np.asarray(context['district_name']),
        'quarter': pd.to_datetime(np.asarray(context['contract_date'])).to_period('Q').astype(str),
        'ape': np.abs(pred - y) / np.where(y > 0, y, np.nan) * 100,
    }).dropna()
    table = frame.groupby(['district_name', 'quarter'])['ape'].agg(n='size', mape_pct='mean')
    return table.reset_index()


def all_metrics(y, pred, context=None, min_cell_n=20):
    """Dictionary with every metric of the protocol.

    With `context`, adds the median and maximum MAPE over district x quarter cells that have at
    least `min_cell_n` rows.
    """
    result = {
        'n': int(len(y)),
        'wape_pct': wape(y, pred),
        'mae': mae(y, pred),
        'mdape_pct': mdape(y, pred),
        'median_bias_pct': median_bias_pct(y, pred),
        'aggregate_bias_pct': aggregate_bias_pct(y, pred),
        'rmse_log': rmse_log(y, pred),
        'r2_log': r2_log(y, pred),
    }
    if context is not None:
        cells = mape_by_district_quarter(context, y, pred)
        cells = cells[cells['n'] >= min_cell_n]
        result['mape_dq_cells'] = int(len(cells))
        result['mape_dq_median_pct'] = float(cells['mape_pct'].median())
        result['mape_dq_max_pct'] = float(cells['mape_pct'].max())
    return result


def paired_bootstrap_wape_diff(y, pred_a, pred_b, n=1000, seed=RANDOM_SEED):
    """Paired bootstrap of WAPE(a) - WAPE(b), in percentage points.

    Negative values mean that model a is better. Returns the point estimate, the bootstrap
    standard error (for the one-standard-error rule), a 95% interval and the share of resamples
    in which a beats b.
    """
    y, pred_a = _arrays(y, pred_a)
    _, pred_b = _arrays(y, pred_b)
    abs_a = np.abs(y - pred_a)
    abs_b = np.abs(y - pred_b)
    abs_y = np.abs(y)
    rng = np.random.default_rng(seed)
    size = len(y)

    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, size, size)
        denom = abs_y[idx].sum()
        diffs[i] = (abs_a[idx].sum() - abs_b[idx].sum()) / denom * 100

    return {
        'diff': wape(y, pred_a) - wape(y, pred_b),
        'se': float(diffs.std(ddof=1)),
        'ci_low': float(np.percentile(diffs, 2.5)),
        'ci_high': float(np.percentile(diffs, 97.5)),
        'prob_a_better': float((diffs < 0).mean()),
    }


def block_bootstrap_wape_diff(y, pred_a, pred_b, blocks, n=1000, seed=RANDOM_SEED):
    """Paired bootstrap of WAPE(a) - WAPE(b) that resamples whole blocks of rows.

    `blocks` assigns every row to a block (for example its contract month or its district). Rows of
    the same block are strongly correlated (shared market conditions, same buildings), so resampling
    blocks gives a larger, more honest standard error than resampling rows. Returns the same keys
    as `paired_bootstrap_wape_diff`.
    """
    y, pred_a = _arrays(y, pred_a)
    _, pred_b = _arrays(y, pred_b)
    codes, uniques = pd.factorize(np.asarray(blocks))
    n_blocks = len(uniques)
    sums = np.column_stack([np.bincount(codes, weights=w, minlength=n_blocks)
                            for w in (np.abs(y - pred_a), np.abs(y - pred_b), np.abs(y))])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_blocks, size=(n, n_blocks))
    total = sums[draws].sum(axis=1)
    diffs = (total[:, 0] - total[:, 1]) / total[:, 2] * 100
    return {
        'diff': wape(y, pred_a) - wape(y, pred_b),
        'se': float(diffs.std(ddof=1)),
        'ci_low': float(np.percentile(diffs, 2.5)),
        'ci_high': float(np.percentile(diffs, 97.5)),
        'prob_a_better': float((diffs < 0).mean()),
    }


def conservative_wape_diff(y, pred_a, pred_b, months, districts, n=1000, seed=RANDOM_SEED):
    """Paired WAPE difference with the most conservative of three standard errors.

    The standard errors come from the row bootstrap, the month-block bootstrap and the
    district-block bootstrap; the largest one is used (and its 95% interval is returned). Keys:
    diff, se, source ('row', 'month' or 'district'), ci_low, ci_high, se_row, se_month, se_district.
    """
    results = {
        'row': paired_bootstrap_wape_diff(y, pred_a, pred_b, n=n, seed=seed),
        'month': block_bootstrap_wape_diff(y, pred_a, pred_b, months, n=n, seed=seed),
        'district': block_bootstrap_wape_diff(y, pred_a, pred_b, districts, n=n, seed=seed),
    }
    source = max(results, key=lambda k: results[k]['se'])
    return {'diff': results[source]['diff'], 'se': results[source]['se'], 'source': source,
            'ci_low': results[source]['ci_low'], 'ci_high': results[source]['ci_high'],
            **{f'se_{k}': v['se'] for k, v in results.items()}}
