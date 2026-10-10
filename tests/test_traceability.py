"""Traceability of the README numbers and static protocol checks of the model notebooks.

1. Every number in the results tables and key-results lines of README.md must equal the value in
   reports/metrics/leaderboard.csv (formatted as in the README) or in reports/metrics/ts_index_*.json.
2. Model notebooks 01..09 must use the temporal split (never a random split or a shuffle).
These tests are skipped, with a message, when the generated reports are missing.
"""
import json
import re

import pandas as pd
import pytest

from rent_model import config

README = config.ROOT / 'README.md'
LEADERBOARD = config.METRICS_DIR / 'leaderboard.csv'
NOTEBOOK_DIR = config.ROOT / 'notebooks'
TRACK_HEADINGS = {'jeonse': '### Jeonse (target: deposit)', 'wolse': '### Wolse (target: monthly rent)'}


@pytest.fixture(scope='module')
def board():
    if not config.METRICS_DIR.exists() or not LEADERBOARD.exists():
        pytest.skip('reports/metrics/leaderboard.csv is missing: run notebooks 01-09 (bash scripts/run_all.sh) to generate it')
    return pd.read_csv(LEADERBOARD)


@pytest.fixture(scope='module')
def readme():
    if not README.exists():
        pytest.skip('README.md is missing')
    return README.read_text(encoding='utf-8')


def section(text, heading):
    """Text of the README between `heading` and the next heading of the same or higher level."""
    start = text.index(heading)
    level = len(heading) - len(heading.lstrip('#'))
    rest = text[start + len(heading):]
    nxt = re.search(rf'^#{{1,{level}}} ', rest, re.M)
    return rest[:nxt.start()] if nxt else rest


def f2(value):
    return f'{value:.2f}'


def test_leaderboard_csv_is_consistent(board):
    valid = board[board['valid']]
    assert board.groupby('track')['model_id'].nunique().nunique() == 1
    assert (valid.groupby('track')['is_chosen_model'].sum() == 1).all()
    assert (valid.groupby('track')['is_unconstrained_leader'].sum() == 1).all()
    assert not board[~board['valid']]['is_chosen_model'].any()      # oracle runs never win


@pytest.mark.parametrize('track', ['jeonse', 'wolse'])
def test_readme_leaderboard_tables_match_csv(board, readme, track):
    valid = board[(board['track'] == track) & board['valid']].set_index('model_id')
    block = section(readme, TRACK_HEADINGS[track])
    row = re.compile(r'^\| (\d+) \| `(\w+)` \| ([\w/\- ]+?) \| (-?\d+\.\d\d) \| (-?\d+\.\d\d) \| (-?\d+\.\d\d) \| (.+?) \| (.*?) \|$', re.M)
    rows = row.findall(block)
    assert len(rows) >= 10, 'could not parse the README leaderboard table'
    for rank, model, family, wape, median_bias, aggregate_bias, _, note in rows:
        r = valid.loc[model]
        assert int(rank) == int(r['rank_valid']), f'{track} {model}: rank'
        assert family == r['family'], f'{track} {model}: family'
        assert wape == f2(r['wape_pct']), f'{track} {model}: WAPE {wape} vs {f2(r["wape_pct"])}'
        assert median_bias == f2(r['median_bias_pct']), f'{track} {model}: median bias'
        assert aggregate_bias == f2(r['aggregate_bias_pct']), f'{track} {model}: aggregate bias'
        assert (note == 'chosen') == bool(r['is_chosen_model']), f'{track} {model}: chosen flag'
        assert (note == 'best accuracy, cannot extrapolate') == bool(r['is_unconstrained_leader']), f'{track} {model}: leader flag'


def test_readme_family_table_matches_csv(board, readme):
    valid = board[board['valid']]
    block = section(readme, '### Supervised vs unsupervised')
    row = re.compile(r'^\| ([\w/\- ]+) \| `(\w+)` \| (\d+\.\d\d) \| `(\w+)` \| (\d+\.\d\d) \|$', re.M)
    rows = row.findall(block)
    assert len(rows) == 7, 'could not parse the README family table'
    for family, j_model, j_wape, w_model, w_wape in rows:
        for track, model, wape in (('jeonse', j_model, j_wape), ('wolse', w_model, w_wape)):
            sub = valid[(valid['track'] == track) & (valid['family'] == family)]
            best = sub.loc[sub['wape_pct'].idxmin()]
            assert model == best['model_id'], f'{family} {track}: best model'
            assert wape == f2(best['wape_pct']), f'{family} {track}: WAPE'


def test_readme_key_results_and_winner_match_csv(board, readme):
    valid = board[board['valid']]
    chosen = valid[valid['is_chosen_model']].set_index('track')
    leader = valid[valid['is_unconstrained_leader']].set_index('track')

    best = re.search(r'\*\*Best accuracy on 2025 \(held-out year\):\*\* `(\w+)` in both tracks, with a WAPE of (\d+\.\d\d)% for Jeonse and (\d+\.\d\d)% for Wolse', readme)
    assert best, 'key-results line "Best accuracy" not found'
    assert best.group(1) == leader.loc['jeonse', 'model_id'] == leader.loc['wolse', 'model_id']
    assert (best.group(2), best.group(3)) == (f2(leader.loc['jeonse', 'wape_pct']), f2(leader.loc['wolse', 'wape_pct']))

    pick = re.search(r'`(\w+)` for Jeonse \(WAPE (\d+\.\d\d)%\) and `(\w+)` for Wolse \(WAPE (\d+\.\d\d)%\)\. Requiring extrapolation costs (\d+\.\d{4}) \(Jeonse\) and (\d+\.\d{4}) \(Wolse\) WAPE points', readme)
    assert pick, 'key-results line "Chosen model" not found'
    assert (pick.group(1), pick.group(3)) == (chosen.loc['jeonse', 'model_id'], chosen.loc['wolse', 'model_id'])
    assert (pick.group(2), pick.group(4)) == (f2(chosen.loc['jeonse', 'wape_pct']), f2(chosen.loc['wolse', 'wape_pct']))
    for track, text in (('jeonse', pick.group(5)), ('wolse', pick.group(6))):
        assert text == f'{chosen.loc[track, "wape_pct"] - leader.loc[track, "wape_pct"]:.4f}', f'{track}: cost of requiring extrapolation'

    for track, label in (('jeonse', 'Jeonse'), ('wolse', 'Wolse')):
        c, lead = chosen.loc[track], leader.loc[track]
        line = re.search(rf'\*\*{label}: `(\w+)`\*\*.*?WAPE (\d+\.\d\d)%, median bias (-?\d+\.\d\d)%, aggregate bias (-?\d+\.\d\d)%\..*?leader `(\w+)` has (\d+\.\d\d)% \(difference (\d+\.\d{{4}}) points', readme, re.S)
        assert line, f'winner bullet for {label} not found'
        assert line.group(1) == c['model_id'] and line.group(5) == lead['model_id']
        assert (line.group(2), line.group(3), line.group(4)) == (f2(c['wape_pct']), f2(c['median_bias_pct']), f2(c['aggregate_bias_pct']))
        assert line.group(6) == f2(lead['wape_pct'])
        assert line.group(7) == f'{c["wape_pct"] - lead["wape_pct"]:.4f}'


def test_readme_run_counts_match_csv(board, readme):
    counts = re.search(r'\((\d+) models x 2 tracks = (\d+) runs; (\d+) valid and (\d+) oracle diagnostics excluded\)', readme)
    pitch = re.search(r'(\d+) valid models compared per track', readme)
    assert counts and pitch, 'run counts not found in the README'
    n_models, n_runs, n_valid, n_oracle = (int(x) for x in counts.groups())
    assert (n_models, n_runs) == (board['model_id'].nunique(), len(board))
    assert (n_valid, n_oracle) == (int(board['valid'].sum()), int((~board['valid']).sum()))
    assert int(pitch.group(1)) == int(board[board['valid']].groupby('track').size().iloc[0])


def test_readme_index_numbers_match_ts_index_reports(readme):
    paths = {t: config.METRICS_DIR / f'ts_index_{t}.json' for t in config.TRACKS}
    if not all(p.exists() for p in paths.values()):
        pytest.skip('reports/metrics/ts_index_*.json are missing: run notebook 07 to generate them')
    values = {}
    for track, path in paths.items():
        report = json.loads(path.read_text(encoding='utf-8'))
        last = sorted(report['index_full'])[-1]
        values[track] = (f2(report['index_full'][last]['index']), f2(report['raw_median_rebased'][last]))
    expected = (f'ends 2025 at {values["jeonse"][0]} (Jeonse) and {values["wolse"][0]} (Wolse), while the raw monthly median rebased '
                f'to 2022-01 reaches {values["jeonse"][1]} and {values["wolse"][1]}')
    assert expected in readme, f'README index sentence differs from the ts_index reports: {expected}'


# ---------------------------------------------------------------------------------------------
# Static protocol checks of the model notebooks
# ---------------------------------------------------------------------------------------------
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob('0[1-9]_*.ipynb'))
# Notebook 07 forecasts a price index, not a supervised model: its split is the `last_month` cut of the index fit
INDEX_ONLY = {'07_price_index_forecasting'}
RANDOM_SPLIT = re.compile(r'\b(train_test_split|ShuffleSplit|StratifiedShuffleSplit|KFold|GroupKFold|shuffle)\b')


def code_of(path):
    nb = json.loads(path.read_text(encoding='utf-8'))
    lines = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            lines += [re.sub(r'#.*$', '', line) for line in ''.join(cell['source']).split('\n')]
    return '\n'.join(lines)


def test_model_notebooks_are_present():
    if not NOTEBOOKS:
        pytest.skip('notebooks/01..09 are missing')
    assert len(NOTEBOOKS) == 9, f'expected 9 model notebooks (01..09), found {[p.name for p in NOTEBOOKS]}'


@pytest.mark.parametrize('path', NOTEBOOKS, ids=lambda p: p.stem)
def test_model_notebook_uses_the_temporal_split_and_no_random_split(path):
    code = code_of(path)
    assert not RANDOM_SPLIT.search(code), f'{path.name}: random split or shuffle found: {RANDOM_SPLIT.search(code).group(0)}'
    if path.stem in INDEX_ONLY:
        assert re.search(r'hedonic_index\([^)]*last_month=', code), f'{path.name}: the TRAIN-only index must use last_month'
    else:
        assert 'temporal_split(' in code, f'{path.name}: temporal_split is never called'
