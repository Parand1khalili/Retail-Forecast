"""In-app, local-only backtesting of the current EMRC forecasting policies.

Replays the production policies at multiple historical cutoffs. The source
workbook is not modified, raw observations are not returned to the browser,
and temporary CSV files are deleted after an in-memory ZIP is created.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from heapq import heappush, heapreplace
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from nested_backtest import report

METRICS = ('volume_kg', 'value_1000_toman')
MAX_PREVIEW_PER_LEVEL = 15
PREVIEW_FIELDS = (
    'metric', 'horizon', 'level', 'group', 'series', 'actual',
    'global_wape', 'hierarchical_wape', 'legacy_global_wape',
    'relative_reduction_vs_legacy', 'improvement_vs_legacy',
    'global_mae', 'hierarchical_mae', 'global_bias', 'hierarchical_bias',
    'baseline_wape', 'relative_reduction_vs_baseline', 'improvement_vs_baseline',
    'origin_count', 'live_legacy_global_wape', 'live_global_wape', 'live_hierarchical_wape',
    'live_relative_reduction_vs_legacy', 'live_improvement_vs_legacy',
)


def _top_groups(path, metric, previews, counters, tallies):
    if not path.exists():
        return
    with path.open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            key = (metric, row['horizon'], row['level'])
            stats = tallies[key]
            stats['groups'] += 1
            result = row.get('improvement_vs_baseline') or row.get('improvement_vs_legacy', 'unchanged')
            if result in ('improved', 'deteriorated', 'unchanged'):
                stats[result] += 1
            if row['level'] == 'Series':
                # Keep every Series in the downloaded CSV, not the browser.
                continue
            counters[key] += 1
            try:
                volume = float(row['actual'])
            except (ValueError, TypeError):
                volume = 0.0
            visible = {field: row.get(field) for field in PREVIEW_FIELDS}
            visible['metric'] = metric
            order = (volume, counters[key], visible)
            heap = previews[key]
            if len(heap) < MAX_PREVIEW_PER_LEVEL:
                heappush(heap, order)
            elif volume > heap[0][0]:
                heapreplace(heap, order)


def run_backtest(dataset, metrics):
    """Return (public dashboard data, downloadable ZIP bytes).

    Historical selection and evaluation are performed by the existing engine;
    no new winner is learned from the holdout data. Both selected metrics
    operate on one immutable, disjoint Market / report-grain dataset.
    """
    if not isinstance(metrics, (list, tuple)) or not metrics or len(set(metrics)) != len(metrics):
        raise ValueError('حداقل یک شاخص فروش معتبر و غیرتکراری انتخاب کنید.')
    if any(metric not in METRICS for metric in metrics):
        raise ValueError('شاخص فروش نامعتبر است.')
    if len(dataset.months) < 18:
        raise ValueError('برای آزمون مدل‌ها حداقل ۱۸ دوره تاریخی لازم است؛ پیش‌بینی کوتاه‌مدت همچنان در دسترس است.')
    dims = dataset.all_dimensions()
    if 'Market' in dims and len(set(dataset.meta[:, dims.index('Market')])) != 1:
        raise ValueError('برای مقایسه منصفانه ابتدا یک سطح Market انتخاب کنید.')
    if len(dataset.report_grain_levels()) > 1:
        raise ValueError('برای آزمون، ابتدا یکی از سطوح تجمیع SKU/Product را انتخاب کنید.')

    summaries = []
    previews = defaultdict(list)
    counters = defaultdict(int)
    tallies = defaultdict(lambda: {'groups': 0, 'improved': 0,
                                   'deteriorated': 0, 'unchanged': 0})
    archive = BytesIO()
    with TemporaryDirectory(prefix='emrc_backtest_') as folder:
        root = Path(folder)
        with ZipFile(archive, 'w', ZIP_DEFLATED, compresslevel=6) as output:
            for metric in metrics:
                summary_path = root / f'{metric}_summary.csv'
                details_path = root / f'{metric}_by_group.csv'
                rows = report(dataset, metric, summary_path, detailed_output=details_path)
                summaries.extend(rows)
                output.write(summary_path, f'{metric}_summary.csv')
                origin_path = root / f'{metric}_by_origin.csv'
                if origin_path.exists():
                    output.write(origin_path, origin_path.name)
                if details_path.exists():
                    output.write(details_path, f'{metric}_by_group.csv')
                    _top_groups(details_path, metric, previews, counters, tallies)

    displayed = []
    group_totals = []
    for key in sorted(tallies):
        metric, horizon, level = key
        group_totals.append({'metric': metric, 'horizon': int(horizon),
                             'level': level, **tallies[key]})
        for _, _, row in sorted(previews[key], reverse=True):
            displayed.append(row)

    market = (str(dataset.meta[0, dims.index('Market')])
              if 'Market' in dims else 'بدون ستون Market')
    return {
        'source': dataset.source,
        'market': market,
        'periodicity': dataset.periodicity,
        'periods': len(dataset.months),
        'series': len(dataset.meta),
        'metrics': list(metrics),
        'summaries': summaries,
        'group_totals': group_totals,
        'group_preview': displayed,
        'preview_limit': MAX_PREVIEW_PER_LEVEL,
        'evaluation_type': 'nested_walk_forward' if any(row.get('audit_scope') == 'nested_walk_forward' for row in summaries) else 'locked_recipes',
        'holdout_available': any(row['audit_type'] in ('untouched_holdout', 'nested_walk_forward')
                                 for row in summaries),
        'description': 'گزارش کامل تمام SKU/Seriesها در ZIP موجود است؛ جدول صفحه فقط بزرگ‌ترین گروه‌ها را نشان می‌دهد.',
    }, archive.getvalue()
