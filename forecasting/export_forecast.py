"""Downloadable Excel forecast for every observed Market/product cell.

Export keeps all geographic levels separate, and uses the same automatic_forecast
recipe as the on-screen prediction. No artificial cross-product combinations.
No third-party Excel writer is required; XLSX is written as streamed OOXML.
"""
from __future__ import annotations

from io import BytesIO
from datetime import datetime, timezone
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED
import math
import numpy as np
import pandas as pd

MAX_EXCEL_ROWS = 1_048_576


def _column(number):
    label = ''
    while number:
        number, remainder = divmod(number - 1, 26)
        label = chr(65 + remainder) + label
    return label


def _cell(value, coordinate):
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('خروجی پیش‌بینی شامل مقدار عددی نامعتبر است.')
        return f'<c r="{coordinate}" t="n"><v>{value:.15g}</v></c>'
    # Inline strings cannot execute formulas; keep exact taxonomy labels.
    text = escape(str(value if value is not None else ''), {'"': '&quot;'})
    return f'<c r="{coordinate}" t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>'


def _sheet(z, filename, headers, records):
    if len(headers) > 16384:
        raise ValueError('تعداد ستون‌ها از محدودیت Excel بیشتر است.')
    with z.open(filename, 'w', force_zip64=True) as stream:
        stream.write(b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                     b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                     b'<sheetData>')
        for index, row in enumerate((headers, *records), start=1):
            if index > MAX_EXCEL_ROWS:
                raise ValueError('تعداد ردیف‌ها از محدودیت Excel بیشتر است.')
            if len(row) != len(headers):
                raise ValueError('تعداد ستون‌های خروجی با سرستون‌ها مطابقت ندارد.')
            xml = '<row r="' + str(index) + '">' + ''.join(
                _cell(v, f'{_column(j)}{index}') for j, v in enumerate(row, start=1)
            ) + '</row>'
            stream.write(xml.encode('utf-8'))
        stream.write(b'</sheetData></worksheet>')


def _forecast_rows(d, outputs):
    names = d.all_dimensions()
    for i, taxonomy in enumerate(d.meta):
        row = list(taxonomy)
        row.extend([float(d.volume[i, -1]), float(d.value[i, -1])])
        for metric in ('volume_kg', 'value_1000_toman'):
            for horizon in (1, 2, 3):
                row.append(float(outputs[metric][horizon - 1][i]))
        yield row


def _history_rows(d, metric):
    values = d.metric(metric)
    for i, taxonomy in enumerate(d.meta):
        yield list(taxonomy) + [float(value) for value in values[i]]


def build_forecast_workbook(dataset):
    """Return an XLSX with forecasts for ALL rows at ALL Market levels.

    Models are selected and fitted independently within each Market level,
    preventing All/City/Region double counting and cross-level leakage.
    """
    from auto_engine import automatic_forecast

    names = dataset.all_dimensions()
    market_index = names.index('Market') if 'Market' in names else None
    levels = (sorted(set(dataset.meta[:, market_index])) if market_index is not None
              else [None])
    forecasts = {metric: [np.zeros(len(dataset.meta), dtype=float) for _ in range(3)]
                 for metric in ('volume_kg', 'value_1000_toman')}
    methods = []
    for market in levels:
        if market is None:
            indices = np.arange(len(dataset.meta))
            segment = dataset
        else:
            indices = np.where(dataset.meta[:, market_index] == market)[0]
            segment = dataset.for_market(market)
        for metric in forecasts:
            for horizon in (1, 2, 3):
                values, recipe = automatic_forecast(segment, metric, horizon)
                if len(values) != len(indices) or not np.isfinite(values).all():
                    raise ValueError('مقادیر خروجی مدل برای ردیف‌های فایل معتبر نیستند.')
                forecasts[metric][horizon - 1][indices] = values
                target = str(pd.Period(dataset.months[-1], freq='M') +
                             horizon * dataset.period_months)
                methods.append([
                    market if market is not None else 'No Market column',
                    metric, horizon, target, recipe.get('name', ''),
                    recipe.get('validation_wape'),
                    ', '.join(recipe.get('validation_origins') or []),
                    str(recipe.get('components') or ''),
                    recipe.get('basis', ''),
                ])

    heads = list(names) + [
        'Last actual volume (' + dataset.volume_unit + ')',
        'Last actual value (' + dataset.value_unit + ')',
    ]
    targets = [str(pd.Period(dataset.months[-1], freq='M') +
                   h * dataset.period_months) for h in (1, 2, 3)]
    for metric, unit in [('Volume', dataset.volume_unit),
                         ('Value', dataset.value_unit)]:
        for horizon, target in enumerate(targets, 1):
            heads.append(f'{metric} Forecast H{horizon} {target} ({unit})')

    if len(dataset.meta) + 1 > MAX_EXCEL_ROWS:
        raise ValueError('تعداد سری‌های فروش بیشتر از محدودیت یک شیت Excel است.')

    metadata = [
        ['Source', dataset.source],
        ['Frequency', dataset.periodicity],
        ['Period months', dataset.period_months],
        ['Last observed period', dataset.months[-1]],
        ['Forecast periods', ', '.join(targets)],
        ['Number of unique sales series', len(dataset.meta)],
        ['Volume unit', dataset.volume_unit],
        ['Value unit', dataset.value_unit],
        ['Method', 'Each Market level is forecast independently; do not sum overlapping Market levels.'],
        ['Rows', 'Observed unique category combinations only; no artificial cartesian product.'],
        ['Validation', 'WAPE describes model selection on historical origins, not accuracy of each row.'],
        ['Generated UTC', datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')],
    ]
    sheets = [
        ('Forecast', heads, _forecast_rows(dataset, forecasts)),
        ('Volume History', list(names) + list(dataset.months),
         _history_rows(dataset, 'volume_kg')),
        ('Value History', list(names) + list(dataset.months),
         _history_rows(dataset, 'value_1000_toman')),
        ('Recipes', ['Market', 'Metric', 'Horizon', 'Target', 'Recipe',
                     'Validation WAPE', 'Validation origins', 'Weights',
                     'Selection basis'], methods),
        ('About', ['Property', 'Value'], metadata),
    ]
    result = BytesIO()
    with ZipFile(result, 'w', compression=ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr('[Content_Types].xml',
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            + ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                      for i in range(1, len(sheets) + 1)) + '</Types>')
        z.writestr('_rels/.rels',
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '</Relationships>')
        z.writestr('xl/workbook.xml',
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets>' + ''.join(
                f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>'
                for i, (name, _, _) in enumerate(sheets, 1)) + '</sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels',
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + ''.join(
                f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
                for i in range(1, len(sheets) + 1)) + '</Relationships>')
        for i, (_, headers, rows) in enumerate(sheets, 1):
            _sheet(z, f'xl/worksheets/sheet{i}.xml', headers, rows)
    return result.getvalue()
