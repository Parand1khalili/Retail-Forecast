"""Export every observed series with three future periods to a native Excel workbook.

The source Dataset is not modified. Every Market level gets its own global model
selection; this avoids training across overlapping All/City/Region aggregates.
The XLSX uses streaming OOXML with no Excel-specific dependency.
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED
import math

from auto_engine import automatic_forecast

EXCEL_MAX_ROWS = 1_048_576
EXCEL_MAX_COLUMNS = 16_384


def _text(value):
    return escape(str(value), {'"': '&quot;', "'": '&apos;'})


def _col(number):
    result = ''
    while number:
        number, rem = divmod(number - 1, 26)
        result = chr(65 + rem) + result
    return result


def _cell(column, row, value):
    ref = f'{_col(column)}{row}'
    if value is None:
        return ''
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError('Non-finite output would make the Excel workbook invalid.')
        return f'<c r="{ref}"><v>{value}</v></c>'
    # Inline strings: prevent spreadsheet formula injection, preserve literal labels.
    text = str(value)
    return f'<c r="{ref}" t="inlineStr"><is><t>{_text(text)}</t></is></c>'


def _period_label(end, frequency):
    if frequency != 'bimonthly':
        return end
    from pandas import Period
    finish = Period(end, freq='M')
    start = finish - 1
    return start.strftime('%b') + '-' + finish.strftime('%b') + ' ' + str(finish.year)


def _modelled_rows(dataset):
    """Return all original observed-series rows, with aligned 1..3 predictions."""
    import numpy as np

    dims = dataset.all_dimensions()
    count = len(dataset.meta)
    predictions = {'volume_kg': np.empty((count, 3)), 'value_1000_toman': np.empty((count, 3))}
    if 'Market' in dims:
        market_col = dims.index('Market')
        levels = list(dict.fromkeys(dataset.meta[:, market_col].tolist()))
    else:
        market_col = None
        levels = [None]

    for level in levels:
        rows = (np.flatnonzero(dataset.meta[:, market_col] == level)
                if market_col is not None else np.arange(count))
        selected = dataset.for_market(level) if market_col is not None else dataset
        for metric in predictions:
            for horizon in (1, 2, 3):
                forecast, _ = automatic_forecast(selected, metric, horizon)
                forecast = np.asarray(forecast, dtype=float)
                if forecast.shape != (len(rows),):
                    raise ValueError('Unexpected forecast vector shape.')
                if not np.isfinite(forecast).all() or (forecast < 0).any():
                    raise ValueError('Forecast contains invalid values.')
                predictions[metric][rows, horizon - 1] = forecast
    return predictions


def export_workbook(dataset, filename):
    """Write full observed+forecast volume/value matrices to filename."""
    import numpy as np
    from pandas import Period

    if len(dataset.months) < 2:
        raise ValueError('دست‌کم دو دوره معتبر برای پیش‌بینی لازم است.')
    dims = dataset.all_dimensions()
    count = len(dataset.meta)
    columns = len(dims) + len(dataset.months) + 3
    if count + 1 > EXCEL_MAX_ROWS:
        raise ValueError('تعداد سری‌ها از ظرفیت یک شیت Excel بیشتر است.')
    if columns > EXCEL_MAX_COLUMNS:
        raise ValueError('تعداد ستون‌ها از ظرفیت Excel بیشتر است.')

    future = [str(Period(dataset.months[-1], freq='M') + h * dataset.period_months)
              for h in (1, 2, 3)]
    observed = [_period_label(end, dataset.periodicity) for end in dataset.months]
    ahead = [_period_label(end, dataset.periodicity) for end in future]
    predicted = _modelled_rows(dataset)

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    sheets = [('Volume', dataset.volume, predicted['volume_kg'], dataset.volume_unit),
              ('Value', dataset.value, predicted['value_1000_toman'], dataset.value_unit)]

    with ZipFile(filename, 'w', compression=ZIP_DEFLATED, compresslevel=3) as z:
        z.writestr('[Content_Types].xml',
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '</Types>')
        z.writestr('_rels/.rels',
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '</Relationships>')
        z.writestr('xl/workbook.xml',
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Volume" sheetId="1" r:id="rId1"/>'
            '<sheet name="Value" sheetId="2" r:id="rId2"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels',
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>'
            '</Relationships>')
        for sheet_number, (_, history, forecast, unit) in enumerate(sheets, 1):
            headers = dims + [f'Actual | {label}' for label in observed] + [
                f'Forecast | {label}' for label in ahead]
            with z.open(f'xl/worksheets/sheet{sheet_number}.xml', 'w', force_zip64=True) as out:
                def write(text):
                    out.write(text.encode('utf-8'))
                write('<?xml version="1.0" encoding="UTF-8"?>'
                      '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                      '<sheetViews><sheetView workbookViewId="0">'
                      '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
                      '</sheetView></sheetViews><sheetData>')
                write('<row r="1">' + ''.join(_cell(i, 1, label)
                      for i, label in enumerate(headers, 1)) + '</row>')
                for index, meta in enumerate(dataset.meta):
                    row_number = index + 2
                    values = list(meta) + history[index].tolist() + forecast[index].tolist()
                    write(f'<row r="{row_number}">' + ''.join(
                        _cell(i, row_number, value) for i, value in enumerate(values, 1)
                    ) + '</row>')
                write('</sheetData><autoFilter ref="A1:' +
                      _col(len(headers)) + str(count + 1) +
                      '"/></worksheet>')
    return {'rows': count, 'future': future, 'sheets': ['Volume', 'Value']}
