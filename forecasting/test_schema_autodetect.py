"""Synthetic end-to-end checks for dynamic pre-Measure categories.

No proprietary RA workbooks are needed. Run with:
    python -m unittest discover -s forecasting -p "test_*.py" -v
"""
from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

import app_core
from app_core import Dataset
from schema import SchemaError, detect_layout


def tiny_workbook(path, index_heading="Index Name"):
    headings = [
        "Market", "Location&Channel", "Element Description", "SKUSize",
        "SKUPackage", "Brand", index_heading, "Year-2022",
        "May' 2026", "Jun' 2026", "Jul' 2026",
    ]
    base = ["City", "Tehran", "Sparkling Water", None, "Can", "Aqua"]
    rows = [headings]
    for size, vol, money in [
        ("330 ml", [100, 110, 120], [1000, 1100, 1200]),
        ("500 ml", [60, 70, 80], [600, 700, 800]),
    ]:
        rows.append(base[:3] + [size] + base[4:] + ["Volume Sales (L)", "0"] + list(map(str, vol)))
        rows.append(base[:3] + [size] + base[4:] + ["Value Sales (Toman)", "0"] + list(map(str, money)))
    # The row below belongs to a distinct Market level and must remain separate.
    rows.append(["OutletType", "Kiosk", "Sparkling Water", "330 ml", "Can",
                 "Aqua", "Volume Sales (L)", "0", "999", "999", "999"])
    rows.append(["OutletType", "Kiosk", "Sparkling Water", "330 ml", "Can",
                 "Aqua", "Value Sales (Toman)", "0", "9999", "9999", "9999"])

    def cell(i, value):
        col = ""
        while i:
            i, digit = divmod(i-1, 26)
            col = chr(65+digit) + col
        text = escape(str(value))
        return f'<c r="{col}1" t="inlineStr"><is><t>{text}</t></is></c>'

    # The worksheet reader only inspects column letters, not the cell row index.
    xml_rows = "".join(
        f'<row r="{j}">' + "".join(cell(i+1, v) for i, v in enumerate(row))
        + '</row>' for j, row in enumerate(rows, 1)
    )
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{xml_rows}</sheetData></worksheet>'
    )
    workbook = (
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Test" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    relations = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Target="worksheets/sheet1.xml" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>'
        '</Relationships>'
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", relations)
        z.writestr("xl/worksheets/sheet1.xml", worksheet)


class DynamicCategoriesTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.path = self.root / "water.xlsx"
        tiny_workbook(self.path)
        patcher = patch.object(app_core, "CACHE", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_index_name_and_all_preceding_dimensions_are_preserved(self):
        d = Dataset.file(self.path)
        self.assertEqual(d.months, ["2026-05", "2026-06", "2026-07"])
        self.assertEqual(len(d.meta), 3)
        self.assertEqual(len(d.for_market('City').meta), 2)
        self.assertEqual(d.for_market('City').volume[:, -1].sum(), 200)
        self.assertEqual(d.for_market('City').value[:, -1].sum(), 2000)
        self.assertEqual(d.for_market('OutletType').volume[:, -1].sum(), 999)
        self.assertIn('Market', d.all_dimensions())
        for name in ["Element Description", "SKUSize", "SKUPackage"]:
            self.assertIn(name, d.all_dimensions())
            self.assertIn(name, d.options()["dimensions"])
        self.assertEqual(d.schema["measure_column"], "Index Name")
        self.assertEqual(len(d.schema["date_columns"]), 3)
        self.assertIn("Location&Channel", d.schema["dimension_columns"])
        self.assertEqual(d.volume_unit, "لیتر")
        self.assertEqual(d.value_unit, "تومان")
        # Cached metadata must preserve dynamic column names and the mapping UI.
        cached = Dataset.file(self.path)
        self.assertEqual(cached.all_dimensions(), d.all_dimensions())
        self.assertEqual(cached.schema, d.schema)

    def test_user_can_exclude_optional_category_column(self):
        keep = ["Location&Channel", "Element Description", "SKUPackage", "Brand"]
        d = Dataset.file(self.path, {"dimension_columns": keep})
        self.assertNotIn("SKUSize", d.all_dimensions())
        self.assertEqual(len(d.meta), 2)
        self.assertEqual(len(d.for_market('City').meta), 1)
        self.assertEqual(d.for_market('City').volume[0, -1], 200)

    def test_required_city_and_brand_cannot_be_excluded(self):
        with self.assertRaises(SchemaError):
            Dataset.file(self.path, {"dimension_columns": ["Brand", "SKUSize"]})

    def test_manual_index_heading_mapping(self):
        unknown = self.root / "unknown.xlsx"
        tiny_workbook(unknown, index_heading="Index Label")
        with self.assertRaises(SchemaError) as cm:
            Dataset.file(unknown)
        self.assertIn("Index Label", cm.exception.prompt["dimension_candidates"])
        d = Dataset.file(unknown, {"fields": {"Measure": "Index Label"}})
        self.assertEqual(d.months[-1], "2026-07")
        self.assertEqual(len(d.meta), 3)
        self.assertEqual(len(d.for_market('City').meta), 2)

    def test_detected_columns_reach_catboost_features(self):
        from auto_engine import Features

        d = Dataset.file(self.path)
        dates = pd.period_range("2025-01", periods=15, freq="M")
        d.months = dates.astype(str).tolist()
        d.volume = np.tile(d.volume[:, :1], (1, 15))
        d.value = np.tile(d.value[:, :1], (1, 15))
        features = Features(d, "volume_kg", 0, 14)
        X, _, _ = features.features(12, 1)
        self.assertIn("SKUSize", X.columns)
        self.assertIn("Element Description", X.columns)
        self.assertEqual(set(X["SKUSize"]), {"330 ml", "500 ml"})


if __name__ == "__main__":
    unittest.main()
