# Master import: runtime and regression verification

The reported old monthly-only error was reproduced via a binary bimonthly upload
to port 8765 while the checkout already matched origin/master. Two legacy Python
listeners were present (18956 and 23004). Their health response identified the
service but carried no source identity. Updating source files does not reload
already-imported Python modules.

`run_forecast.ps1` now prefers the project's actual `.venv` interpreter. Health
reports a startup source fingerprint, interpreter, imported module paths, project
root and process ID. Reusing a server requires the expected source identity and
interpreter. A stale server is restarted only when its project, PID and absolute
application command match. Unknown/legacy port owners require explicit inspection;
the launcher fails instead of silently using or killing them. HTML uses no-store.

The startup sample request must not override a user-selected import frequency or
an import already in progress. `check_ui.mjs` covers that ordering regression.

Parser diagnostics distinguish opposite-frequency headers, unmatched measures,
invalid numbers, partially missing periods and entirely unavailable periods.
An unavailable leading period may be omitted; a partially missing period is
rejected, not converted to zero. Interior gaps are rejected. Cache v5 invalidates
older parses that used weaker invalid-number validation.

The real locally available master was tested and reached numeric validation:
its first Jul-Aug 2020 period is partially missing. It remains rejected until
the source data are completed. No proprietary workbook is included or altered.

Synthetic fixtures cover 36 bimonthly observations, pre-Measure metadata,
Year/MAT/YTD exclusion, grams/thousand-tomans, multiple brands and geographic
levels, zeros, missing values, caching, HTTP upload/path imports and state
preservation. Existing tests cover monthly compatibility, manual mapping,
six-period seasonal features and completed-target validation. Real CatBoost
selection and inference run on a tiny synthetic panel.

Run from the project root with the same interpreter as the launcher:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s forecasting -p 'test_*.py' -v
.\.venv\Scripts\python.exe -m compileall -q forecasting
```

Also run `forecasting/check_ui.mjs` with Node, then launch twice to verify reuse.
All generated Excel/model/cache artifacts must remain outside Git.

## Strict Master integrity (cache v6)

Each imported workbook is checked before it can enter forecasting. Duplicate full source identities are rejected before any optional user-selected dimension projection. Volume/value rows must pair at the original grain. All selected sheets must have consistent units and dimensions; unknown units require explicit confirmation. Partial missing cells, conflicting overlapping periods, incomplete scope/series coverage and ambiguous mappings block import. Identical complete scope/period overlaps are counted only once. Separate Market and confirmed Element Caption scopes remain separate. No new zero-sales rows are fabricated for absent identities; provide explicit source zeros only when the business meaning is known. Fully unavailable periods are recorded; interior gaps are rejected. Non-sales measures, excluded dimensions, non-period columns and non-sales sheets are listed in the audit. The schema preview offers a downloadable JSON reconciliation of source and imported volume/value totals for each scope and period. No totals across overlapping Market/grain scopes are presented as market size. Existing caches are invalidated. The policy is intentionally conservative: revised snapshots or changing SKU coverage need a source-confirmed merge policy before automatic acceptance. A real Master benchmark is still required.
