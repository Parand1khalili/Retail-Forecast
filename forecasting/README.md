# Yoghurt retail-audit forecast experiment

The experiment predicts **June, July and August 2026 together**, using only numerical observations up to **May 2026**. The target is published volume in kilograms, not reconstructed calendar-month transactions or unconstrained demand.

## Scope and evaluation

- All companies, brands and subbrands; complete combinations of the four source subcategory fields.
- Training unit: city × company × brand × subbrand × subcategory combination. City sales are summed to national reporting levels. Region, channel, subcity and national-total rows are not added to city sales.
- History: March 2022–May 2026. The Jun26 sheet supplies history before December 2023; Aug 26 supplies the later history and held-out actuals.
- Current source taxonomy is used. This is not an archived, vintage-by-vintage reconstruction of classification changes.
- Literal zeros remain zeros. A zero means not observed in the audited market, not that the product has necessarily been discontinued or has no demand.
- All additive SKU rows are retained when forming groups. Duplicate descriptions are not deleted.

Five baselines and two CatBoost residual models are compared. CatBoost learns the change from the latest observed volume, with separate direct models for horizons 1, 2 and 3. Features use prior volume, derived historical unit prices, past parent-group sales and the known calendar. Future prices, distribution measures, Year, MAT and YTD columns are excluded.

The two CatBoost presets use the same modest settings, with one preset normalizing the target change by a past-only sales scale. Fixed 50/50 blends with the validation-selected baseline are also compared. The selection criterion is pooled WAPE at the national full-subcategory-combination level.

Validation origins are September 2025, December 2025 and February 2026. Each predicts three following monthly labels; the final validation target is May 2026. Method selection is persisted to `selection_locked.json` before final holdout predictions. Holdout actuals are used only after forecasts have been persisted to `predictions_locked.npz`. No tuning is performed on June–August results.

WAPE is sum of absolute forecast errors divided by sum of observed sales. It is reported separately for brand, subbrand, complete subcategory combination and city combination levels. A small total-market error can hide errors that cancel between brands or segments. Zero-sales rows still contribute forecast errors to WAPE.

The customer's usual publication delay is about two months. This experiment follows their request to forecast the next three **data labels after May**; it does not claim that May data were available at calendar end-May. Overlapping retail-audit windows and their underlying monthly conversion are not reconstructed.

## Running

Use the Python executable returned by the Codex workspace dependency loader. Extracted read-only source caches are under `analysis/`. Additional forecasting wheels are unpacked under `analysis/model_deps/`; dependency versions are recorded by wheel names.

Run `forecasting/backtest.py`, then run `forecasting/build_results.mjs` with the bundled Node executable. The Node builder uses the bundled artifact tool through the local `node_modules` junction. Source Excel workbooks are never written.

Final result workbook and detailed CSVs are saved under `outputs/yoghurt_backtest/`. Saved CatBoost models, the selection record, source cache fingerprints and the training audit support reproducibility. The pipeline can be reused for other categories after adapting their source mapping, volume unit and category fields; a separate model should first be evaluated for each category.
