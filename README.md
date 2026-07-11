# Vista Heights — Real-Estate Analytics

End-to-end data-science project for a co-living / PG property ("Vista Heights"):
data cleaning, EDA, feature engineering, ML model comparison, and an executive
dashboard — all driven from 8 raw Supabase exports.

## Quickstart

```bash
pip install -r requirements.txt
python main.py                 # full pipeline: clean -> features -> EDA -> train -> dashboard
```

Individual stages:

```bash
python src/preprocessing.py        # write cleaned CSVs to outputs/
python src/feature_engineering.py  # write feature tables
python -m src.eda                  # plot pack -> outputs/figures/
python -m src.train                # train + compare all models -> outputs/leaderboard_*.csv
python -m src.train late_payment   # one problem only
python predict.py late_payment     # score with the saved best model
streamlit run dashboard.py         # interactive app  (or: python dashboard.py -> outputs/dashboard.html)
```

## Project structure

```
realestate_analytics/
├── config.py              # paths, raw-file map, column roles, model config
├── utils.py               # loaders, cleaning primitives, metrics
├── main.py                # one-command pipeline
├── predict.py             # load best model + score
├── dashboard.py           # Streamlit / Plotly dashboard (+ HTML fallback)
├── requirements.txt
├── src/
│   ├── preprocessing.py       # per-table cleaning (Step 2)
│   ├── feature_engineering.py # feature tables (Step 5)
│   ├── eda.py                 # plot pack (Step 3)
│   └── train.py               # ML + model comparison (Steps 6-7)
├── outputs/
│   ├── clean_*.csv  feat_*.csv
│   ├── figures/*.png
│   ├── models/*_best.pkl
│   └── leaderboard_*.csv  importance_*.csv  dashboard.html
└── reports/report.md      # full written report (Step 11)
```

## Data

Single property, 40 billing months (Apr-2023 → Jul-2026). Grain and keys are
documented in `reports/report.md`. The raw uploads stay in the parent folder;
`config.RAW_FILES` maps each to a business name.

## Prediction problems

**Validation is time-based** (chronological train/test split + `TimeSeriesSplit` CV)
for every time-dependent target — no random shuffling, no temporal leakage.

| Problem | Type | Target | Best model (time-based holdout) |
|---|---|---|---|
| Late payment | classification | `is_unpaid` | NaiveBayes — F1 0.69** |
| Monthly revenue | regression | `total_amount` | LinearRegression — R² 0.96 |
| Electricity cost | regression | `amount` | LinearRegression — R² 0.99* |

\* near-deterministic (amount ≈ units × unit_cost) — monitoring/anomaly baseline.
\*\* under a (wrong) random split boosted trees scored F1 0.90; the honest time-based
split drops to 0.69, exposing the 2026 collection-regime change the random split hid.
Linear models now win the regressions because trees cannot extrapolate beyond the
revenue levels seen in training — expected and correct under out-of-time evaluation.

XGBoost / LightGBM / CatBoost and SHAP are installed and included in the leaderboard;
`outputs/shap_*.csv` holds mean-|SHAP| feature importances.

## Beyond the core models

| Module | Output |
|---|---|
| `src/forecasting.py` | 6-month forecast of revenue / tenants / electricity; walk-forward method selection (**primary** forecaster) |
| `src/revenue_ml.py` | second revenue model: XGBoost/RF/Linear on lagged monthly features, walk-forward, compared vs the time-series model |
| `src/revenue_multivariate.py` | third model: revenue + **historical occupancy** (from booking dataset), corr r=0.983, feature importance + SHAP, ±5% occupancy scenario |
| `src/apartment_forecasting.py` | per-apartment electricity units + amount forecast (39 apartments) |
| `src/anomaly.py` | IsolationForest anomaly lists for electricity + invoices |
| `src/segmentation.py` | KMeans tenant segments + LTV (real per-`tenant_id` billing) |
| `src/eda_advanced.py` | seasonal decomposition, electricity heatmap, collection cohort, ticket analysis |

**Dashboard** (`streamlit run dashboard.py`) has 14 tabs. Analytics/ML: Executive
Summary · Revenue Forecast (time-series + ML) · Occupancy Forecast · Apartment-wise
Forecast · Late Payment Risk · Anomaly Alerts · Tenant Segmentation · ML Performance ·
Recommendations. Operational (from previously-unused datasets): Asset Management ·
Available Beds · Maintenance · Apartment Performance · Notice & Exit. Data-prep for the
operational tabs lives in `src/operational_analytics.py`; see
[reports/dataset_audit.md](reports/dataset_audit.md) for what each dataset supports and
why some requested metrics are not built. `python dashboard.py` writes a static
`outputs/dashboard.html` fallback.

**Not built (no real join exists):** tenant-exit, per-bed profitability, tenant↔maintenance,
historical bed occupancy, asset-location costing — each documented with its missing
join in [`reports/blocked_predictions.md`](reports/blocked_predictions.md). No data was
mocked to force these.
