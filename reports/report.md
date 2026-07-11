# Vista Heights — Real-Estate Analytics Report

*Prepared by: Data Science team · Property: Vista Heights (single co-living / PG asset) · Data window: Apr-2023 → Jul-2026*

---

## 1. Executive Summary

Vista Heights is a single co-living property billed monthly across ~190 beds. Over
the 40-month history the property generated **₹10.08 crore** of total billings from
6,675 invoices to 896 distinct tenants.

Three findings dominate:

1. **Collection has collapsed in 2026.** The non-payment rate held at ~4% through
   2023–2025 but jumped to **45% in 2026**, concentrating **₹1.15 crore of revenue
   at risk** (unpaid invoice value) almost entirely in the current year. This is the
   single most important business signal in the data and needs to be triaged as
   either (a) a genuine collections breakdown or (b) a labelling artefact where
   recent invoices are simply "not yet paid". Both are addressed below.
2. **The asset is growing but not full.** Average active tenants rose 130 → 183
   (2023 → 2025) and current physical occupancy is **70.3%** with **31 beds on
   notice** and **24 vacant** — meaning roughly 30% of bed capacity is idle or at
   risk of turning over.
3. **Operations are largely healthy.** SLA breach rate on 1,416 maintenance tickets
   is only 4.2%, but average resolution time is 116 hours and AC/electrical issues
   dominate the queue.

Predictive models were built for late payment, monthly revenue and electricity cost;
the late-payment model (ROC-AUC 0.97) is the one with real operational value.

---

## 2. Business Problem

The property operator wants to (a) protect revenue by catching non-payment and
tenant exits early, (b) understand occupancy and electricity cost drivers, and
(c) get a single analytical view across billing, beds, electricity, assets and
maintenance — data that currently lives in eight disconnected exports.

---

## 3. Dataset Description (Step 1 — Discovery)

Eight CSVs were discovered and profiled automatically. All belong to one property,
"Vista Heights".

| # | Business name | Rows | Cols | Grain | Primary key |
|---|---|---|---|---|---|
| 6 | **invoices** | 6,675 | 13 | tenant × billing-month | `invoice_id` |
| 7 | **electricity** | 1,376 | 8 | apartment × billing-month | (`apartment_code`,`billing_month`) |
| 8 | **beds_snapshot** | 192 | 9 | bed (current) | (`apartment_code`,`bed_code`) |
| 10 | **notices** | 25 | 8 | tenant on notice | `phone` / (name) |
| 15 | **beds_catalog** | 391 (359 unique) | 8 | bed listing | (`apartment_code`,`bed_code`) |
| 16 | **assets** | 1,700 | 12 | physical asset | `asset_code` |
| 17 | **meters** | 41 (40 unique) | 7 | apartment EB meter | `apartment_code` |
| 18 | **tickets** | 1,416 | 17 | maintenance ticket | `ticket_number` |

**Column roles** (representative):
- *Dates*: `notice_date`, `estimated_exit_date` (notices); `created_at`, `sla_deadline`,
  `resolved_at`, `closed_at` (tickets); `purchase_date` (assets). `billing_month` is a
  period string, not a date.
- *Numerical*: `rent_amount`, `electricity_amount`, `total_amount`, `credit_days`,
  `units_consumed`, `unit_cost`, `amount`, `monthly_rate`, `current_rate`, `purchase_price`.
- *Categorical*: `bed_type`, `toilet_type`, `gender_allowed`, `status`, `issue_type`,
  `priority`, `category`, `condition`.
- *Identifiers*: `invoice_id`, `tenant_id`, `property_id` (UUIDs); `apartment_code`,
  `bed_code`, `asset_code`, `ticket_number` (business codes).

### Entity-Relationship map

```
                         PROPERTY: Vista Heights
                                  |
        +-------------------------+--------------------------+
        |                         |                          |
   METERS (17)              BEDS_CATALOG (15)            ASSETS (16)
   apartment_code          apartment_code+bed_code       asset_code
        |                         |                     (apartment_code col
        |                         |                      is 100% NULL -> no join)
        |                  BEDS_SNAPSHOT (8)
        |                  apartment_code+bed_code
        |                         |
   ELECTRICITY (7)          TICKETS (18)            NOTICES (10)
   apartment_code+month     apartment_code+bed_code  apartment_code+bed_code
        |                         |                     +full_name/phone
        +-----------+-------------+
                    | (apartment_code — verified 100% join coverage)
                    |
              INVOICES (6)  <-- tenant_id, property_id (UUID)
                    ^
                    |  JOIN GAP: invoices carry NO apartment_code/bed_code,
                    |  and tenant_id (UUID) does not map to the tenant_name
                    |  / phone used elsewhere.
```

**Relationships that work:** `apartment_code` joins meters ↔ electricity ↔
beds_snapshot ↔ tickets ↔ notices at **100% coverage** (verified). `bed_code` refines
to the bed level.

**The critical integration gap:** `invoices` (the money table) is keyed on
`tenant_id`/`property_id` UUIDs and has **no apartment or bed code**, and the UUID
`tenant_id` cannot be resolved to the `full_name`/`phone` used in notices and tickets.
So billing cannot currently be tied to a specific bed, tenant identity or maintenance
history. Closing this gap (an ID crosswalk table) is the highest-value data-engineering
task and would unlock tenant-level profitability and churn analytics.

---

## 4. Data Quality (Step 2)

Every issue found, with the fix applied in `src/preprocessing.py`:

| Issue | Where | Detail | Fix |
|---|---|---|---|
| Duplicate rows | beds_catalog (32), meters (1) | exact-duplicate listings | `drop_duplicates()` |
| Inconsistent categories | `gender_allowed` (`Male`/`male`), `priority` (`high`/`High`) | case splits one category into two | lower-case + strip |
| Dead columns (100% null) | assets.`apartment_code`, assets.`warranty_expiry`, meters.`eb_card_number`, `eb_consumer_number`, tickets.`assigned_to` | no information | dropped |
| Constant column | meters.`eb_sanctioned_load` = 0 for all | zero variance | dropped |
| Mostly-null columns | assets.`serial_number` (85% null), `brand` (80%), `purchase_price` (82%); tickets.`tenant_phone` (98%), `issue_sub_type` (84%) | sparse | kept, flagged; not used as model features |
| Negative values | electricity.`units_consumed` (1), invoices.`electricity_amount` (1) | physically impossible | clipped to 0 |
| Arithmetic integrity | invoices: `total_amount ≠ rent+electricity+other` on **90 rows** | totals don't reconcile | recomputed total from components |
| Zero values (valid) | invoices.`rent_amount`==0 (139), `electricity_amount`==0 (1,089); electricity units==0 (146) | credit/vacant months — legitimate | retained |
| Outliers | electricity `units_consumed` up to 9,258 (mean 623); `amount` up to ₹1.06 L | real high-consumption apartments | retained, flagged via `deviation_from_avg` |
| Wrong dtype | `phone`, `eb_meter_number` read as int/float; date columns as strings | | dates parsed to UTC datetimes |

**Data-leakage / target-leakage review:**
- *Monthly revenue model*: `total_amount = rent + electricity + other` by construction,
  so the raw components are trivial leakage. The model therefore predicts revenue from
  **drivers** (calendar, tenant history, credit terms) with components excluded where
  they would leak; kept components are treated as a reconciliation baseline, not a
  forecast.
- *Electricity cost model*: `amount ≈ units_consumed × unit_cost` — near-deterministic.
  The high R² is expected and the model serves as an **anomaly/monitoring baseline**,
  not a genuine prediction.
- *Late-payment model*: `prior_unpaid` / `prior_invoices` are **legitimate lagged
  history** (known before the current invoice) — not leakage. **Temporal caveat:** the
  2026 spike in `is_unpaid` (45% vs 4%) strongly suggests recent invoices are labelled
  unpaid simply because they are *not yet due/paid*. Any production model must use a
  time-based split and an "invoice age" cut-off, or it will learn "recent = unpaid".

---

## 5. EDA (Step 3)

Generated by `src/eda.py` into `outputs/figures/` — univariate (histograms, boxplots,
pie charts), bivariate (violin of total by payment status, units-vs-amount scatter),
multivariate (correlation heatmap), and trend lines for revenue, collection rate,
active tenants, electricity cost and revenue-at-risk. Highlights:

- Revenue trends steadily upward, from ₹15.6 L (Jun-2023) to a peak **₹33.2 L
  (Jun-2026)**, tracking tenant growth.
- The units↔amount scatter is almost perfectly linear (two unit-cost tiers: ₹11.0 and
  ₹11.5), confirming the electricity determinism noted above.
- Revenue-at-risk bars are flat and small until 2026, then spike sharply.

---

## 6. Business Insights (Step 4)

| Insight | Value |
|---|---|
| Total billings (40 months) | **₹10.08 crore** |
| Revenue at risk (unpaid value) | **₹1.15 crore** (11.4% of billings) |
| Overall non-payment rate | 12.0% — but **2023–25 ≈ 4%, 2026 = 45%** |
| Best revenue month | Jun-2026 — ₹33.2 L, 195 active tenants |
| Worst revenue month | Jun-2023 — ₹15.6 L, 121 active tenants |
| Occupancy (current, physical beds) | 70.3% occupied · 31 on notice · 24 vacant |
| Average active tenants by year | 130 → 162 → 183 → 179 (2023→2026) |
| Average rent / average invoice total | ₹13,515 / ₹15,105 |
| Rent by bed type | Triple ₹13.5 K · Double ₹15.5 K · Single ₹19.5 K · Executive ₹25.5 K |
| Electricity spend (total) | ₹97.3 L · avg 623 units per apartment-month |
| Maintenance | 143 open / 1,416 tickets · 4.2% SLA breach · 116 h avg resolution |
| Top issue types | AC (279), Other (223), Electrical (169), Furniture (145), RO water (127) |
| Notices | 25 tenants on notice · avg notice period 40 days |

**Interpretation:**
- **Revenue leakage is the story.** ₹1.15 crore is unpaid and it is a 2026 phenomenon.
  If real, collections need immediate intervention; if a labelling artefact, the KPI
  and the model both need an invoice-age correction. Either way it must be resolved
  before any headline "12% default rate" is trusted.
- **Occupancy is the growth lever.** 30% of beds are idle or turning over; Executive
  and Single beds carry the highest rate, so filling/holding those has the largest
  revenue impact.
- **Ops are not the bottleneck** — low breach rate — but the 116-hour resolution mean
  and AC-issue dominance suggest a preventive-maintenance program on air-conditioning
  would cut the largest ticket category.
- **Seasonality**: revenue and tenant counts show mid-year peaks (Jun), consistent with
  academic-year intake cycles typical of PG housing.

---

## 7. Machine Learning (Steps 6–7)

Applicable prediction problems were identified automatically. Each was trained across a
13-algorithm bank (Linear/Logistic, Ridge, DecisionTree, RandomForest, ExtraTrees,
GradientBoosting, AdaBoost, KNN, NaiveBayes, SVM/SVR — plus XGBoost/LightGBM/CatBoost
when installed), 5-fold cross-validated, with feature importance persisted.

### Validation method — time-based (leakage-free)
All time-dependent targets now use a **chronological train/test split** (train on the
earliest 80% of billing months, test on the most recent 20%) and **`TimeSeriesSplit`**
cross-validation instead of random shuffling. This guarantees the model is never
trained on the future to predict the past.

**The switch is consequential.** Under a random split the late-payment model scored
F1 0.90; under the correct time-based split the best F1 is **0.69**. The gap is the
leakage the random split was hiding: the 2026 non-payment surge (45% vs ~4% in
2023–25) is largely unlearnable from earlier history. The time-based number is the
honest one and confirms the "2026 collection collapse" finding is a genuine regime
change, not noise. Out-of-time models show high precision / low recall — they keep
predicting "paid" from a mostly-paying past and miss the new defaults.

### 7.1 Late-payment prediction (classification) — **the valuable model**
- **Target:** `is_unpaid`
- **Features:** rent/electricity amounts, credit_days, calendar, tenant history
  (`prior_invoices`, `prior_unpaid`, `prior_unpaid_ratio`, `is_new_tenant`), rent growth.
- **Best model (time-based split):** NaiveBayes — **F1 0.69** (precision 0.76,
  recall 0.63). Boosted trees drop to F1 ≤ 0.33 out-of-time: they memorise the
  mostly-paying past and miss the new 2026 defaults, while NaiveBayes's simpler
  likelihoods generalise better across the regime shift.
- **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC + TimeSeriesSplit CV-F1.
- **Business impact:** rank current invoices by non-payment probability for a
  collections worklist against the ₹1.15 crore at risk (see dashboard "Late Payment
  Risk" tab — 161 of 175 latest-month invoices flagged high-risk).

### 7.2 Monthly-revenue prediction (regression)
- **Target:** `total_amount` · **Best (time-based):** LinearRegression R² 0.96,
  MAE ₹500.
- Under out-of-time evaluation linear models overtake trees: tree ensembles cannot
  extrapolate beyond the revenue levels seen in training, and 2026 amounts exceed the
  2023–25 range. Useful as a budgeting/reconciliation check.

### 7.3 Electricity-cost prediction (regression)
- **Target:** `amount` · **Best (time-based):** LinearRegression R² 0.9997.
- Deterministic (amount ≈ units × unit_cost); deployed as an **anomaly baseline** —
  large residuals flag meter or billing errors.

Model bank includes XGBoost, LightGBM and CatBoost (installed) alongside the
scikit-learn estimators; SHAP mean-|value| importances are exported per problem
(`outputs/shap_*.csv`). Full leaderboards: `outputs/leaderboard_*.csv`; tree/coef
importances: `outputs/importance_*.csv`; best models: `outputs/models/*_best.pkl`.

### 7.4 Time-series forecasting (real 40-month series) — `src/forecasting.py`
Three candidate methods (Holt-Winters, linear+seasonal, seasonal-naive) are compared
by **walk-forward / rolling-origin backtest** (expanding window, one month at a time)
and the lowest-MAPE method is selected per series — no leakage, each forecast sees only
prior data. Walk-forward results:
- **Revenue**: seasonal-naive, MAPE ≈ 3.9%, next-month ≈ ₹30.9 L.
- **Active tenants**: seasonal-naive, MAPE ≈ 6.2%, next-month ≈ 199.
- **Electricity cost**: Holt-Winters, MAPE ≈ 5.2%.
Outputs: `outputs/forecast_*.csv`, `outputs/forecast_summary.csv`,
`outputs/figures/forecast.png`.

### 7.4a Revenue: Time-Series vs Machine Learning — `src/revenue_ml.py`
A **second** revenue model complements (does not replace) the primary time-series
forecaster. It frames next-month revenue as supervised regression on the real
`property_month` panel using **only lagged + calendar features** known before month
*t*: lagged revenue (t-1, t-2, t-3, t-12), a *lagged* 3-month mean, lagged tenant
count, month and year. Deliberately **excluded as leakage**: `arpu` (= revenue /
active_tenants — contains the target), `revenue_roll3` and `revenue_mom` (pandas
windows include month *t*), and all contemporaneous same-month aggregates
(active_tenants, collection_rate, units, electricity_billed are unknown at forecast
time). Chronological expanding-window walk-forward, no random split.

XGBoost, RandomForest and LinearRegression were trained and the time-series model was
evaluated on **identical** walk-forward windows (12 test months):

| Model | MAPE | MAE | RMSE | R² |
|---|---|---|---|---|
| **LinearRegression (ML)** | **4.8%** | ₹1.41 L | ₹1.90 L | 0.24 |
| RandomForest | 5.7% | ₹1.65 L | ₹2.04 L | 0.13 |
| XGBoost | 6.1% | ₹1.79 L | ₹2.14 L | 0.04 |
| Time-Series (seasonal-naive) | 8.1% | ₹2.50 L | ₹3.22 L | −1.17 |

**Which is better, and why:** for one-month-ahead revenue the **ML linear
auto-regressive model wins** — last month's revenue plus recent trend track the strong
month-to-month persistence, whereas the seasonal-naive time-series compares against a
value 12 months old and misses the 2024→2026 growth. Linear beats XGBoost because ~28
monthly rows are far too few for gradient boosting (it overfits); the linear model
matches the true low complexity of the signal. **Caveat:** a few dozen monthly points
give wide error bars, so the time-series model remains the primary forecaster for
multi-month horizons and the ML model is a complementary one-step cross-check. Outputs:
`outputs/`leaderboard_revenue_ml.csv',`figures/revenue_model_comparison.png`.

### 7.4a-ii Multivariate revenue: revenue + occupancy — `src/revenue_multivariate.py`
A **new booking/stay dataset** (1,162 rows, real `onboarding_date`/`actual_exit_date`
per `bed_id`) unlocks genuine **historical monthly occupancy** — the earlier
"no bed-occupancy history" blocker is removed. `feature_engineering.occupancy_month`
computes occupied beds, vacant beds, occupancy %, active tenants, move-ins, move-outs,
notices and new bookings per month (denominator = 192 physical beds), joined into
`property_month` on billing month.

A third revenue model adds **lagged** occupancy drivers (occupancy %, active tenants,
avg rental, new bookings, move-outs, move-ins, notices — all shifted, leakage-free) on
top of the revenue-lag baseline, and is compared against the revenue-only model on
identical walk-forward windows:

| Feature set | Best model | MAPE | RMSE | R² |
|---|---|---|---|---|
| Revenue-only | LinearRegression | 4.81% | ₹1.90 L | 0.24 |
| **Revenue + Occupancy** | **Ridge** | **4.89%** | **₹1.72 L** | **0.38** |

**Finding:** occupancy % correlates with revenue at **r = 0.983** (p ≈ 8.9e-30), and
lagged occupancy is the **3rd-strongest feature** (after `rev_lag1`, `rev_lag12`) in
both permutation importance and SHAP. On 12-month one-step MAPE the two models are a
statistical tie (4.89% vs 4.81%), but the multivariate model **explains more variance
(R² 0.38 vs 0.24)** and — uniquely — supports **occupancy scenario analysis**: a +5%
occupancy lift raises next-month revenue ≈ ₹0.50 L, a −5% drop cuts it ≈ ₹0.50 L. The
manager's hypothesis (revenue depends on occupancy, not only past revenue) is confirmed;
the time-series model remains primary for multi-month horizons. Outputs:
`comparison_multivariate.csv`, `backtest_multivariate.csv`,
`occupancy_revenue_scatter.csv`, `perm_importance_multivariate.csv`,
`shap_multivariate.csv`, `model_meta_multivariate.json`,
`figures/revenue_multivariate.png`.

### 7.4b Apartment-wise forecasting — `src/apartment_forecasting.py`
The electricity table has a real apartment × month grain, so **electricity units and
billing amount are forecast per apartment** (39 apartments with ≥12 months history,
per-apartment method selection + backtest MAPE). Highest projected next-month spend:
B41 ₹16.0 K, C41 ₹15.4 K, A41 ₹14.2 K. Outputs:
`outputs/forecast_apartment_*.csv`, `outputs/figures/forecast_apartments.png`.
**Apartment-level RENT revenue is NOT forecastable** — invoices carry no
`apartment_code` and bed tables have no time history (see `blocked_predictions.md`).
Electricity `amount` is the only real apartment-level money series.

### 7.5 Anomaly detection — `src/anomaly.py`
IsolationForest (2% contamination) on real columns:
- **Electricity** (units, amount, deviation): 28 anomalies flagged → meter/billing
  review list.
- **Invoices** (rent, electricity, total, credit_days): 134 anomalies → mis-billing
  review list.
Outputs: `outputs/anomalies_*.csv`, `outputs/figures/anomalies.png`.

### 7.6 Tenant segmentation & LTV (real billing behaviour) — `src/segmentation.py`
KMeans on per-`tenant_id` aggregates (tenure, total billed, avg rent, unpaid ratio,
LTV) — built by grouping the invoices table on its own real key, **no cross-table
join**. Silhouette-selected k=2:
- **Segment 0 — transient/new** (757 tenants): ~5-month tenure, ₹58 K paid LTV,
  higher unpaid ratio (0.2).
- **Segment 1 — anchor/long-stay** (139 tenants): ~22-month tenure, **₹325 K paid
  LTV**, lower unpaid ratio (0.1).
The 139 anchor tenants (16% of base) drive a disproportionate share of lifetime
value — retention effort should concentrate here.
Outputs: `outputs/tenant_segments*.csv`, `outputs/figures/tenant_segments.png`.

### 7.7 Blocked prediction tasks
Five requested predictions (tenant-exit, per-bed profitability, tenant↔maintenance
behaviour, historical bed occupancy, asset-location costing) **cannot be built
without inventing data** and are documented — with the exact missing join and the
unblock condition for each — in `reports/blocked_predictions.md`. No mocked mappings
were created.

---

## 8. Evaluation

- Regression: RMSE, MAE, MAPE, R², cross-validated R².
- Classification: Accuracy, Precision, Recall, F1, ROC-AUC, cross-validated F1.
- Tree models dominate; linear/SVM lag on the non-linear targets. Feature importance is
  exported per problem (SHAP is auto-enabled when the `shap` package is installed).

---

## 9. Recommendations

1. **Triage the 2026 non-payment spike immediately** — confirm whether it is a
   collections failure or unpaid-but-not-due invoices; fix the KPI definition and add an
   invoice-age field.
2. **Build a tenant/property ID crosswalk** linking `invoices.tenant_id` to
   `apartment_code`/`bed_code` and tenant identity. This is the single highest-ROI data
   task — it unlocks churn, LTV and per-bed profitability.
3. **Deploy the late-payment model** with a time-split retrain and a collections
   worklist driven by predicted risk.
4. **Attack occupancy** — 30% of beds idle/on-notice; focus retention on high-rate
   Executive/Single beds.
5. **Preventive AC maintenance** — the largest ticket category; a quarterly service
   cycle should cut volume and the 116-hour resolution mean.
6. **Fix billing hygiene** — reconcile the 90 non-summing invoices at source.

---

## 10. Future Scope

- Exit/notice and occupancy forecasting once identities are linked.
- Tenant segmentation and lifetime-value modelling.
- Real-time dashboard fed from Supabase instead of static exports.
- Anomaly detection on electricity and billing residuals.
- Automated data-quality gate (Great Expectations) in the pipeline.

---

## 11. Reproducibility

`python main.py` runs the whole pipeline (clean → features → EDA → train → dashboard).
Code is modular and PEP8-compliant under `realestate_analytics/`; see `README.md`.
```
