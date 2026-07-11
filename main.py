"""One-command pipeline: clean -> features -> EDA -> train -> dashboard.

    python main.py            # run everything
    python main.py --no-eda   # skip figure generation
"""
from __future__ import annotations

import argparse

from src import (anomaly, apartment_forecasting, eda, eda_advanced,
                 feature_engineering as fe, forecasting, occupancy_ml,
                 preprocessing, revenue_ml, revenue_multivariate, segmentation,
                 train)


def run(do_eda: bool = True):
    print(">> cleaning")
    cleaned = preprocessing.clean_all()
    for name, df in cleaned.items():
        df.to_csv(preprocessing.config.OUT_DIR / f"clean_{name}.csv", index=False)

    print(">> features")
    feats = fe.build_all(cleaned)

    if do_eda:
        print(">> eda")
        eda.univariate(cleaned, feats)
        eda.bivariate(cleaned)
        eda.multivariate(feats)
        eda.trends(feats)
        print(">> advanced eda")
        eda_advanced.main()

    print(">> forecasting")
    forecasting.run()

    print(">> apartment-wise forecasting")
    apartment_forecasting.forecast_apartments()

    print(">> occupancy ML model (vs Holt-Winters)")
    occupancy_ml.run()

    print(">> ML revenue model (vs time-series)")
    revenue_ml.run()

    print(">> multivariate revenue model (revenue + occupancy)")
    revenue_multivariate.run()

    print(">> anomaly detection")
    anomaly.run()

    print(">> tenant segmentation")
    segmentation.run()

    print(">> training")
    train.main()

    print(">> dashboard")
    import dashboard
    dashboard.export_html()
    print("\nPIPELINE COMPLETE. See outputs/.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-eda", action="store_true")
    args = ap.parse_args()
    run(do_eda=not args.no_eda)
