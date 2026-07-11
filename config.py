"""Central configuration for the Vista Heights real-estate analytics project.

Single source of truth for paths, raw-file mapping, column roles and model
hyper-parameters. Every other module imports from here so nothing is hard-coded
twice.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT.parent                      # the folder holding the source CSVs
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"
REPORT_DIR = ROOT / "reports"

for _d in (OUT_DIR, FIG_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Raw-file mapping
#   The uploads are named "Supabase Snippet Untitled query (N).csv".
#   We give each a business name so the rest of the code is readable.
# --------------------------------------------------------------------------- #
RAW_FILES = {
    "invoices":     "Supabase Snippet Untitled query (6).csv",   # billing fact table
    "electricity":  "Supabase Snippet Untitled query (7).csv",   # apartment-month EB usage
    "beds_snapshot": "Supabase Snippet Untitled query (8).csv",  # current bed occupancy
    "notices":      "Supabase Snippet Untitled query (10).csv",  # exit notices
    "beds_catalog": "Supabase Snippet Untitled query (15).csv",  # bed inventory + pricing
    "assets":       "Supabase Snippet Untitled query (16).csv",  # physical assets
    "meters":       "Supabase Snippet Untitled query (17).csv",  # EB meter master
    "tickets":      "Supabase Snippet Untitled query (18).csv",  # maintenance tickets
    "bookings":     "Supabase Snippet Untitled query (20).csv",  # booking / stay history
}

TOTAL_BEDS = 192   # physical bed capacity (distinct bed_id in booking history)

# --------------------------------------------------------------------------- #
# Column roles (used by preprocessing / EDA)
# --------------------------------------------------------------------------- #
DATE_COLS = {
    "invoices": [],                       # billing_month is a period string
    "notices": ["notice_date", "estimated_exit_date"],
    "tickets": ["created_at", "sla_deadline", "resolved_at", "closed_at"],
    "assets": ["purchase_date", "warranty_expiry"],
    "bookings": ["booking_date", "onboarding_date", "estimated_exit_date",
                 "actual_exit_date", "notice_date"],
}

# Columns that are 100% null / constant in the raw data -> drop on load.
DEAD_COLS = {
    "assets": ["apartment_code", "warranty_expiry"],
    "meters": ["eb_card_number", "eb_consumer_number", "eb_sanctioned_load"],
    "tickets": ["assigned_to"],
    "bookings": ["expected_payment_date", "kyc_front_url", "kyc_back_url"],
}

# Categorical text columns that need case normalisation.
CASE_NORMALISE = {
    "beds_snapshot": ["gender_allowed"],
    "beds_catalog": ["gender_allowed"],
    "tickets": ["priority"],
}

# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Target definitions discovered from the data (see train.py).
TARGETS = {
    "late_payment": {          # classification
        "table": "invoices",
        "target": "is_unpaid",
        "type": "classification",
    },
    "monthly_revenue": {       # regression
        "table": "invoices",
        "target": "total_amount",
        "type": "regression",
    },
    "electricity_cost": {      # regression
        "table": "electricity",
        "target": "amount",
        "type": "regression",
    },
}

PROPERTY_NAME = "Vista Heights"
