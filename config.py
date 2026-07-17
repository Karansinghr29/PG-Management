
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
RAW_DIR = ROOT / "Data"         # new production datasets live in ./Data
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
# Only files that PHYSICALLY exist in the new Data/ folder are listed here.
# beds_snapshot, beds_catalog and notices no longer exist as files — they are
# DERIVED in preprocessing from the bookings dataset.
RAW_FILES = {
    "invoices":    "Supabase Snippet Untitled query (28).csv",  # billing fact table
    "bookings":    "Supabase Snippet Untitled query (22).csv",  # booking / stay history
    "electricity": "Supabase Snippet Untitled query (26).csv",  # EB meter readings
    "meters":      "Supabase Snippet Untitled query (26).csv",  # EB meter master (same file)
    "eb_bills":    "Supabase Snippet Untitled query (25).csv",  # EB bill payments
    "tickets":     "Supabase Snippet Untitled query (24).csv",  # maintenance tickets
    "assets":      "Supabase Snippet Untitled query (27).csv",  # physical assets
    "tenants":     "Supabase Snippet Untitled query (23).csv",  # tenant master (KYC/rating)
    "payments":    "Supabase Snippet Untitled query (29).csv",  # payments / receipts ledger
    "expenses":    "Supabase Snippet Untitled query (30).csv",  # expenses ledger
}

TOTAL_BEDS = 192   # physical bed capacity

# --------------------------------------------------------------------------- #
# Column roles (used by preprocessing)
# --------------------------------------------------------------------------- #
DATE_COLS = {
    "invoices": ["invoice_date", "due_date", "created_at"],  # billing_month -> period
    "bookings": ["onboarding_date", "estimated_exit_date", "actual_exit_date",
                 "notice_date"],
    "tickets": ["created_at", "sla_deadline", "resolved_at", "closed_at"],
    "assets": ["purchase_date", "warranty_expiry", "invoice_date"],
    "electricity": ["reading_date"],
    "eb_bills": ["bill_date", "payment_date", "billing_period_start",
                 "billing_period_end"],
    "payments": ["payment_date", "created_at"],
    "expenses": ["expense_date", "created_at"],
    "tenants": ["created_at", "date_of_joining", "rating_last_computed"],
    # derived tables:
    "notices": ["notice_date", "estimated_exit_date"],
}

# Columns dropped on load only when present (drop_dead_cols is null-safe).
DEAD_COLS = {
    "meters": ["eb_card_number", "eb_consumer_number", "eb_sanctioned_load"],
    "tickets": ["assigned_to"],
}

# Categorical text columns that need case normalisation.
CASE_NORMALISE = {
    "tickets": ["priority"],
    "bookings": ["staying_status"],
    "tenants": ["staying_status"],
}

# staying_status -> bed_lifecycle_status (used when deriving the bed snapshot).
STAYING_STATUS_MAP = {
    "staying": "occupied",
    "on-notice": "notice",
    "booked": "booked",
    "new": "booked",
    "exited": "vacant",
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
