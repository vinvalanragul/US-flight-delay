"""
Page 2: Preprocessing & Data Cleaning
"""
import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import find_dataset_source, load_flight_records
from src.preprocessing import clean_and_preprocess
from src.config import DELAY_THRESHOLD_MINUTES, FEATURE_COLUMNS

st.title("🧹 Data Preprocessing & Leakage Prevention")
st.markdown("Rigorous data hygiene, missing value resolution, non-completed flight filtering, and strict feature boundary enforcement.")

records, meta = load_flight_records(find_dataset_source(), limit=20000)
cleaned, quality = clean_and_preprocess(records)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Raw Ingested Rows", f"{quality['total_raw_records']:,}")
with col2:
    st.metric("Total Rows Excluded", f"{quality['total_removed']:,}")
with col3:
    st.metric("Completed Flights Retained", f"{quality['total_cleaned_records']:,}", f"{quality['retention_rate_pct']}% retention")
with col4:
    st.metric("Operational Threshold", f">= {DELAY_THRESHOLD_MINUTES} min")

st.subheader("Data Cleaning Audit Log")
audit_rows = [
    {
        "Filter Step": "Cancelled Flights (CANCELLED == 1.0)",
        "Records Dropped": f"{quality['removed_cancelled']:,}",
        "Engineering Justification": "Cancelled flights never reach gate arrival; arrival delay is undefined (NaN). Imputing synthetic arrival times would corrupt delay statistics."
    },
    {
        "Filter Step": "Diverted Flights (DIVERTED == 1.0)",
        "Records Dropped": f"{quality['removed_diverted']:,}",
        "Engineering Justification": "Diverted flights land at an alternative aerodrome; original schedule elapsed times and route distances are invalid."
    },
    {
        "Filter Step": "Missing Arrival Delay (ARR_DELAY is null/NaN)",
        "Records Dropped": f"{quality['removed_missing_arrival_delay']:,}",
        "Engineering Justification": "Ground telemetry was unrecorded or corrupted; cannot supervise binary classification target without valid ground truth."
    },
    {
        "Filter Step": "Duplicate Flight Signatures",
        "Records Dropped": f"{quality['removed_duplicates']:,}",
        "Engineering Justification": "Redundant identical airline-flight-date-route timestamps removed to prevent overfitting and sample inflation."
    }
]
st.table(audit_rows)

st.subheader("🛡️ Strict Data Leakage Audit")
st.markdown("""
A common flaw in academic flight delay projects is including `DEP_DELAY` (actual departure delay) in a pre-departure model.
Because a flight departing 60 minutes late will almost certainly arrive late, including `DEP_DELAY` creates an artificial 95%+ accuracy
that is useless for real passengers booking flights or airlines planning schedules.

Our pipeline separates features:
""")

c_left, c_right = st.columns(2)
with c_left:
    st.success("✅ **Preserved Pre-Departure Features (Allowed):**")
    for feat in FEATURE_COLUMNS:
        st.markdown(f"- `{feat}`")

with c_right:
    st.error("🚫 **Purged Post-Departure Features (Strictly Forbidden):**")
    leaked = ["DEP_DELAY", "ARR_DELAY", "TAXI_OUT", "WHEELS_OFF", "WHEELS_ON", "TAXI_IN", "ACTUAL_ELAPSED_TIME", "DELAY_DUE_CARRIER", "DELAY_DUE_WEATHER", "DELAY_DUE_NAS", "DELAY_DUE_LATE_AIRCRAFT"]
    for l in leaked:
        st.markdown(f"- `{l}`")

st.info(f"Target variable derived: **`is_delayed = 1`** when `ARR_DELAY >= {DELAY_THRESHOLD_MINUTES}` minutes, else **`0`**.")
