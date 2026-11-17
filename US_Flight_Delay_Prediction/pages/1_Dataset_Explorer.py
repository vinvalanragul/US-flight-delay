"""
Page 1: Dataset Explorer
"""
import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import find_dataset_source, load_flight_records

st.title("🔍 Dataset Explorer")
st.markdown("Inspect raw schema, flight distributions, column data types, and filter records.")

src = find_dataset_source()
records, meta = load_flight_records(src, limit=20000)

col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"**Loaded Sample:** {meta['loaded_rows']:,} rows")
with col2:
    st.info(f"**Total Detected Source:** {meta['total_source_rows']:,} rows")
with col3:
    st.info(f"**Coverage:** {meta['date_coverage']}")

st.subheader("Data Schema & Attribute Definitions (32 Columns)")
col_meta = [
    ("FL_DATE", "Date of flight (YYYY-MM-DD)", "Date / String"),
    ("AIRLINE", "Full commercial operating airline name", "Categorical"),
    ("AIRLINE_CODE", "Standard 2-letter IATA carrier code (AA, DL, UA, etc.)", "Categorical"),
    ("ORIGIN", "Origin aerodrome 3-letter IATA identifier", "Categorical"),
    ("DEST", "Destination aerodrome 3-letter IATA identifier", "Categorical"),
    ("CRS_DEP_TIME", "Scheduled gate departure time in local standard (HHMM)", "Numerical (Time)"),
    ("DEP_DELAY", "Actual departure delay in minutes (POST-DEPARTURE - LEAKAGE RISK)", "Numerical"),
    ("CRS_ARR_TIME", "Scheduled gate arrival time in local standard (HHMM)", "Numerical (Time)"),
    ("ARR_DELAY", "Actual gate arrival delay in minutes (Used to define TARGET)", "Numerical"),
    ("CANCELLED", "Flight cancelled indicator (1.0 = Cancelled, 0.0 = Completed)", "Binary"),
    ("DIVERTED", "Flight diverted indicator (1.0 = Diverted, 0.0 = Standard)", "Binary"),
    ("DISTANCE", "Great-circle geodesic distance between airports (miles)", "Numerical"),
    ("DELAY_DUE_*", "Post-flight delay attribution causes (Carrier, Weather, NAS, Late Aircraft)", "Numerical")
]

st.table([{"Attribute": c[0], "Description": c[1], "Data Type": c[2]} for c in col_meta])

# Interactive filter
airlines = sorted(list(set(r.get("AIRLINE", "") for r in records if r.get("AIRLINE"))))
selected_airline = st.selectbox("Filter records by Airline:", ["All Airlines"] + airlines)

filtered = [r for r in records if selected_airline == "All Airlines" or r.get("AIRLINE") == selected_airline]

st.subheader(f"Raw Records Preview ({len(filtered):,} matching flights)")
# Display sample records as table
sample_cols = ["FL_DATE", "AIRLINE", "ORIGIN", "DEST", "CRS_DEP_TIME", "DEP_DELAY", "CRS_ARR_TIME", "ARR_DELAY", "CANCELLED", "DISTANCE"]
table_data = []
for row in filtered[:25]:
    table_data.append({col: row.get(col, "") for col in sample_cols})
st.dataframe(table_data, use_container_width=True)
