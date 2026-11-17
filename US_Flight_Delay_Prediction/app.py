"""
US Flight Delay Prediction: End-to-End Big Data Analytics Pipeline
Main Application Dashboard (Streamlit)
Author: Big Data & Machine Learning Engineer (CA3 Assessment)
"""

import streamlit as st
import os
import sys

# Ensure local source directory is available
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import SUPPORTED_KEYS, DELAY_THRESHOLD_MINUTES
from src.data_loader import find_dataset_source, get_file_info, load_flight_records
from src.preprocessing import clean_and_preprocess
from src.analytics import compute_overall_kpis
from src.mapreduce_interface import get_backend_status

st.set_page_config(
    page_title="US Flight Delay Prediction | Big Data Pipeline",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom header styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #0284c7 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size: 2.2rem;">✈️ US Flight Delay Prediction System</h1>
    <p style="margin: 6px 0 0 0; opacity: 0.9; font-size: 1.1rem;">
        An End-to-End Big Data Analytics Pipeline from Hadoop MapReduce to Machine Learning
    </p>
    <div style="margin-top: 10px; font-size: 0.85rem; opacity: 0.8;">
        Bureau of Transportation Statistics (BTS) / Kaggle Dataset
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar backend & environment indicator
st.sidebar.title("🛠️ System Control")
backend_info = get_backend_status()
selected_backend = st.sidebar.radio(
    "Active MapReduce Backend:",
    ["local", "hadoop"],
    format_func=lambda x: "⚡ Local Simulation (In-Memory)" if x == "local" else "🐘 Genuine Hadoop Streaming (HDFS/YARN)",
    help="Toggle between in-memory Python MapReduce simulation and authentic Hadoop Streaming cluster execution."
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Hadoop Status:** `{backend_info['hadoop']['status']}`")
st.sidebar.markdown(f"**Operational Threshold:** `{DELAY_THRESHOLD_MINUTES} min`")
st.sidebar.markdown("---")
st.sidebar.info("Use the sidebar navigation above to explore dataset attributes, MapReduce stages, machine learning evaluations, and interactive pre-departure risk prediction.")

# Ingest and cache data
@st.cache_data
def get_cached_flight_data(limit=25000):
    src = find_dataset_source()
    records, meta = load_flight_records(src, limit=limit)
    cleaned, quality = clean_and_preprocess(records)
    kpis = compute_overall_kpis(cleaned)
    return records, meta, cleaned, quality, kpis

with st.spinner("Loading and verifying dataset..."):
    records, meta, cleaned, quality, kpis = get_cached_flight_data()

# Overview KPI Cards
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Flights Loaded", f"{meta['loaded_rows']:,}", help=f"Source total: {meta['total_source_rows']:,}")
with col2:
    st.metric("Cleaned Completed", f"{quality['total_cleaned_records']:,}", f"{quality['retention_rate_pct']}% retention")
with col3:
    st.metric("Delayed Flights (>=15m)", f"{quality['delayed_flights']:,}", f"{quality['delay_rate_pct']}% delay rate")
with col4:
    st.metric("On-Time Flights", f"{quality['ontime_flights']:,}", f"{100 - quality['delay_rate_pct']:.1f}%")
with col5:
    st.metric("Avg Arrival Delay", f"{kpis['avg_delay_min']} min", f"Max: {kpis['max_delay_min']} min")

st.markdown("### 🏛️ Pipeline Architecture")
st.markdown("""
```text
+-----------------------+     +------------------------+     +------------------------+
| 1. Ingestion Layer   | --> | 2. Preprocessing & QA  | --> | 3. MapReduce Engine    |
| - BTS/Kaggle 32-col  |     | - Filter Cancelled/Div |     | - Mapper: Key -> Tuple |
| - Chunked Loader     |     | - Target: ARR_DELAY>=15|     | - Shuffle & Sort       |
| - archive (4).zip    |     | - Anti-Leakage Sandbox |     | - Reducer Aggregations |
+-----------------------+     +------------------------+     +------------------------+
                                                                         |
                                                                         v
+-----------------------+     +------------------------+     +------------------------+
| 6. Decision Support   | <-- | 5. ML Predictive Suite | <-- | 4. Big Data Analytics  |
| - Real-time Inference |     | - Logistic Regression  |     | - Airline Punctuality  |
| - Risk Class & Proba  |     | - Random Forest        |     | - Temporal Cascades    |
| - Business Recs       |     | - ROC-AUC & F1-Score   |     | - Airport Congestion   |
+-----------------------+     +------------------------+     +------------------------+
```
""")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📋 Dataset & Environment Status")
    st.markdown(f"- **Active Source:** `{meta['file_path']}`")
    st.markdown(f"- **File Size:** `{meta['file_size_mb']} MB`")
    st.markdown(f"- **Detected Date Coverage:** `{meta['date_coverage']}`")
    st.markdown(f"- **Schema Columns:** `32 Bureau of Transportation Statistics attributes`")
    st.markdown(f"- **Active MapReduce Mode:** `{'Local Python Simulation' if selected_backend == 'local' else 'Genuine Hadoop Cluster'}`")

with col_right:
    st.subheader("⚙️ Technology Stack")
    st.markdown("""
    - **Language:** Python 3.10+
    - **Big Data Engine:** Apache Hadoop 3.x with Hadoop Streaming (Ubuntu / WSL2)
    - **MapReduce Protocol:** Tab-separated Unix stream protocol (`sys.stdin` / `sys.stdout`)
    - **Data Processing:** Pure-Python associative aggregators & Pandas compatibility
    - **Machine Learning:** Scikit-learn Pipeline (Logistic Regression & Random Forest)
    - **Visualization & UI:** Streamlit Interactive Multi-Page Framework
    """)

st.success("System verified. Navigate to **1_Dataset_Explorer** or **3_MapReduce** to inspect stage-by-stage big data transformations.")
