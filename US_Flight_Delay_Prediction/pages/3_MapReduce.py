"""
Page 3: MapReduce Processing Engine
"""
import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import SUPPORTED_KEYS
from src.data_loader import find_dataset_source, load_flight_records
from src.preprocessing import clean_and_preprocess
from src.mapreduce_interface import run_mapreduce, get_backend_status

st.title("⚙️ Hadoop MapReduce Workbench")
st.markdown("Execute associative MapReduce jobs locally or submit to genuine Hadoop Streaming clusters over HDFS.")

backend_status = get_backend_status()

col_b1, col_b2 = st.columns(2)
with col_b1:
    backend_choice = st.radio(
        "Choose Execution Backend:",
        ["local", "hadoop"],
        format_func=lambda x: "⚡ Local Simulation (In-Memory Python)" if x == "local" else "🐘 Genuine Hadoop Streaming (Ubuntu / WSL2)",
        horizontal=True
    )

with col_b2:
    group_key = st.selectbox(
        "Select MapReduce Grouping Key:",
        SUPPORTED_KEYS,
        index=0,
        help="The key used by Mapper to partition and group records during Shuffle & Sort."
    )

st.markdown("---")

records, _ = load_flight_records(find_dataset_source(), limit=20000)
cleaned, _ = clean_and_preprocess(records)

with st.spinner(f"Executing MapReduce job grouped by {group_key} using {backend_choice} backend..."):
    mr_output = run_mapreduce(cleaned, group_by_key=group_key, backend=backend_choice)

if backend_choice == "hadoop":
    st.subheader("🐘 Genuine Hadoop Verification Status")
    if not mr_output.get("success", False):
        st.warning(f"**Status:** `{mr_output['status']}`")
        st.markdown(f"**Diagnostic Details:** {mr_output['message']}")
        st.info("To run genuine Hadoop Streaming jobs, execute the cluster setup steps documented in **8_Hadoop_Documentation** inside Ubuntu/WSL2.")
        st.markdown(f"- **Target HDFS Input:** `{mr_output['hdfs_input_path']}`")
        st.markdown(f"- **Target HDFS Output:** `{mr_output['hdfs_output_path']}`")
    else:
        st.success(f"**Status:** `{mr_output['status']}`")
        st.code(mr_output.get("job_command", ""), language="bash")
        with st.expander("View Hadoop Job Logs"):
            st.text(mr_output.get("log", ""))

# Display Local Simulation 3-Stage Visualizer
st.subheader("🔬 MapReduce 3-Stage Workflow Inspection")

tab_map, tab_shuffle, tab_reduce = st.tabs(["1. Mapper Phase", "2. Shuffle & Sort Phase", "3. Reducer Output"])

with tab_map:
    st.markdown("**Mapper Logic:** Emits intermediate key-value pairs: `Key -> (1, ARR_DELAY, IS_DELAYED, MAX_DELAY)`")
    if "mapper_sample" in mr_output:
        st.write("First 8 intermediate records emitted by Mapper:")
        sample_tuples = [
            {"Key": p[0], "Emitted Tuple": f"(count={p[1][0]}, delay_sum={p[1][1]:.1f}, is_delayed={p[1][2]}, max_delay={p[1][3]:.1f})"}
            for p in mr_output["mapper_sample"][:8]
        ]
        st.table(sample_tuples)
    st.markdown(f"Total intermediate records emitted: **{mr_output.get('mapped_pairs_count', len(cleaned)):,}**")

with tab_shuffle:
    st.markdown("**Shuffle & Sort Logic:** Hadoop or the in-memory engine partitions intermediate records, groups values sharing identical keys, and lexicographically sorts unique keys.")
    st.markdown(f"Unique partition keys sorted: **{mr_output.get('shuffle_keys_count', len(mr_output.get('reduced_results', [])))}**")
    if "reduced_results" in mr_output:
        st.write("Keys partitioned into Reducer nodes:", [r["key"] for r in mr_output["reduced_results"]])

with tab_reduce:
    st.markdown("**Reducer Logic:** Associatively aggregates intermediate values to compute Total Flights, Delayed Flights, On-Time Flights, Average Delay, and Delay Rate (%).")
    results = mr_output.get("reduced_results", mr_output.get("records", []))
    if results:
        table_view = [
            {
                "Grouping Key": r["key"],
                "Total Flights": f"{r['total_flights']:,}",
                "Delayed Flights (>=15m)": f"{r['delayed_flights']:,}",
                "On-Time Flights": f"{r['ontime_flights']:,}",
                "Delay Rate (%)": f"{r['delay_rate_pct']:.2f}%",
                "Avg Delay (min)": f"{r['avg_delay_min']:.2f} m",
                "Max Delay (min)": f"{r['max_delay_min']:.1f} m"
            }
            for r in results
        ]
        st.dataframe(table_view, use_container_width=True)
        
        # Download Aggregated CSV
        import csv
        import io
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
        st.download_button(
            label="📥 Download MapReduce Aggregated CSV",
            data=buf.getvalue(),
            file_name=f"mapreduce_output_{group_key.lower()}.csv",
            mime="text/csv"
        )
