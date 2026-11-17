# Hadoop Streaming Implementation Guide

## US Flight Delay Prediction: End-to-End Big Data Analytics Pipeline

This directory contains the production-grade **Hadoop Streaming** implementation for the US Flight Delay Prediction system.

### 1. MapReduce Protocol Specification

Hadoop Streaming uses Unix standard streams (`sys.stdin` and `sys.stdout`) with tab-separated key-value pairs.

#### A. Mapper (`mapper.py`)
- **Input**: Raw flight CSV records from `sys.stdin`.
- **Filtering**:
  - Drops rows where `CANCELLED == 1.0` or `DIVERTED == 1.0`.
  - Drops rows where `ARR_DELAY` is missing or non-numeric.
- **Grouping Keys**: Configurable via `HADOOP_GROUP_KEY` environment variable:
  - `AIRLINE`: Carrier full name (e.g. `American Airlines Inc.`)
  - `ORIGIN`: 3-letter origin airport IATA code (e.g. `ORD`, `ATL`)
  - `DEST`: 3-letter destination airport IATA code (e.g. `DFW`, `LAX`)
  - `MONTH`: 2-digit month string (e.g. `Month-07`)
  - `CRS_DEP_HOUR`: 2-digit scheduled departure hour (e.g. `16:00`)
- **Emitted Tuple**:
  ```text
  <KEY>\t1\t<ARR_DELAY>\t<IS_DELAYED>\t<MAX_DELAY>
  ```
  Where:
  - `1`: Count of flights (integer)
  - `<ARR_DELAY>`: Actual arrival delay in minutes (float)
  - `<IS_DELAYED>`: 1 if `ARR_DELAY >= 15.0`, else 0 (integer)
  - `<MAX_DELAY>`: Arrival delay in minutes for tracking extreme delays

#### B. Shuffle & Sort (Hadoop Internal Engine)
- Hadoop partitions and sorts records lexicographically by `<KEY>`.
- Guarantees that all records sharing the same `<KEY>` are streamed contiguously to the reducer.

#### C. Reducer (`reducer.py`)
- **Input**: Tab-delimited intermediate tuples grouped by key.
- **Aggregation Logic**:
  - Accumulates `TOTAL_FLIGHTS = sum(count)`
  - Accumulates `TOTAL_DELAY = sum(arr_delay)`
  - Accumulates `DELAYED_FLIGHTS = sum(is_delayed)`
  - Computes `ONTIME_FLIGHTS = TOTAL_FLIGHTS - DELAYED_FLIGHTS`
  - Computes `DELAY_RATE_PCT = (DELAYED_FLIGHTS / TOTAL_FLIGHTS) * 100`
  - Computes `AVG_DELAY_MIN = TOTAL_DELAY / TOTAL_FLIGHTS`
  - Tracks `MAX_DELAY_MIN = max(max_delay)`
- **Emitted Output**:
  ```text
  <KEY>\t<TOTAL_FLIGHTS>\t<DELAYED_FLIGHTS>\t<ONTIME_FLIGHTS>\t<DELAY_RATE_PCT>\t<AVG_DELAY_MIN>\t<MAX_DELAY_MIN>
  ```

---

### 2. Local Unix Pipeline Verification
You can verify the exact logic without a Hadoop cluster using standard Unix pipes:
```bash
cat data/flights_sample.csv | python3 hadoop_streaming/mapper.py | sort | python3 hadoop_streaming/reducer.py
```

---

### 3. Running on Ubuntu / WSL2 Hadoop Cluster
1. **Ensure Hadoop daemons are started**:
   ```bash
   start-dfs.sh
   start-yarn.sh
   jps
   ```
2. **Execute the automated runner**:
   ```bash
   ./hadoop_streaming/run_hadoop_job.sh AIRLINE data/flights_sample.csv
   ```
3. **Inspect the output**:
   ```bash
   hdfs dfs -cat /user/hadoop/flight_delay/output_airline/part-00000
   ```
