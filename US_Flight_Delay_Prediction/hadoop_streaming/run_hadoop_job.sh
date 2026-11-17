#!/usr/bin/env bash
# ==============================================================================
# Hadoop Streaming Execution Script for US Flight Delay Analytics
# Environment: Ubuntu / Ubuntu WSL2 on Windows
# Author: Big Data Engineer & Hadoop Specialist
# ==============================================================================

set -euo pipefail

GROUP_KEY="${1:-AIRLINE}"
INPUT_CSV="${2:-data/flights_sample.csv}"
HDFS_INPUT_DIR="/user/hadoop/flight_delay/input"
HDFS_OUTPUT_DIR="/user/hadoop/flight_delay/output_$(echo "$GROUP_KEY" | tr '[:upper:]' '[:lower:]')"
LOCAL_OUTPUT_DIR="outputs/hadoop_$(echo "$GROUP_KEY" | tr '[:upper:]' '[:lower:]')"

echo "=========================================================="
echo " US Flight Delay Analytics — Hadoop Streaming Job Runner "
echo "=========================================================="
echo "Grouping Key:      $GROUP_KEY"
echo "Local Input:       $INPUT_CSV"
echo "HDFS Input Dir:    $HDFS_INPUT_DIR"
echo "HDFS Output Dir:   $HDFS_OUTPUT_DIR"
echo "Local Output Dir:  $LOCAL_OUTPUT_DIR"
echo "=========================================================="

# 1. Check Java Environment
echo "[Step 1/8] Verifying Java Installation..."
if ! command -v java >/dev/null 2>&1; then
    echo "ERROR: Java is not installed or not in PATH."
    echo "Please install OpenJDK 8 or 11: sudo apt update && sudo apt install -y openjdk-11-jdk"
    exit 1
fi
java -version 2>&1 | head -n 1
echo "Java OK."

# 2. Check Hadoop Environment
echo "[Step 2/8] Verifying Hadoop Installation..."
if ! command -v hadoop >/dev/null 2>&1; then
    echo "ERROR: 'hadoop' command not found."
    echo "Ensure HADOOP_HOME is set and \$HADOOP_HOME/bin is in your PATH."
    exit 1
fi
HADOOP_VER=$(hadoop version | head -n 1)
echo "Hadoop version: $HADOOP_VER"

# Locate Hadoop Streaming JAR dynamically
if [ -n "${HADOOP_HOME:-}" ] && [ -d "$HADOOP_HOME" ]; then
    STREAMING_JAR=$(find "$HADOOP_HOME" -name "hadoop-streaming*.jar" | head -n 1)
else
    STREAMING_JAR=$(find /usr -name "hadoop-streaming*.jar" 2>/dev/null | head -n 1 || true)
fi

if [ -z "$STREAMING_JAR" ] || [ ! -f "$STREAMING_JAR" ]; then
    echo "ERROR: Unable to locate hadoop-streaming*.jar in HADOOP_HOME ($HADOOP_HOME)."
    echo "Please verify your Hadoop installation or provide STREAMING_JAR explicitly."
    exit 1
fi
echo "Found Streaming JAR: $STREAMING_JAR"

# 3. Check HDFS Availability
echo "[Step 3/8] Checking HDFS Daemons..."
if ! hdfs dfs -ls / >/dev/null 2>&1; then
    echo "ERROR: Cannot reach HDFS NameNode."
    echo "Ensure Hadoop daemons are running: start-dfs.sh && start-yarn.sh"
    exit 1
fi
echo "HDFS NameNode is reachable."

# 4. Prepare HDFS Input Directory
echo "[Step 4/8] Preparing HDFS Input Directory..."
hdfs dfs -mkdir -p "$HDFS_INPUT_DIR"
FILENAME=$(basename "$INPUT_CSV")
echo "Uploading $INPUT_CSV to $HDFS_INPUT_DIR/$FILENAME ..."
hdfs dfs -put -f "$INPUT_CSV" "$HDFS_INPUT_DIR/$FILENAME"
echo "Input file successfully staged in HDFS."

# 5. Safe Cleanup of Existing HDFS Output Directory
echo "[Step 5/8] Checking HDFS Output Path Safety..."
if hdfs dfs -test -d "$HDFS_OUTPUT_DIR"; then
    echo "Notice: Output directory $HDFS_OUTPUT_DIR exists. Removing previous output safely..."
    hdfs dfs -rm -r -skipTrash "$HDFS_OUTPUT_DIR"
fi

# 6. Execute Hadoop Streaming Job
echo "[Step 6/8] Submitting Hadoop Streaming Job to YARN/Hadoop..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAPPER_PATH="$SCRIPT_DIR/mapper.py"
REDUCER_PATH="$SCRIPT_DIR/reducer.py"

chmod +x "$MAPPER_PATH" "$REDUCER_PATH"

hadoop jar "$STREAMING_JAR" \
    -D mapreduce.job.name="US_Flight_Delay_Analysis_${GROUP_KEY}" \
    -D mapreduce.job.reduces=2 \
    -cmdenv HADOOP_GROUP_KEY="$GROUP_KEY" \
    -files "$MAPPER_PATH","$REDUCER_PATH" \
    -input "$HDFS_INPUT_DIR/$FILENAME" \
    -output "$HDFS_OUTPUT_DIR" \
    -mapper "python3 mapper.py" \
    -reducer "python3 reducer.py"

echo "Hadoop Streaming job completed successfully!"

# 7. Retrieve Output from HDFS
echo "[Step 7/8] Fetching output from HDFS to local storage..."
mkdir -p "$LOCAL_OUTPUT_DIR"
hdfs dfs -getmerge "$HDFS_OUTPUT_DIR" "$LOCAL_OUTPUT_DIR/hadoop_result.tsv"
echo "Output saved locally at: $LOCAL_OUTPUT_DIR/hadoop_result.tsv"

# 8. Display Output Summary
echo "[Step 8/8] Sample Output Preview:"
echo "----------------------------------------------------------"
head -n 15 "$LOCAL_OUTPUT_DIR/hadoop_result.tsv"
echo "----------------------------------------------------------"
echo "Job execution and verification completed successfully."
