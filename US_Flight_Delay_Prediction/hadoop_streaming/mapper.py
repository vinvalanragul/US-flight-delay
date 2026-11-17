#!/usr/bin/env python3
"""
Hadoop Streaming Mapper for US Flight Delay Prediction.
Reads CSV flight rows from sys.stdin and emits tab-delimited intermediate records:
<KEY>\t<COUNT>\t<ARR_DELAY>\t<IS_DELAYED>\t<MAX_DELAY>

Supported environment variable HADOOP_GROUP_KEY:
AIRLINE (default), ORIGIN, DEST, MONTH, CRS_DEP_HOUR
"""

import sys
import os
import csv

GROUP_KEY = os.environ.get("HADOOP_GROUP_KEY", "AIRLINE").upper()
DELAY_THRESHOLD = 15.0

def main():
    # Use standard csv reader over sys.stdin
    reader = csv.reader(sys.stdin)
    header = None
    col_idx = {}
    
    for row in reader:
        if not row:
            continue
            
        # Detect header row
        if header is None and ("FL_DATE" in row or "AIRLINE" in row or "ORIGIN" in row):
            header = [col.strip().upper() for col in row]
            col_idx = {name: i for i, name in enumerate(header)}
            continue
            
        if not col_idx:
            # Fallback if no header detected, assume standard schema
            continue
            
        try:
            # Check cancelled and diverted
            canc_idx = col_idx.get("CANCELLED", -1)
            if canc_idx != -1 and len(row) > canc_idx:
                if row[canc_idx].strip() in ["1.0", "1", "True"]:
                    continue
                    
            div_idx = col_idx.get("DIVERTED", -1)
            if div_idx != -1 and len(row) > div_idx:
                if row[div_idx].strip() in ["1.0", "1", "True"]:
                    continue
                    
            # Check arrival delay
            arr_del_idx = col_idx.get("ARR_DELAY", -1)
            if arr_del_idx == -1 or len(row) <= arr_del_idx:
                continue
            arr_del_str = row[arr_del_idx].strip()
            if not arr_del_str or arr_del_str.lower() in ["nan", "null", "none"]:
                continue
            arr_delay = float(arr_del_str)
            
            # Extract key
            if GROUP_KEY == "AIRLINE":
                key = row[col_idx["AIRLINE"]] if "AIRLINE" in col_idx and len(row) > col_idx["AIRLINE"] else "Unknown"
            elif GROUP_KEY == "ORIGIN":
                key = row[col_idx["ORIGIN"]] if "ORIGIN" in col_idx and len(row) > col_idx["ORIGIN"] else "UNK"
            elif GROUP_KEY == "DEST":
                key = row[col_idx["DEST"]] if "DEST" in col_idx and len(row) > col_idx["DEST"] else "UNK"
            elif GROUP_KEY == "MONTH":
                date_str = row[col_idx["FL_DATE"]] if "FL_DATE" in col_idx and len(row) > col_idx["FL_DATE"] else ""
                try:
                    parts = date_str.split("-")
                    key = f"Month-{int(parts[1]):02d}"
                except Exception:
                    key = "Month-Unknown"
            elif GROUP_KEY == "CRS_DEP_HOUR":
                dep_str = row[col_idx["CRS_DEP_TIME"]] if "CRS_DEP_TIME" in col_idx and len(row) > col_idx["CRS_DEP_TIME"] else "0"
                try:
                    h = int(dep_str.split(".")[0].zfill(4)[:2])
                    key = f"{h:02d}:00"
                except Exception:
                    key = "00:00"
            else:
                key = "Total"
                
            is_delayed = 1 if arr_delay >= DELAY_THRESHOLD else 0
            
            # Emit key and partial aggregates
            # Key\tCount\tDelaySum\tDelayedCount\tMaxDelay
            sys.stdout.write(f"{key}\t1\t{arr_delay:.2f}\t{is_delayed}\t{arr_delay:.2f}\n")
            
        except Exception:
            # Tolerant mapper skips malformed rows
            continue

if __name__ == "__main__":
    main()
