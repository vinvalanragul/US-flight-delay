"""
Data Preprocessing & Cleaning Pipeline for US Flight Delay Prediction.
Adheres strictly to the operational target definition:
- is_delayed = 1 if ARR_DELAY >= 15.0 minutes, else 0
- Filters out CANCELLED (1.0) and DIVERTED (1.0) flights
- Prevents Data Leakage by isolating pre-departure features only
- Produces before-and-after data quality metrics
"""

from typing import List, Dict, Any, Tuple
from datetime import datetime
from src.config import DELAY_THRESHOLD_MINUTES, FEATURE_COLUMNS, TARGET_COLUMN

def clean_and_preprocess(
    records: List[Dict[str, Any]],
    output_clean_csv: str = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Cleans raw flight records and constructs the machine-learning ready dataset.
    Returns (cleaned_records, quality_report).
    """
    total_raw = len(records)
    cancelled_count = 0
    diverted_count = 0
    missing_arr_delay_count = 0
    invalid_date_or_time_count = 0
    duplicates_count = 0
    
    seen_flight_signatures = set()
    cleaned = []
    
    for row in records:
        # Check duplicate by flight signature (Date, Airline, FlightNum, Origin, Dest, SchedDepTime)
        sig = (
            row.get("FL_DATE", ""),
            row.get("AIRLINE_CODE", row.get("AIRLINE", "")),
            row.get("FL_NUMBER", ""),
            row.get("ORIGIN", ""),
            row.get("DEST", ""),
            row.get("CRS_DEP_TIME", "")
        )
        if sig in seen_flight_signatures and any(sig):
            duplicates_count += 1
            continue
        seen_flight_signatures.add(sig)
        
        # 1. Filter out Cancelled flights
        # Justification: Cancelled flights do not arrive; their arrival delay is undefined.
        is_canc = row.get("CANCELLED", "0.0")
        if is_canc in ["1.0", "1", "True", True]:
            cancelled_count += 1
            continue
            
        # 2. Filter out Diverted flights
        # Justification: Diverted flights land at an alternative airport, so original arrival delay is not standard.
        is_div = row.get("DIVERTED", "0.0")
        if is_div in ["1.0", "1", "True", True]:
            diverted_count += 1
            continue
            
        # 3. Check ARR_DELAY availability
        arr_delay_str = row.get("ARR_DELAY", "").strip()
        if arr_delay_str == "" or arr_delay_str.lower() in ["nan", "null", "none"]:
            missing_arr_delay_count += 1
            continue
            
        try:
            arr_delay_val = float(arr_delay_str)
        except ValueError:
            missing_arr_delay_count += 1
            continue
            
        # 4. Parse Date and Time safely
        fl_date_str = row.get("FL_DATE", "").strip()
        month_val = 1
        day_of_week_val = 1
        try:
            dt = datetime.strptime(fl_date_str, "%Y-%m-%d")
            month_val = dt.month
            # Weekday: 1=Monday, 7=Sunday
            day_of_week_val = dt.isoweekday()
        except Exception:
            try:
                dt = datetime.strptime(fl_date_str.split()[0], "%m/%d/%Y")
                month_val = dt.month
                day_of_week_val = dt.isoweekday()
            except Exception:
                invalid_date_or_time_count += 1
                continue
                
        # Scheduled Departure Hour
        crs_dep_time_str = str(row.get("CRS_DEP_TIME", "0")).split(".")[0].zfill(4)
        try:
            crs_dep_hour = int(crs_dep_time_str[:2])
            if crs_dep_hour < 0 or crs_dep_hour > 23:
                crs_dep_hour = 12
        except Exception:
            crs_dep_hour = 12
            
        # Scheduled Arrival Hour
        crs_arr_time_str = str(row.get("CRS_ARR_TIME", "0")).split(".")[0].zfill(4)
        try:
            crs_arr_hour = int(crs_arr_time_str[:2])
            if crs_arr_hour < 0 or crs_arr_hour > 23:
                crs_arr_hour = 14
        except Exception:
            crs_arr_hour = 14
            
        # Distance
        dist_str = str(row.get("DISTANCE", "500")).split(".")[0]
        try:
            distance_val = float(dist_str)
        except Exception:
            distance_val = 500.0
            
        # Binary target definition: 1 if ARR_DELAY >= 15.0 else 0
        is_delayed = 1 if arr_delay_val >= DELAY_THRESHOLD_MINUTES else 0
        
        airline = row.get("AIRLINE", "Unknown")
        origin = row.get("ORIGIN", "UNK")
        dest = row.get("DEST", "UNK")
        
        # Build clean record containing pre-departure features + target + raw metadata
        cleaned_record = {
            "FL_DATE": fl_date_str,
            "AIRLINE": airline,
            "ORIGIN": origin,
            "DEST": dest,
            "MONTH": month_val,
            "DAY_OF_WEEK": day_of_week_val,
            "CRS_DEP_HOUR": crs_dep_hour,
            "CRS_ARR_HOUR": crs_arr_hour,
            "DISTANCE": distance_val,
            "ARR_DELAY": arr_delay_val,
            TARGET_COLUMN: is_delayed
        }
        cleaned.append(cleaned_record)
        
    total_cleaned = len(cleaned)
    delayed_count = sum(1 for r in cleaned if r[TARGET_COLUMN] == 1)
    ontime_count = total_cleaned - delayed_count
    delay_rate = (delayed_count / total_cleaned * 100) if total_cleaned > 0 else 0.0
    
    quality_report = {
        "total_raw_records": total_raw,
        "removed_cancelled": cancelled_count,
        "removed_diverted": diverted_count,
        "removed_missing_arrival_delay": missing_arr_delay_count,
        "removed_duplicates": duplicates_count,
        "removed_invalid_format": invalid_date_or_time_count,
        "total_removed": total_raw - total_cleaned,
        "retention_rate_pct": round((total_cleaned / total_raw * 100) if total_raw > 0 else 0, 2),
        "total_cleaned_records": total_cleaned,
        "delayed_flights": delayed_count,
        "ontime_flights": ontime_count,
        "delay_rate_pct": round(delay_rate, 2),
        "target_threshold_minutes": DELAY_THRESHOLD_MINUTES
    }
    
    if output_clean_csv:
        import csv
        import os
        os.makedirs(os.path.dirname(os.path.abspath(output_clean_csv)), exist_ok=True)
        with open(output_clean_csv, "w", newline="", encoding="utf-8") as f:
            if cleaned:
                writer = csv.DictWriter(f, fieldnames=list(cleaned[0].keys()))
                writer.writeheader()
                writer.writerows(cleaned)
                
    return cleaned, quality_report

def extract_features_and_target(
    cleaned_records: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Extracts strictly pre-departure feature dictionaries and the binary target list.
    Enforces data leakage prevention.
    """
    X = []
    y = []
    for r in cleaned_records:
        feat = {col: r[col] for col in FEATURE_COLUMNS if col in r}
        X.append(feat)
        y.append(int(r[TARGET_COLUMN]))
    return X, y
