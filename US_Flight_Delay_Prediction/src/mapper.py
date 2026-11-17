"""
MapReduce Mapper for US Flight Delay Analytics.
Extracts the grouping key and emits intermediate key-value tuples.
Value format: (flight_count, arrival_delay_sum, delayed_count, max_delay)
"""

from typing import Dict, Any, Tuple, Optional
from src.config import DELAY_THRESHOLD_MINUTES

def map_flight_record(
    record: Dict[str, Any],
    group_by_key: str = "AIRLINE"
) -> Optional[Tuple[str, Tuple[int, float, int, float]]]:
    """
    Processes a single flight record and emits (key, (1, arr_delay, is_delayed, arr_delay)).
    Returns None if record is cancelled, diverted, or missing valid arrival delay.
    """
    # Exclude cancelled or diverted
    if str(record.get("CANCELLED", "0.0")) in ["1.0", "1", "True", True]:
        return None
    if str(record.get("DIVERTED", "0.0")) in ["1.0", "1", "True", True]:
        return None
        
    arr_delay_str = str(record.get("ARR_DELAY", "")).strip()
    if not arr_delay_str or arr_delay_str.lower() in ["nan", "null", "none"]:
        return None
        
    try:
        arr_delay = float(arr_delay_str)
    except ValueError:
        return None
        
    # Extract grouping key
    if group_by_key == "AIRLINE":
        key = record.get("AIRLINE", "Unknown")
    elif group_by_key == "ORIGIN":
        key = record.get("ORIGIN", "UNK")
    elif group_by_key == "DEST":
        key = record.get("DEST", "UNK")
    elif group_by_key == "MONTH":
        if "MONTH" in record:
            key = f"Month-{int(record['MONTH']):02d}"
        else:
            fl_date = str(record.get("FL_DATE", ""))
            try:
                parts = fl_date.split("-")
                key = f"Month-{int(parts[1]):02d}" if len(parts) >= 2 else "Month-Unknown"
            except Exception:
                key = "Month-Unknown"
    elif group_by_key == "CRS_DEP_HOUR":
        if "CRS_DEP_HOUR" in record:
            hour = int(record["CRS_DEP_HOUR"])
        else:
            time_str = str(record.get("CRS_DEP_TIME", "0")).split(".")[0].zfill(4)
            try:
                hour = int(time_str[:2])
            except Exception:
                hour = 0
        key = f"{hour:02d}:00"
    else:
        key = str(record.get(group_by_key, "Unknown"))
        
    is_delayed = 1 if arr_delay >= DELAY_THRESHOLD_MINUTES else 0
    
    # Value tuple: (count=1, delay_sum=arr_delay, delayed_count=is_delayed, max_delay=arr_delay)
    return key, (1, arr_delay, is_delayed, arr_delay)
