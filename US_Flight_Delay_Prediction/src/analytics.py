"""
Big Data Analytics Module for US Flight Delays.
Computes comprehensive aggregations, temporal curves, ranking tables,
and statistical distributions with sample-size safeguards.
"""

from typing import List, Dict, Any
from collections import defaultdict
from src.config import DELAY_THRESHOLD_MINUTES

def compute_overall_kpis(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes headline KPIs from cleaned flight records."""
    total = len(records)
    if total == 0:
        return {
            "total_flights": 0,
            "delayed_flights": 0,
            "ontime_flights": 0,
            "delay_rate_pct": 0.0,
            "avg_delay_min": 0.0,
            "max_delay_min": 0.0,
            "severe_delay_flights": 0
        }
        
    delayed = sum(1 for r in records if r.get("is_delayed", 0) == 1)
    ontime = total - delayed
    delays = [float(r.get("ARR_DELAY", 0.0)) for r in records if r.get("ARR_DELAY") is not None]
    
    avg_delay = round(sum(delays) / len(delays), 2) if delays else 0.0
    max_delay = round(max(delays), 2) if delays else 0.0
    severe = sum(1 for d in delays if d >= 120.0) # 2+ hours delay
    
    return {
        "total_flights": total,
        "delayed_flights": delayed,
        "ontime_flights": ontime,
        "delay_rate_pct": round((delayed / total) * 100, 2),
        "avg_delay_min": avg_delay,
        "max_delay_min": max_delay,
        "severe_delay_flights": severe,
        "severe_delay_pct": round((severe / total) * 100, 2)
    }

def aggregate_by_group(
    records: List[Dict[str, Any]],
    group_col: str,
    min_sample_size: int = 10
) -> List[Dict[str, Any]]:
    """
    Aggregates flight records by a specified column.
    Filters out groups below min_sample_size to prevent deceptive rankings.
    """
    groups = defaultdict(lambda: {"count": 0, "delayed": 0, "delay_sum": 0.0, "max_delay": float("-inf")})
    
    for r in records:
        val = r.get(group_col)
        if val is None or val == "":
            val = "Unknown"
        val_str = str(val)
        
        arr_delay = float(r.get("ARR_DELAY", 0.0))
        is_del = 1 if r.get("is_delayed") == 1 or arr_delay >= DELAY_THRESHOLD_MINUTES else 0
        
        g = groups[val_str]
        g["count"] += 1
        g["delayed"] += is_del
        g["delay_sum"] += arr_delay
        if arr_delay > g["max_delay"]:
            g["max_delay"] = arr_delay
            
    results = []
    for key, stats in groups.items():
        if stats["count"] < min_sample_size:
            continue
        tot = stats["count"]
        del_count = stats["delayed"]
        avg_del = round(stats["delay_sum"] / tot, 2)
        rate = round((del_count / tot) * 100, 2)
        results.append({
            "group": key,
            "total_flights": tot,
            "delayed_flights": del_count,
            "ontime_flights": tot - del_count,
            "delay_rate_pct": rate,
            "avg_delay_min": avg_del,
            "max_delay_min": round(stats["max_delay"], 2) if stats["max_delay"] != float("-inf") else 0.0
        })
        
    return sorted(results, key=lambda x: x["delay_rate_pct"], reverse=True)

def compute_delay_severity_distribution(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Categorizes arrival delays into standard FAA severity bands."""
    categories = {
        "Early / On-Time (<15 min)": 0,
        "Mild Delay (15-45 min)": 0,
        "Moderate Delay (45-120 min)": 0,
        "Severe Delay (>120 min)": 0
    }
    
    total = len(records)
    for r in records:
        arr_delay = float(r.get("ARR_DELAY", 0.0))
        if arr_delay < 15.0:
            categories["Early / On-Time (<15 min)"] += 1
        elif arr_delay < 45.0:
            categories["Mild Delay (15-45 min)"] += 1
        elif arr_delay < 120.0:
            categories["Moderate Delay (45-120 min)"] += 1
        else:
            categories["Severe Delay (>120 min)"] += 1
            
    percentages = {
        cat: round((count / total * 100) if total > 0 else 0.0, 2)
        for cat, count in categories.items()
    }
    
    return {
        "counts": categories,
        "percentages": percentages,
        "total": total
    }

def compute_temporal_analysis(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyzes delay trends across Month, Day of Week, and Departure Hour."""
    by_month = aggregate_by_group(records, "MONTH", min_sample_size=1)
    by_month_sorted = sorted(by_month, key=lambda x: int(str(x["group"]).split("-")[-1]) if str(x["group"]).replace("-", "").isdigit() else str(x["group"]))
    
    weekday_names = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
    by_weekday_raw = aggregate_by_group(records, "DAY_OF_WEEK", min_sample_size=1)
    by_weekday = []
    for item in by_weekday_raw:
        try:
            w_int = int(item["group"])
            name = weekday_names.get(w_int, f"Day {w_int}")
        except Exception:
            name = str(item["group"])
        by_weekday.append({**item, "weekday_name": name})
    by_weekday_sorted = sorted(by_weekday, key=lambda x: int(x["group"]) if str(x["group"]).isdigit() else 0)
    
    by_hour = aggregate_by_group(records, "CRS_DEP_HOUR", min_sample_size=1)
    by_hour_sorted = sorted(by_hour, key=lambda x: int(str(x["group"]).split(":")[0]) if str(x["group"]).split(":")[0].isdigit() else 0)
    
    return {
        "by_month": by_month_sorted,
        "by_weekday": by_weekday_sorted,
        "by_hour": by_hour_sorted
    }
