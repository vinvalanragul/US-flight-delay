"""
Local MapReduce Simulator for US Flight Delay Pipeline.
Explicitly implements:
1. Mapper execution
2. Shuffle & Sort (grouping and key sorting)
3. Reducer aggregation
Provides sample inspection for viva and dashboard demonstrations.
"""

from typing import List, Dict, Any, Tuple
from collections import defaultdict
from src.mapper import map_flight_record
from src.reducer import reduce_flight_group

def run_mapreduce_simulation(
    records: List[Dict[str, Any]],
    group_by_key: str = "AIRLINE"
) -> Dict[str, Any]:
    """
    Executes a complete 3-stage MapReduce job locally:
    Returns {
        "mapper_sample": list of sample mapped tuples,
        "mapped_count": total emitted pairs,
        "shuffle_keys_count": unique keys sorted,
        "reduced_results": list of aggregated dicts,
        "summary": overall totals
    }
    """
    # Stage 1: Mapper Phase
    mapped_pairs: List[Tuple[str, Tuple[int, float, int, float]]] = []
    skipped_records = 0
    
    for row in records:
        mapped = map_flight_record(row, group_by_key=group_by_key)
        if mapped is not None:
            mapped_pairs.append(mapped)
        else:
            skipped_records += 1
            
    # Stage 2: Shuffle and Sort Phase
    # Groups intermediate values by key and sorts the keys alphabetically
    grouped_data = defaultdict(list)
    for key, val in mapped_pairs:
        grouped_data[key].append(val)
        
    sorted_keys = sorted(grouped_data.keys())
    
    # Stage 3: Reducer Phase
    reduced_results = []
    total_all_flights = 0
    total_all_delayed = 0
    total_all_delay_sum = 0.0
    
    for key in sorted_keys:
        vals = grouped_data[key]
        reduced_dict = reduce_flight_group(key, vals)
        reduced_results.append(reduced_dict)
        
        total_all_flights += reduced_dict["total_flights"]
        total_all_delayed += reduced_dict["delayed_flights"]
        # Recover sum for overall average
        total_all_delay_sum += (reduced_dict["avg_delay_min"] * reduced_dict["total_flights"])
        
    overall_delay_rate = round((total_all_delayed / total_all_flights * 100) if total_all_flights > 0 else 0.0, 2)
    overall_avg_delay = round((total_all_delay_sum / total_all_flights) if total_all_flights > 0 else 0.0, 2)
    
    # Sort reduced results by total flights descending or delay rate descending
    sorted_reduced = sorted(reduced_results, key=lambda x: x["total_flights"], reverse=True)
    
    return {
        "group_by_key": group_by_key,
        "total_records_ingested": len(records),
        "skipped_records": skipped_records,
        "mapped_pairs_count": len(mapped_pairs),
        "mapper_sample": mapped_pairs[:10],
        "shuffle_keys_count": len(sorted_keys),
        "reduced_results": sorted_reduced,
        "summary": {
            "total_flights": total_all_flights,
            "delayed_flights": total_all_delayed,
            "ontime_flights": total_all_flights - total_all_delayed,
            "overall_delay_rate_pct": overall_delay_rate,
            "overall_avg_delay_min": overall_avg_delay
        }
    }
