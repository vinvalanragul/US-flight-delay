"""
MapReduce Reducer for US Flight Delay Analytics.
Aggregates intermediate mapped values and computes delay rate and metrics.
"""

from typing import List, Tuple, Dict, Any

def reduce_flight_group(
    key: str,
    values: List[Tuple[int, float, int, float]]
) -> Dict[str, Any]:
    """
    Reduces a list of intermediate tuples for a single key.
    Each tuple is: (count, delay_sum, delayed_count, max_delay)
    """
    total_flights = 0
    total_delay_sum = 0.0
    total_delayed = 0
    max_delay = float("-inf")
    
    for count, delay_sum, delayed_count, m_del in values:
        total_flights += count
        total_delay_sum += delay_sum
        total_delayed += delayed_count
        if m_del > max_delay:
            max_delay = m_del
            
    if total_flights == 0:
        return {
            "key": key,
            "total_flights": 0,
            "delayed_flights": 0,
            "ontime_flights": 0,
            "delay_rate_pct": 0.0,
            "avg_delay_min": 0.0,
            "max_delay_min": 0.0
        }
        
    avg_delay = round(total_delay_sum / total_flights, 2)
    ontime_flights = total_flights - total_delayed
    delay_rate_pct = round((total_delayed / total_flights) * 100, 2)
    
    return {
        "key": key,
        "total_flights": total_flights,
        "delayed_flights": total_delayed,
        "ontime_flights": ontime_flights,
        "delay_rate_pct": delay_rate_pct,
        "avg_delay_min": avg_delay,
        "max_delay_min": round(max_delay, 2) if max_delay != float("-inf") else 0.0
    }
