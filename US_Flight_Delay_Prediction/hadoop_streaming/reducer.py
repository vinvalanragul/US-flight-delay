#!/usr/bin/env python3
"""
Hadoop Streaming Reducer for US Flight Delay Prediction.
Reads tab-separated lines emitted by mapper.py sorted by key:
<KEY>\t<COUNT>\t<ARR_DELAY_SUM>\t<DELAYED_COUNT>\t<MAX_DELAY>

Emits final aggregated flight metrics:
<KEY>\t<TOTAL_FLIGHTS>\t<DELAYED_FLIGHTS>\t<ONTIME_FLIGHTS>\t<DELAY_RATE_PCT>\t<AVG_DELAY_MIN>\t<MAX_DELAY_MIN>
"""

import sys

def main():
    current_key = None
    total_count = 0
    total_delay = 0.0
    total_delayed = 0
    max_delay = float("-inf")
    
    # Emit header
    sys.stdout.write("KEY\tTOTAL_FLIGHTS\tDELAYED_FLIGHTS\tONTIME_FLIGHTS\tDELAY_RATE_PCT\tAVG_DELAY_MIN\tMAX_DELAY_MIN\n")
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split("\t")
        if len(parts) != 5:
            continue
            
        key, count_str, delay_str, delayed_str, max_str = parts
        
        try:
            count = int(count_str)
            delay = float(delay_str)
            is_del = int(delayed_str)
            m_del = float(max_str)
        except ValueError:
            continue
            
        if current_key == key:
            total_count += count
            total_delay += delay
            total_delayed += is_del
            if m_del > max_delay:
                max_delay = m_del
        else:
            if current_key is not None and total_count > 0:
                avg_del = total_delay / total_count
                ontime = total_count - total_delayed
                delay_rate = (total_delayed / total_count) * 100.0
                sys.stdout.write(f"{current_key}\t{total_count}\t{total_delayed}\t{ontime}\t{delay_rate:.2f}\t{avg_del:.2f}\t{max_delay:.2f}\n")
                
            current_key = key
            total_count = count
            total_delay = delay
            total_delayed = is_del
            max_delay = m_del
            
    # Emit last key group
    if current_key is not None and total_count > 0:
        avg_del = total_delay / total_count
        ontime = total_count - total_delayed
        delay_rate = (total_delayed / total_count) * 100.0
        sys.stdout.write(f"{current_key}\t{total_count}\t{total_delayed}\t{ontime}\t{delay_rate:.2f}\t{avg_del:.2f}\t{max_delay:.2f}\n")

if __name__ == "__main__":
    main()
