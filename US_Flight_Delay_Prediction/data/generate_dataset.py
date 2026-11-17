"""
Dataset Generator for US Flight Delay Analytics Pipeline.
Generates an authentic 32-column sample dataset matching the schema of:
Kaggle / Bureau of Transportation Statistics (BTS) 2019-2023 Flight Delay & Cancellation dataset.
Used for local testing, Windows laptop execution, and quick pipeline validation.
"""

import csv
import os
import random
from datetime import datetime, timedelta

def generate_flight_dataset(output_path: str, num_records: int = 15000, seed: int = 42):
    random.seed(seed)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    airlines = [
        ("American Airlines Inc.", "AA", 19805),
        ("Delta Air Lines Inc.", "DL", 19790),
        ("United Air Lines Inc.", "UA", 19977),
        ("Southwest Airlines Co.", "WN", 19393),
        ("JetBlue Airways", "B6", 20409),
        ("Spirit Air Lines", "NK", 20416),
        ("Alaska Airlines Inc.", "AS", 19930)
    ]
    
    airports = [
        ("ORD", "Chicago, IL"),
        ("ATL", "Atlanta, GA"),
        ("DFW", "Dallas/Fort Worth, TX"),
        ("DEN", "Denver, CO"),
        ("LAX", "Los Angeles, CA"),
        ("JFK", "New York, NY"),
        ("SFO", "San Francisco, CA"),
        ("CLT", "Charlotte, NC"),
        ("MIA", "Miami, FL"),
        ("BOS", "Boston, MA"),
        ("SEA", "Seattle, WA"),
        ("PHX", "Phoenix, AZ"),
        ("LAS", "Las Vegas, NV"),
        ("EWR", "Newark, NJ"),
        ("MCO", "Orlando, FL")
    ]
    
    start_date = datetime(2023, 1, 1)
    
    headers = [
        "FL_DATE", "AIRLINE", "AIRLINE_DOT", "AIRLINE_CODE", "DOT_CODE", "FL_NUMBER",
        "ORIGIN", "ORIGIN_CITY", "DEST", "DEST_CITY", "CRS_DEP_TIME", "DEP_TIME",
        "DEP_DELAY", "TAXI_OUT", "WHEELS_OFF", "WHEELS_ON", "TAXI_IN", "CRS_ARR_TIME",
        "ARR_TIME", "ARR_DELAY", "CANCELLED", "CANCELLATION_CODE", "DIVERTED",
        "CRS_ELAPSED_TIME", "ELAPSED_TIME", "AIR_TIME", "DISTANCE",
        "DELAY_DUE_CARRIER", "DELAY_DUE_WEATHER", "DELAY_DUE_NAS", "DELAY_DUE_SECURITY", "DELAY_DUE_LATE_AIRCRAFT"
    ]
    
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for i in range(num_records):
            days_offset = random.randint(0, 364)
            fl_date = (start_date + timedelta(days=days_offset)).strftime("%Y-%m-%d")
            
            airline_name, airline_code, dot_code = random.choice(airlines)
            fl_num = random.randint(100, 6800)
            
            orig_apt, orig_city = random.choice(airports)
            dest_apt, dest_city = random.choice([a for a in airports if a[0] != orig_apt])
            
            dep_hour = random.randint(5, 22)
            dep_min = random.choice([0, 15, 30, 45, 10, 20, 40, 50])
            crs_dep_time = dep_hour * 100 + dep_min
            
            # Distance based roughly on route
            distance = random.randint(250, 2600)
            crs_elapsed = int(50 + (distance / 450.0) * 60)
            
            arr_hour = (dep_hour + (dep_min + crs_elapsed) // 60) % 24
            arr_min = (dep_min + crs_elapsed) % 60
            crs_arr_time = arr_hour * 100 + arr_min
            
            # Cancellation rate ~ 1.8%
            is_cancelled = 1.0 if random.random() < 0.018 else 0.0
            # Diverted rate ~ 0.3%
            is_diverted = 1.0 if (not is_cancelled and random.random() < 0.003) else 0.0
            
            if is_cancelled:
                writer.writerow([
                    fl_date, airline_name, f"DOT-{dot_code}", airline_code, dot_code, fl_num,
                    orig_apt, orig_city, dest_apt, dest_city, crs_dep_time, "",
                    "", "", "", "", "", crs_arr_time,
                    "", "", 1.0, random.choice(["A", "B", "C"]), 0.0,
                    crs_elapsed, "", "", distance,
                    "", "", "", "", ""
                ])
                continue
                
            if is_diverted:
                dep_delay = round(random.gauss(10, 25), 1)
                writer.writerow([
                    fl_date, airline_name, f"DOT-{dot_code}", airline_code, dot_code, fl_num,
                    orig_apt, orig_city, dest_apt, dest_city, crs_dep_time, crs_dep_time + int(dep_delay),
                    dep_delay, 18.0, "", "", "", crs_arr_time,
                    "", "", 0.0, "", 1.0,
                    crs_elapsed, "", "", distance,
                    "", "", "", "", ""
                ])
                continue
            
            # Normal completed flight with realistic delay distribution:
            # - Evening flights have higher delay rates due to cascading delays
            # - B6, NK, and congested hubs (ORD, EWR, JFK, SFO) have slightly higher delay tendencies
            delay_factor = 0.0
            if dep_hour >= 16:
                delay_factor += 0.35
            elif dep_hour >= 12:
                delay_factor += 0.15
            
            if orig_apt in ["ORD", "EWR", "JFK", "SFO", "ATL"]:
                delay_factor += 0.2
            if airline_code in ["B6", "NK"]:
                delay_factor += 0.15
            elif airline_code in ["DL", "AS"]:
                delay_factor -= 0.1
                
            # Determine if delayed (BTS defines delay as ARR_DELAY >= 15 min)
            # US industry baseline delay rate is approx 18-22%
            prob_delay = max(0.08, min(0.48, 0.19 + delay_factor * 0.3))
            is_delayed_flight = random.random() < prob_delay
            
            if is_delayed_flight:
                # Delay in range [15, 240]
                arr_delay = round(random.expovariate(1.0 / 35.0) + 15.0, 1)
                dep_delay = round(max(-5.0, arr_delay + random.gauss(-4, 8)), 1)
            else:
                # On-time or early: range [-35, 14]
                arr_delay = round(random.gauss(-5, 6), 1)
                if arr_delay >= 15:
                    arr_delay = 14.0
                dep_delay = round(max(-15.0, arr_delay + random.gauss(-2, 4)), 1)
                
            taxi_out = round(max(8.0, random.gauss(17, 6)), 1)
            taxi_in = round(max(4.0, random.gauss(8, 3)), 1)
            air_time = round(max(30.0, crs_elapsed - taxi_out - taxi_in + random.gauss(0, 5)), 1)
            actual_elapsed = taxi_out + air_time + taxi_in
            
            dep_time = (crs_dep_time + int(dep_delay)) % 2400
            arr_time = (crs_arr_time + int(arr_delay)) % 2400
            
            carrier_delay = ""
            weather_delay = ""
            nas_delay = ""
            sec_delay = ""
            late_ac_delay = ""
            
            if arr_delay >= 15:
                # Allocate delay breakdown
                rem = arr_delay
                c_del = round(random.uniform(0, rem), 1)
                rem -= c_del
                late_del = round(random.uniform(0, rem), 1)
                rem -= late_del
                nas_del = round(rem, 1)
                carrier_delay = str(c_del)
                late_ac_delay = str(late_del)
                nas_delay = str(nas_del)
                weather_delay = "0.0"
                sec_delay = "0.0"
                
            writer.writerow([
                fl_date, airline_name, f"DOT-{dot_code}", airline_code, dot_code, fl_num,
                orig_apt, orig_city, dest_apt, dest_city, crs_dep_time, dep_time,
                dep_delay, taxi_out, crs_dep_time + 15, crs_arr_time - 10, taxi_in, crs_arr_time,
                arr_time, arr_delay, 0.0, "", 0.0,
                crs_elapsed, actual_elapsed, air_time, distance,
                carrier_delay, weather_delay, nas_delay, sec_delay, late_ac_delay
            ])
            
    print(f"Generated {num_records} flight records at {output_path}")

if __name__ == "__main__":
    generate_flight_dataset("US_Flight_Delay_Prediction/data/flights_sample.csv", num_records=15000)
