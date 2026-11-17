"""
Dataset Preparation & Inspection Script.
Checks for archive (4).zip, unzips or inspects headers, and verifies schema.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import find_dataset_source, get_file_info, load_flight_records
from src.preprocessing import clean_and_preprocess

def main():
    print("=" * 60)
    print("US Flight Delay Dataset Inspection & Preparation Utility")
    print("=" * 60)
    
    source_path = find_dataset_source()
    print(f"Target dataset source: {source_path}")
    
    info = get_file_info(source_path)
    print(f"File exists:     {info['exists']}")
    print(f"File size:       {info.get('size_mb', 0)} MB")
    print(f"Is ZIP archive:  {info.get('is_zip', False)}")
    if info.get("is_zip"):
        print(f"Archive contents: {info.get('internal_files', [])}")
        
    print("\nLoading dataset records (sample limit: 20,000 for preview)...")
    records, meta = load_flight_records(source_path, limit=20000)
    
    print(f"Total rows detected: {meta['total_source_rows']}")
    print(f"Loaded rows:         {meta['loaded_rows']}")
    print(f"Columns count:       {meta['columns_count']}")
    print(f"Date coverage:       {meta['date_coverage']}")
    print(f"Columns present:\n  {', '.join(meta['schema'][:16])}\n  {', '.join(meta['schema'][16:])}")
    
    print("\nRunning cleaning and preprocessing pipeline...")
    cleaned, report = clean_and_preprocess(records, output_clean_csv="data/cleaned_flights_sample.csv")
    
    print(f"Cleaned records:     {report['total_cleaned_records']}")
    print(f"Removed cancelled:   {report['removed_cancelled']}")
    print(f"Removed diverted:    {report['removed_diverted']}")
    print(f"Removed missing arr: {report['removed_missing_arrival_delay']}")
    print(f"Delayed flights:     {report['delayed_flights']} ({report['delay_rate_pct']}%)")
    print(f"On-time flights:     {report['ontime_flights']}")
    print("\nData preparation complete. Output saved to data/cleaned_flights_sample.csv")

if __name__ == "__main__":
    main()
