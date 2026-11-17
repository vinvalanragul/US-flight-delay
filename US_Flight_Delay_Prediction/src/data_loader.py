"""
Data Loader module for US Flight Delay dataset.
Handles zip archives (e.g. archive (4).zip), large CSV files via chunking,
schema inspection, and configurable row limits.
"""

import os
import csv
import zipfile
from typing import List, Dict, Any, Generator, Tuple
from src.config import DATASET_CANDIDATES, DATA_DIR

def find_dataset_source() -> str:
    """Finds the first existing dataset archive or CSV file."""
    for candidate in DATASET_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    # Default fallback path
    return os.path.join(DATA_DIR, "flights_sample.csv")

def get_file_info(file_path: str) -> Dict[str, Any]:
    """Returns file metadata including file size and type."""
    if not os.path.exists(file_path):
        return {"exists": False, "path": file_path}
    
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    is_zip = zipfile.is_zipfile(file_path)
    
    internal_files = []
    if is_zip:
        with zipfile.ZipFile(file_path, "r") as z:
            internal_files = [f.filename for f in z.infolist()]
            
    return {
        "exists": True,
        "path": file_path,
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 2),
        "is_zip": is_zip,
        "internal_files": internal_files
    }

def open_csv_stream(file_path: str):
    """Returns a file-like object and filename for reading CSV content."""
    if zipfile.is_zipfile(file_path):
        z = zipfile.ZipFile(file_path, "r")
        # Find csv inside
        csv_names = [f for f in z.namelist() if f.endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV file found inside archive: {file_path}")
        target_csv = csv_names[0]
        # Return TextIOWrapper over binary stream
        import io
        return io.TextIOWrapper(z.open(target_csv), encoding="utf-8", errors="replace"), target_csv
    else:
        return open(file_path, mode="r", encoding="utf-8", errors="replace"), os.path.basename(file_path)

def load_flight_records(
    file_path: str = None,
    limit: int = 50000
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Loads flight records with a configurable row limit for laptop-friendly development.
    Returns (records, metadata).
    """
    if file_path is None:
        file_path = find_dataset_source()
        
    info = get_file_info(file_path)
    if not info["exists"]:
        # If dataset does not exist, generate default sample
        from data.generate_dataset import generate_flight_dataset
        sample_path = os.path.join(DATA_DIR, "flights_sample.csv")
        generate_flight_dataset(sample_path, num_records=15000)
        file_path = sample_path
        info = get_file_info(file_path)

    f, inner_name = open_csv_stream(file_path)
    reader = csv.DictReader(f)
    
    records = []
    initial_row_count = 0
    date_min = None
    date_max = None
    
    for row in reader:
        initial_row_count += 1
        d_val = row.get("FL_DATE", "")
        if d_val:
            if date_min is None or d_val < date_min:
                date_min = d_val
            if date_max is None or d_val > date_max:
                date_max = d_val
                
        if limit is None or len(records) < limit:
            records.append(row)
            
    f.close()
    
    schema = list(records[0].keys()) if records else []
    metadata = {
        "file_path": file_path,
        "inner_filename": inner_name,
        "file_size_mb": info["size_mb"],
        "total_source_rows": initial_row_count,
        "loaded_rows": len(records),
        "is_sampled": (initial_row_count > len(records)),
        "columns_count": len(schema),
        "schema": schema,
        "date_coverage": f"{date_min} to {date_max}" if date_min and date_max else "N/A"
    }
    
    return records, metadata

def load_flight_chunks(
    file_path: str = None,
    chunk_size: int = 10000,
    max_chunks: int = None
) -> Generator[List[Dict[str, Any]], None, None]:
    """Generates chunks of flight records for memory-efficient big data processing."""
    if file_path is None:
        file_path = find_dataset_source()
        
    f, _ = open_csv_stream(file_path)
    reader = csv.DictReader(f)
    
    chunk = []
    chunk_count = 0
    
    for row in reader:
        chunk.append(row)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
            chunk_count += 1
            if max_chunks is not None and chunk_count >= max_chunks:
                f.close()
                return
                
    if chunk:
        yield chunk
    f.close()
