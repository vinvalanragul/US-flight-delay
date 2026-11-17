"""
Unified MapReduce Interface for US Flight Delay Prediction.
Exposes a consistent API supporting:
- backend="local": Pure-Python simulation of Mapper, Shuffle/Sort, and Reducer
- backend="hadoop": Genuine Hadoop Streaming execution via HDFS/YARN
"""

from typing import List, Dict, Any
from src.mapreduce_simulator import run_mapreduce_simulation
from src.hadoop_runner import run_hadoop_streaming_job, check_hadoop_environment

def run_mapreduce(
    records: List[Dict[str, Any]],
    group_by_key: str = "AIRLINE",
    backend: str = "local",
    input_file_path: str = "data/flights_sample.csv"
) -> Dict[str, Any]:
    """
    Executes MapReduce using the selected backend.
    """
    backend = backend.lower()
    if backend == "hadoop":
        result = run_hadoop_streaming_job(group_by_key=group_by_key, input_file=input_file_path)
        return result
    else:
        sim_result = run_mapreduce_simulation(records, group_by_key=group_by_key)
        sim_result["backend"] = "local"
        sim_result["status"] = "Success (Local Simulation)"
        sim_result["success"] = True
        return sim_result

def get_backend_status() -> Dict[str, Any]:
    """Returns availability and status of both execution backends."""
    hadoop_env = check_hadoop_environment()
    return {
        "local": {
            "available": True,
            "description": "In-memory Python simulation of Mapper, Shuffle & Sort, and Reducer phases",
            "ideal_for": "Local Windows testing, interactive development, fast iteration"
        },
        "hadoop": {
            "available": hadoop_env["is_ready"],
            "status": hadoop_env["status"],
            "description": "Genuine Hadoop Streaming over HDFS and YARN daemons",
            "ideal_for": "Multi-node cluster or Ubuntu/WSL2 distributed execution",
            "details": hadoop_env
        }
    }
