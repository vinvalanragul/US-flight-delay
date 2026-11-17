"""
Hadoop Environment Inspector and Runner.
Detects Hadoop CLI, Java, HDFS availability, and executes Hadoop Streaming jobs.
If Hadoop is not installed or configured, accurately reports 'Hadoop not configured'
without fabricating job execution or IDs.
"""

import os
import shutil
import subprocess
from typing import Dict, Any, List

def check_hadoop_environment() -> Dict[str, Any]:
    """
    Inspects whether Java, Hadoop, and HDFS are available in the current environment.
    """
    java_bin = shutil.which("java")
    hadoop_bin = shutil.which("hadoop")
    hdfs_bin = shutil.which("hdfs")
    
    java_version = None
    if java_bin:
        try:
            res = subprocess.run([java_bin, "-version"], capture_output=True, text=True, timeout=5)
            lines = (res.stderr or res.stdout).splitlines()
            java_version = lines[0] if lines else "Installed"
        except Exception:
            java_version = "Unknown"
            
    hadoop_version = None
    if hadoop_bin:
        try:
            res = subprocess.run([hadoop_bin, "version"], capture_output=True, text=True, timeout=5)
            lines = (res.stdout or res.stderr).splitlines()
            hadoop_version = lines[0] if lines else "Installed"
        except Exception:
            hadoop_version = "Unknown"
            
    hdfs_reachable = False
    if hdfs_bin:
        try:
            res = subprocess.run([hdfs_bin, "dfs", "-ls", "/"], capture_output=True, text=True, timeout=5)
            hdfs_reachable = (res.returncode == 0)
        except Exception:
            hdfs_reachable = False
            
    is_ready = bool(hadoop_bin and hdfs_reachable)
    status_label = "Ready" if is_ready else ("Hadoop installed, HDFS down" if hadoop_bin else "Hadoop not configured")
    
    return {
        "status": status_label,
        "is_ready": is_ready,
        "java_path": java_bin,
        "java_version": java_version,
        "hadoop_path": hadoop_bin,
        "hadoop_version": hadoop_version,
        "hdfs_reachable": hdfs_reachable,
        "wsl_available": bool(shutil.which("wsl"))
    }

def run_hadoop_streaming_job(
    group_by_key: str = "AIRLINE",
    input_file: str = "data/flights_sample.csv"
) -> Dict[str, Any]:
    """
    Submits a genuine Hadoop Streaming job if the environment is ready.
    If not configured, returns clear unverified diagnostic status.
    """
    env_info = check_hadoop_environment()
    
    if not env_info["is_ready"]:
        return {
            "success": False,
            "backend": "hadoop",
            "status": "Hadoop not configured",
            "message": "Genuine Hadoop cluster or HDFS daemons not detected in the current environment.",
            "details": env_info,
            "hdfs_input_path": f"/user/hadoop/flight_delay/input/{os.path.basename(input_file)}",
            "hdfs_output_path": f"/user/hadoop/flight_delay/output_{group_by_key.lower()}",
            "records": []
        }
        
    script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hadoop_streaming", "run_hadoop_job.sh")
    try:
        cmd = ["bash", script_path, group_by_key, input_file]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        output_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "outputs",
            f"hadoop_{group_by_key.lower()}",
            "hadoop_result.tsv"
        )
        
        records = []
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                header = None
                for line in f:
                    parts = line.strip().split("\t")
                    if not parts:
                        continue
                    if header is None:
                        header = parts
                        continue
                    if len(parts) >= 7:
                        records.append({
                            "key": parts[0],
                            "total_flights": int(parts[1]),
                            "delayed_flights": int(parts[2]),
                            "ontime_flights": int(parts[3]),
                            "delay_rate_pct": float(parts[4]),
                            "avg_delay_min": float(parts[5]),
                            "max_delay_min": float(parts[6])
                        })
                        
        return {
            "success": (proc.returncode == 0),
            "backend": "hadoop",
            "status": "Success" if proc.returncode == 0 else "Execution Failed",
            "job_command": " ".join(cmd),
            "log": proc.stdout + "\n" + proc.stderr,
            "hdfs_input_path": f"/user/hadoop/flight_delay/input/{os.path.basename(input_file)}",
            "hdfs_output_path": f"/user/hadoop/flight_delay/output_{group_by_key.lower()}",
            "records": records
        }
    except Exception as e:
        return {
            "success": False,
            "backend": "hadoop",
            "status": "Exception",
            "message": str(e),
            "records": []
        }
