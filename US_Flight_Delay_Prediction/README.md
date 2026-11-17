# US Flight Delay Prediction: An End-to-End Big Data Analytics Pipeline from Hadoop MapReduce to Machine Learning

A complete, production-grade Big Data Analytics mini-project for college CA3 assessment.

## 🚀 Quick Start (Windows)

1. **Clone or extract** this folder to your Windows machine.
2. **Double-click** `scripts/run_windows.bat`
   
*Or run manually in PowerShell:*
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python tests\test_pipeline.py
streamlit run app.py
```

---

## 🐘 Ubuntu / WSL2 Hadoop Execution

To execute genuine distributed Hadoop Streaming jobs:
```bash
# 1. Start Hadoop cluster services
start-dfs.sh && start-yarn.sh

# 2. Run automated streaming job
chmod +x hadoop_streaming/run_hadoop_job.sh
./hadoop_streaming/run_hadoop_job.sh AIRLINE data/flights_sample.csv

# 3. View HDFS outputs
hdfs dfs -cat /user/hadoop/flight_delay/output_airline/part-00000
```

---

## 📁 Project Directory Structure

```text
US_Flight_Delay_Prediction/
├── app.py                      # Multi-page Streamlit Dashboard entry point
├── pages/
│   ├── 1_Dataset_Explorer.py    # Schema inspection, table preview, filtering
│   ├── 2_Preprocessing.py       # Data cleaning, missing values, anti-leakage audit
│   ├── 3_MapReduce.py           # Local vs Hadoop backend, 3-stage visualizer
│   ├── 4_Flight_Analytics.py    # Punctuality rankings, temporal cascades
│   ├── 5_Machine_Learning.py    # Logistic Regression vs Random Forest benchmark
│   ├── 6_Predict_Flight.py      # Pre-departure interactive flight delay inference
│   ├── 7_Business_Insights.py   # Data-driven recommendations & operational advice
│   └── 8_Hadoop_Documentation.py# Architecture diagrams, WSL2 guide, Viva Q&A
├── src/
│   ├── config.py                # Paths, operational thresholds, feature lists
│   ├── data_loader.py           # Chunked loader, zip support (archive (4).zip)
│   ├── preprocessing.py         # Cancelled/diverted filter, target derivation
│   ├── mapper.py                # Pure-python Mapper
│   ├── reducer.py               # Pure-python Reducer
│   ├── mapreduce_simulator.py   # Mapper, Shuffle & Sort, Reducer execution
│   ├── mapreduce_interface.py   # Dual-backend selector (local vs hadoop)
│   ├── hadoop_runner.py         # Submits Hadoop streaming, inspects HDFS
│   ├── analytics.py             # KPIs, distributions, aggregations
│   ├── train_model.py           # Model training & model persistence
│   ├── evaluate_model.py        # Accuracy, Precision, Recall, F1, ROC-AUC
│   └── prediction.py            # Pre-departure risk estimator
├── hadoop_streaming/
│   ├── mapper.py                # Tab-delimited STDIN/STDOUT Hadoop mapper
│   ├── reducer.py               # Aggregating Hadoop reducer
│   ├── run_hadoop_job.sh        # Bash runner for Ubuntu / WSL2
│   ├── run_hadoop_job.ps1       # PowerShell wrapper for Windows
│   └── README.md                # Hadoop Streaming protocol documentation
├── scripts/
│   ├── run_windows.bat          # One-click Windows startup batch script
│   ├── prepare_data.py          # Previews & cleans datasets
│   └── export_report.py         # Generates DOCX & Markdown reports
├── data/
│   ├── flights_sample.csv       # 15,000-record BTS-authentic sample dataset
│   └── README.md
├── reports/
│   ├── project_report.docx      # Formatted Academic Report for Word/Office
│   ├── project_report.md        # Complete Markdown Report source
│   └── viva_questions.md        # Comprehensive 25+ Viva Voce questions & answers
├── tests/
│   └── test_pipeline.py         # 9 automated tests covering entire pipeline
└── requirements.txt             # Python dependencies
```

---

## 🧪 Automated Tests
Run the automated test suite with:
```bash
python tests/test_pipeline.py
```
*All 9 test cases covering target thresholding, anti-leakage verification, mapper emissions, reducer aggregations, and model evaluations pass automatically.*
