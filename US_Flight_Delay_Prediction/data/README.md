# Data Directory

This directory stores flight delay datasets:

1. `flights_sample.csv`: A high-quality, 15,000-record sample dataset matching the authentic 32-column schema of the Bureau of Transportation Statistics (BTS) / Kaggle 2019-2023 Flight Delay dataset.
2. Full dataset: You may place `archive (4).zip` or `flights_sample_3m.csv` in this folder or in the project root. The `src/data_loader.py` module will automatically detect and load it using memory-efficient chunked processing.
