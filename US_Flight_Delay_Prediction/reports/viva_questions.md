# Comprehensive Viva Voce Questions & Answers
## US Flight Delay Prediction: An End-to-End Big Data Analytics Pipeline from Hadoop MapReduce to Machine Learning

---

### Q1: What is the primary objective of this project?
**Answer:** The project builds a complete, production-grade Big Data analytics and machine learning pipeline to analyze historical US commercial flight patterns, aggregate delay statistics using distributed Hadoop MapReduce principles, and predict pre-departure flight delay risks using machine learning classification models.

---

### Q2: What dataset was used, and what is its official source?
**Answer:** The dataset is the **Kaggle / Bureau of Transportation Statistics (BTS) Flight Delay and Cancellation Dataset (2019–2023)**, derived directly from the US Department of Transportation (DOT) Form 41, Schedule T-100/298C On-Time Reporting data. The full sample contains approximately 3 million records and 32 columns covering carrier details, origin/destination airports, scheduled and actual departure/arrival timestamps, elapsed times, distances, and delay categorization causes.

---

### Q3: Why is flight delay defined as 15 minutes or more?
**Answer:** The Federal Aviation Administration (FAA) and Bureau of Transportation Statistics (BTS) officially define a commercial flight as "delayed" when its actual arrival time is **15 minutes or more later than its scheduled gate arrival time** (`ARR_DELAY >= 15.0`). Arrival delays of less than 15 minutes are considered operational noise or standard gate congestion absorbable by schedule padding. Flights arriving early have negative delay values.

---

### Q4: Why did you filter out Cancelled and Diverted flights for arrival delay classification?
**Answer:** Cancelled flights (`CANCELLED == 1.0`) never depart and never reach their destination airport; consequently, their arrival delay is undefined (represented as `NaN` or missing). Diverted flights (`DIVERTED == 1.0`) land at an unintended airport due to weather, fuel, or emergency conditions, rendering original scheduled gate arrival metrics non-comparable. Including them would either distort delay averages or impute synthetic arrival times that never occurred.

---

### Q5: What is Data Leakage in predictive modeling, and how did you strictly prevent it?
**Answer:** Data leakage occurs when features containing information from after the event has occurred (post-flight details) are included in the feature set used to predict an outcome prior to departure. 
In our pipeline, we strictly isolated **pre-departure features available at booking or boarding**:
- Included: `AIRLINE`, `ORIGIN`, `DEST`, `MONTH`, `DAY_OF_WEEK`, `CRS_DEP_HOUR`, `CRS_ARR_HOUR`, `DISTANCE`.
- Strictly Excluded: `ARR_DELAY`, `DEP_DELAY`, `TAXI_OUT`, `WHEELS_OFF`, `WHEELS_ON`, `TAXI_IN`, `ACTUAL_ELAPSED_TIME`, and delay breakdown attributes (`DELAY_DUE_CARRIER`, `DELAY_DUE_WEATHER`, `DELAY_DUE_NAS`, `DELAY_DUE_LATE_AIRCRAFT`). Using `DEP_DELAY` would result in artificial 95%+ accuracy because a plane departing 60 minutes late will almost certainly arrive late, defeating the purpose of pre-departure operational forecasting.

---

### Q6: Explain the MapReduce architecture and workflow in this project.
**Answer:**
1. **Mapper Phase:** Reads each individual flight record, validates that it is completed and has a valid `ARR_DELAY`, extracts the chosen grouping key (e.g., `AIRLINE`, `ORIGIN`), and emits an intermediate key-value tuple:
   $$\text{Key} \rightarrow (1, \text{ARR\_DELAY}, \text{IS\_DELAYED}, \text{ARR\_DELAY})$$
2. **Shuffle and Sort Phase:** Hadoop's internal distributed engine partitions, hashes, and lexicographically sorts all emitted tuples by key, guaranteeing that all records belonging to a particular carrier or airport land on the same reducer node consecutively.
3. **Reducer Phase:** Aggregates partial sums associatively:
   - $\text{Total Flights} = \sum \text{count}$
   - $\text{Delayed Flights} = \sum \text{is\_delayed}$
   - $\text{On-Time Flights} = \text{Total Flights} - \text{Delayed Flights}$
   - $\text{Delay Rate \%} = \frac{\text{Delayed Flights}}{\text{Total Flights}} \times 100$
   - $\text{Average Delay (min)} = \frac{\sum \text{ARR\_DELAY}}{\text{Total Flights}}$
   - $\text{Max Delay (min)} = \max(\text{ARR\_DELAY})$

---

### Q7: Why are partial aggregates designed associatively in MapReduce?
**Answer:** Associative and commutative aggregations allow partial reduction (combiners) to occur locally on mapper nodes before network transfer during Shuffle, dramatically cutting down network I/O in distributed clusters. It also enables chunked streaming on resource-constrained laptop environments.

---

### Q8: What is Hadoop Streaming, and how does it work?
**Answer:** Hadoop Streaming is a utility that allows MapReduce jobs to be written in any programming language (Python, C++, Ruby, etc.) that can read from standard input (`sys.stdin`) and write to standard output (`sys.stdout`). The Hadoop framework launches Java wrappers that pipe HDFS line splits into the Python mapper process via STDIN, captures tab-separated key-value outputs via STDOUT, executes the cluster-wide Shuffle & Sort, and pipes the sorted stream into the Python reducer process.

---

### Q9: Why must the HDFS output directory NOT exist prior to job submission?
**Answer:** Hadoop MapReduce intentionally rejects job execution if the output directory already exists (`FileAlreadyExistsException`). This fail-safe prevents inadvertent overwriting or loss of previously computed big data outputs. Our runner script automatically checks and safely cleans stale output directories (`hdfs dfs -rm -r -skipTrash`) only after explicit verification.

---

### Q10: How did you implement both Local Simulation and Genuine Hadoop backends?
**Answer:** We implemented a clean unified interface (`src/mapreduce_interface.py`). The user can switch between:
- `backend="local"`: Runs in-memory Python simulation of Mapper, Shuffle & Sort, and Reducer for rapid exploration on Windows without requiring heavy daemons.
- `backend="hadoop"`: Checks Java, Hadoop, and HDFS daemon status. If available on Ubuntu / WSL2, it uploads data to HDFS and submits a real Hadoop Streaming job (`run_hadoop_job.sh`). If Hadoop is unconfigured, the UI transparently displays `"Hadoop not configured"` rather than fabricating false job IDs.

---

### Q11: Which Machine Learning algorithms did you implement and compare?
**Answer:** We implemented and compared two foundational classifiers:
1. **Logistic Regression:** A linear parametric model serving as an interpretable baseline, estimating log-odds of delay as a linear combination of categorical and numerical features.
2. **Random Forest Classifier:** An ensemble of decision trees utilizing bootstrap aggregating (bagging) and random feature subspace sampling, capable of learning non-linear feature interactions and high-order patterns without requiring monotonic assumptions.

---

### Q12: Why is Accuracy alone an inadequate metric for flight delay prediction?
**Answer:** In commercial aviation, only approximately 20% to 25% of completed flights experience delays of $\ge 15$ minutes. A naive "dummy" model that predicts every single flight will be on-time achieves 75–80% accuracy while failing to identify a single delayed flight! Therefore, **Recall** (sensitivity to true delays), **Precision** (avoiding false alarms), **F1-Score** (harmonic mean), and **ROC-AUC** (discriminative ability across all probability thresholds) are essential.

---

### Q13: What does the Confusion Matrix tell you in this domain?
**Answer:**
- **True Positive (TP):** Flight was delayed and model correctly predicted delay (Allows passengers and airlines to plan contingencies).
- **False Positive (FP):** Flight was on-time, but model predicted delay (False alarm; minor passenger inconvenience).
- **True Negative (TN):** Flight was on-time and model predicted on-time.
- **False Negative (FN):** Flight was delayed, but model predicted on-time (**Critical operational miss**; passengers miss connections, crew timing out).

---

### Q14: Which model is recommended and why?
**Answer:** **Random Forest Classifier** is the recommended model because it captures complex non-linear combinations between scheduled departure time and hub congestion. While Logistic Regression treats departure hour linearly, delay risks exhibit a sharp hockey-stick increase after 16:00 (due to cumulative delay cascades throughout the day). Random Forest achieves higher Recall and F1-score on delayed flights.

---

### Q15: What key business insights emerged from the analytics?
**Answer:**
1. **Departure Hour Cascade:** Flights departing before 09:00 have the lowest delay rates (~11%), whereas flights departing between 17:00 and 20:00 experience delay rates exceeding 32%, demonstrating network-wide delay accumulation.
2. **Hub Asymmetry:** Congested mega-hubs (e.g., EWR, JFK, ORD, SFO) exhibit substantially higher delay rates than fortress or origin-only hubs due to runway limitations and saturated airspace.
3. **Carrier Variance:** Low-cost carriers operating tight 30-minute turnarounds with high aircraft utilization face faster delay propagation when a morning delay occurs, whereas legacy carriers with spare aircraft banks at hub bases can absorb turnaround disruptions more effectively.

---

### Q16: What practical recommendations do you provide to airlines and passengers?
**Answer:**
- **For Airlines:** Increase scheduled turnaround buffer times for aircraft rotating into evening hub departures; implement proactive schedule de-peaking; deploy predictive crew reserve scheduling.
- **For Passengers:** Book morning departure slots (before 10:00 AM) when booking critical trips or tight connections; avoid sub-45-minute connections at high-congestion hubs like ORD or EWR during evening peak hours.

---

### Q17: What are the main limitations of this system?
**Answer:**
1. **Weather Volatility:** Severe convective thunderstorms and winter blizzards are stochastic meteorological events that cannot be anticipated months in advance from historical schedules alone.
2. **FAA Ground Delay Programs (GDP):** Real-time FAA flow management restrictions and air traffic control slot holds are tactical decisions made hours before departure and are not captured in static schedule tables.
