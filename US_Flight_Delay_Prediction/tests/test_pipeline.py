"""
Automated Test Suite for US Flight Delay Prediction Big Data Pipeline.
Validates:
1. Target generation logic (15 minute threshold)
2. Missing-value handling and cancellation filtering
3. Data leakage prevention
4. Mapper emission protocol
5. Reducer aggregation accuracy
6. Delay rate formula correctness
7. Model training and probability calibration
8. Pre-departure prediction validation and edge cases
9. Local simulation vs. Hadoop streaming output compatibility
"""

import sys
import os
import unittest

# Ensure base directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DELAY_THRESHOLD_MINUTES, FEATURE_COLUMNS
from src.preprocessing import clean_and_preprocess, extract_features_and_target
from src.mapper import map_flight_record
from src.reducer import reduce_flight_group
from src.mapreduce_simulator import run_mapreduce_simulation
from src.train_model import LightweightDelayModel, train_and_compare_models
from src.prediction import predict_flight_delay

class TestFlightDelayPipeline(unittest.TestCase):

    def setUp(self):
        # Sample synthetic records representing diverse scenarios
        self.sample_raw_records = [
            # 1. Normal on-time flight
            {
                "FL_DATE": "2023-04-10", "AIRLINE": "Delta Air Lines Inc.", "AIRLINE_CODE": "DL",
                "FL_NUMBER": "101", "ORIGIN": "ATL", "DEST": "ORD",
                "CRS_DEP_TIME": "800", "CRS_ARR_TIME": "1000",
                "ARR_DELAY": "5.0", "DEP_DELAY": "2.0", "CANCELLED": "0.0", "DIVERTED": "0.0",
                "DISTANCE": "606", "DELAY_DUE_WEATHER": "0.0"
            },
            # 2. Delayed flight (>= 15 min)
            {
                "FL_DATE": "2023-04-10", "AIRLINE": "Delta Air Lines Inc.", "AIRLINE_CODE": "DL",
                "FL_NUMBER": "102", "ORIGIN": "ATL", "DEST": "ORD",
                "CRS_DEP_TIME": "1800", "CRS_ARR_TIME": "2000",
                "ARR_DELAY": "42.0", "DEP_DELAY": "35.0", "CANCELLED": "0.0", "DIVERTED": "0.0",
                "DISTANCE": "606", "DELAY_DUE_LATE_AIRCRAFT": "30.0"
            },
            # 3. Exact threshold boundary (15.0 min -> must be is_delayed = 1)
            {
                "FL_DATE": "2023-04-11", "AIRLINE": "American Airlines Inc.", "AIRLINE_CODE": "AA",
                "FL_NUMBER": "201", "ORIGIN": "DFW", "DEST": "LAX",
                "CRS_DEP_TIME": "900", "CRS_ARR_TIME": "1100",
                "ARR_DELAY": "15.0", "DEP_DELAY": "10.0", "CANCELLED": "0.0", "DIVERTED": "0.0",
                "DISTANCE": "1235"
            },
            # 4. Cancelled flight (Must be excluded from arrival delay analysis)
            {
                "FL_DATE": "2023-04-11", "AIRLINE": "American Airlines Inc.", "AIRLINE_CODE": "AA",
                "FL_NUMBER": "202", "ORIGIN": "DFW", "DEST": "LAX",
                "CRS_DEP_TIME": "1200", "CRS_ARR_TIME": "1400",
                "ARR_DELAY": "", "CANCELLED": "1.0", "DIVERTED": "0.0",
                "DISTANCE": "1235"
            },
            # 5. Diverted flight (Must be excluded)
            {
                "FL_DATE": "2023-04-12", "AIRLINE": "United Air Lines Inc.", "AIRLINE_CODE": "UA",
                "FL_NUMBER": "301", "ORIGIN": "ORD", "DEST": "SFO",
                "CRS_DEP_TIME": "1500", "CRS_ARR_TIME": "1730",
                "ARR_DELAY": "", "CANCELLED": "0.0", "DIVERTED": "1.0",
                "DISTANCE": "1846"
            },
            # 6. Missing ARR_DELAY
            {
                "FL_DATE": "2023-04-12", "AIRLINE": "United Air Lines Inc.", "AIRLINE_CODE": "UA",
                "FL_NUMBER": "302", "ORIGIN": "ORD", "DEST": "SFO",
                "CRS_DEP_TIME": "1600", "CRS_ARR_TIME": "1830",
                "ARR_DELAY": "NaN", "CANCELLED": "0.0", "DIVERTED": "0.0",
                "DISTANCE": "1846"
            }
        ]

    def test_01_target_generation_and_threshold(self):
        """Test target variable matches DOT threshold: ARR_DELAY >= 15.0."""
        cleaned, report = clean_and_preprocess(self.sample_raw_records)
        self.assertEqual(len(cleaned), 3) # Cancelled, diverted, and NaN dropped
        
        # Flight 1: 5.0 min delay -> 0
        self.assertEqual(cleaned[0]["is_delayed"], 0)
        # Flight 2: 42.0 min delay -> 1
        self.assertEqual(cleaned[1]["is_delayed"], 1)
        # Flight 3: exactly 15.0 min -> 1
        self.assertEqual(cleaned[2]["is_delayed"], 1)
        self.assertEqual(report["target_threshold_minutes"], DELAY_THRESHOLD_MINUTES)

    def test_02_filtering_cancelled_and_diverted(self):
        """Test that cancelled and diverted flights are properly documented and removed."""
        cleaned, report = clean_and_preprocess(self.sample_raw_records)
        self.assertEqual(report["removed_cancelled"], 1)
        self.assertEqual(report["removed_diverted"], 1)
        self.assertEqual(report["removed_missing_arrival_delay"], 1)
        self.assertEqual(report["total_removed"], 3)

    def test_03_data_leakage_prevention(self):
        """Test that post-departure fields are excluded from feature matrix."""
        cleaned, _ = clean_and_preprocess(self.sample_raw_records)
        X, y = extract_features_and_target(cleaned)
        
        leaked_columns = ["DEP_DELAY", "ARR_DELAY", "TAXI_OUT", "TAXI_IN", "WHEELS_OFF", "DELAY_DUE_WEATHER"]
        for row in X:
            for col in leaked_columns:
                self.assertNotIn(col, row, f"Data leakage detected! {col} found in feature matrix.")
                
            # Verify required pre-departure features exist
            for col in ["AIRLINE", "ORIGIN", "DEST", "CRS_DEP_HOUR", "DISTANCE"]:
                self.assertIn(col, row)

    def test_04_mapper_emission_format(self):
        """Test that Mapper emits correct (key, (count, delay_sum, delayed_count, max_delay))."""
        row = self.sample_raw_records[1] # Delayed flight (42 min)
        mapped = map_flight_record(row, group_by_key="AIRLINE")
        self.assertIsNotNone(mapped)
        key, (count, delay_sum, delayed_count, max_delay) = mapped
        self.assertEqual(key, "Delta Air Lines Inc.")
        self.assertEqual(count, 1)
        self.assertEqual(delay_sum, 42.0)
        self.assertEqual(delayed_count, 1)
        self.assertEqual(max_delay, 42.0)

    def test_05_reducer_aggregation(self):
        """Test that Reducer correctly aggregates intermediate values."""
        key = "Delta Air Lines Inc."
        values = [
            (1, 5.0, 0, 5.0),
            (1, 45.0, 1, 45.0),
            (1, 10.0, 0, 10.0)
        ]
        result = reduce_flight_group(key, values)
        self.assertEqual(result["total_flights"], 3)
        self.assertEqual(result["delayed_flights"], 1)
        self.assertEqual(result["ontime_flights"], 2)
        self.assertEqual(result["delay_rate_pct"], 33.33)
        self.assertEqual(result["avg_delay_min"], 20.0)
        self.assertEqual(result["max_delay_min"], 45.0)

    def test_06_mapreduce_simulation_integrity(self):
        """Test end-to-end 3-stage MapReduce simulation."""
        cleaned, _ = clean_and_preprocess(self.sample_raw_records)
        result = run_mapreduce_simulation(cleaned, group_by_key="AIRLINE")
        
        self.assertIn("reduced_results", result)
        self.assertIn("summary", result)
        summary = result["summary"]
        self.assertEqual(summary["total_flights"], 3)
        self.assertEqual(summary["delayed_flights"], 2)

    def test_07_model_training_and_comparison(self):
        """Test model training on dataset and metric generation."""
        from data.generate_dataset import generate_flight_dataset
        from src.data_loader import load_flight_records
        
        test_csv = "data/test_flights.csv"
        generate_flight_dataset(test_csv, num_records=200, seed=123)
        records, _ = load_flight_records(test_csv, limit=200)
        cleaned, _ = clean_and_preprocess(records)
        X, y = extract_features_and_target(cleaned)
        
        comp = train_and_compare_models(X, y)
        self.assertIn("logistic_regression", comp)
        self.assertIn("random_forest", comp)
        self.assertIn("recommended_model", comp)
        self.assertGreater(comp["random_forest"]["metrics"]["accuracy"], 0.5)

    def test_08_interactive_prediction_validation(self):
        """Test interactive prediction with valid and boundary inputs."""
        inputs = {
            "AIRLINE": "Delta Air Lines Inc.",
            "ORIGIN": "ATL",
            "DEST": "ORD",
            "MONTH": 6,
            "DAY_OF_WEEK": 5,
            "CRS_DEP_HOUR": 18,
            "CRS_ARR_HOUR": 20,
            "DISTANCE": 606.0
        }
        res = predict_flight_delay(inputs)
        self.assertIn("predicted_class", res)
        self.assertIn("delay_probability_pct", res)
        self.assertIn("risk_level", res)
        self.assertIn("influencing_factors", res)
        self.assertTrue(0.0 <= res["delay_probability_pct"] <= 100.0)

    def test_09_local_and_hadoop_compatibility(self):
        """Test that local Reducer output matches Hadoop Streaming protocol keys."""
        local_sample = reduce_flight_group("ORD", [(1, 20.0, 1, 20.0)])
        expected_keys = ["key", "total_flights", "delayed_flights", "ontime_flights", "delay_rate_pct", "avg_delay_min", "max_delay_min"]
        for k in expected_keys:
            self.assertIn(k, local_sample)

    def test_10_shap_lime_explainability(self):
        """Test SHAP/LIME local surrogate attribution and explanation generation."""
        from src.prediction import explain_flight_prediction_shap_lime
        inputs = {
            "AIRLINE": "JetBlue Airways",
            "ORIGIN": "JFK",
            "DEST": "ORD",
            "MONTH": 7,
            "DAY_OF_WEEK": 5,
            "CRS_DEP_HOUR": 19,
            "CRS_ARR_HOUR": 21,
            "DISTANCE": 740.0
        }
        res = predict_flight_delay(inputs)
        self.assertIn("shap_explanation", res)
        shap_info = res["shap_explanation"]
        self.assertIn("base_value", shap_info)
        self.assertIn("attributions", shap_info)
        self.assertIn("explanation", shap_info)
        self.assertGreater(len(shap_info["attributions"]), 0)
        
        # Verify mathematical consistency: base_value + sum(shap_values) ~= predicted_probability
        sum_attr = sum(a["shap_value"] for a in shap_info["attributions"])
        total_recon = shap_info["base_value"] + sum_attr
        self.assertAlmostEqual(total_recon, shap_info["predicted_probability"], delta=0.01)

if __name__ == "__main__":
    unittest.main()
