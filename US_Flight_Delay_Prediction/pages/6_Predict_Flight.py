"""
Page 6: Interactive Flight Delay Prediction & SHAP / LIME Interpretability
Provides:
- Validated pre-departure flight delay probability & risk classification
- SHAP / LIME local surrogate attribution waterfall decomposition
- Understandable natural language explanation of model predictions
"""

import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prediction import predict_flight_delay, KNOWN_AIRPORTS, KNOWN_AIRLINES
from src.train_model import LightweightDelayModel

st.set_page_config(page_title="Predict a Flight | SHAP/LIME Explainability", layout="wide")

st.title("✈️ Predict a Flight Delay & SHAP / LIME Explainability")
st.markdown("""
Enter scheduled pre-departure parameters to forecast arrival delay risk (&ge; 15 min).
Every prediction includes a **SHAP / LIME feature attribution waterfall** explaining precisely why the model predicted a delay or on-time arrival.
""")

st.info("⚠️ **Pre-Departure Principle:** Evaluates only features known before gate pushback. Post-departure factors (actual departure delay, taxi time) are strictly excluded to avoid data leakage.")

# Quick Scenario Presets
st.markdown("##### ⚡ Quick Scenario Presets")
col_p1, col_p2, col_p3 = st.columns(3)
preset_selected = None
with col_p1:
    if st.button("🔴 Friday Evening JFK ➔ ORD (Peak Delay Risk)"):
        preset_selected = ("JetBlue Airways", "JFK", "ORD", 7, 5, 19, 21, 740.0)
with col_p2:
    if st.button("🟢 Tuesday Morning ATL ➔ CLT (Low Delay Risk)"):
        preset_selected = ("Delta Air Lines Inc.", "ATL", "CLT", 10, 2, 7, 8, 226.0)
with col_p3:
    if st.button("🟡 Sunday Transcon SFO ➔ EWR (Medium Risk)"):
        preset_selected = ("United Air Lines Inc.", "SFO", "EWR", 12, 7, 16, 23, 2565.0)

# Form defaults
def_airline = preset_selected[0] if preset_selected else "Delta Air Lines Inc."
def_origin = preset_selected[1] if preset_selected else "ORD"
def_dest = preset_selected[2] if preset_selected else "LAX"
def_month = preset_selected[3] if preset_selected else 7
def_dow = preset_selected[4] if preset_selected else 5
def_dep = preset_selected[5] if preset_selected else 18
def_arr = preset_selected[6] if preset_selected else 21
def_dist = preset_selected[7] if preset_selected else 1200.0

with st.form("flight_prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        airline = st.selectbox("Operating Airline:", KNOWN_AIRLINES, index=KNOWN_AIRLINES.index(def_airline) if def_airline in KNOWN_AIRLINES else 0)
        origin = st.selectbox("Origin Airport:", list(KNOWN_AIRPORTS.keys()), index=list(KNOWN_AIRPORTS.keys()).index(def_origin) if def_origin in KNOWN_AIRPORTS else 0)
        dest = st.selectbox("Destination Airport:", [a for a in KNOWN_AIRPORTS.keys() if a != origin], index=0)
        distance = st.number_input("Flight Distance (miles):", min_value=100.0, max_value=4000.0, value=float(def_dist), step=50.0)
    with col2:
        month = st.slider("Month of Travel:", min_value=1, max_value=12, value=int(def_month))
        day_of_week = st.selectbox("Day of Week:", [
            (1, "Monday"), (2, "Tuesday"), (3, "Wednesday"), (4, "Thursday"),
            (5, "Friday"), (6, "Saturday"), (7, "Sunday")
        ], format_func=lambda x: x[1], index=int(def_dow) - 1)
        dep_hour = st.slider("Scheduled Departure Hour (24-Hour format):", min_value=5, max_value=23, value=int(def_dep))
        arr_hour = st.slider("Scheduled Arrival Hour (24-Hour format):", min_value=6, max_value=23, value=int(def_arr))

    submitted = st.form_submit_button("🚀 Run Pre-Departure Delay Inference")

if submitted or preset_selected:
    inputs = {
        "AIRLINE": airline,
        "ORIGIN": origin,
        "DEST": dest,
        "MONTH": month,
        "DAY_OF_WEEK": day_of_week[0] if isinstance(day_of_week, tuple) else day_of_week,
        "CRS_DEP_HOUR": dep_hour,
        "CRS_ARR_HOUR": arr_hour,
        "DISTANCE": distance
    }
    
    result = predict_flight_delay(inputs)
    shap_data = result.get("shap_explanation", {})
    
    st.markdown("---")
    st.subheader("1. Inference Result & Risk Assessment")
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        if result["predicted_class"] == 1:
            st.error(f"### Predicted Class: {result['predicted_label']}")
        else:
            st.success(f"### Predicted Class: {result['predicted_label']}")
        st.metric(
            "Estimated Delay Probability (&ge; 15m)",
            f"{result['delay_probability_pct']}%",
            f"Risk Level: {result['risk_level']}"
        )
        st.caption(f"Operational threshold: 35.0% | Baseline dataset prior: {shap_data.get('base_value_pct', 22.0)}%")
        
    with res_col2:
        st.markdown("#### Model Decision Explanation (SHAP / LIME)")
        st.markdown(f"> *{shap_data.get('explanation', '')}*")
        
    st.markdown("---")
    st.subheader("2. SHAP / LIME Local Feature Attribution Waterfall")
    st.markdown("""
    In cooperative game theory (SHAP) and local surrogate modeling (LIME), the prediction is decomposed as:
    $$\\text{Predicted Probability } f(x) = E[f(x)] + \\sum_{i=1}^M \\phi_i(x)$$
    Below is each feature's marginal push towards (red / positive) or away from (green / negative) delay:
    """)
    
    attributions = shap_data.get("attributions", [])
    if attributions:
        attr_display = []
        for a in attributions:
            direction = "🔺 Increases Delay" if a["shap_value"] > 0 else "🔻 Decreases Delay"
            sign_str = f"+{a['shap_pct']}%" if a["shap_value"] > 0 else f"{a['shap_pct']}%"
            attr_display.append({
                "Feature": a["feature"],
                "Input Value": a["value"],
                "SHAP Attribution": sign_str,
                "Effect Direction": direction,
                "Domain Rationale": a["description"]
            })
        st.table(attr_display)
        
        # Attribution Bar Chart
        st.markdown("#### Visual Feature Attribution Contributions (Percentage Points)")
        chart_dict = {f"{a['feature']} ({a['value']})": a["shap_pct"] for a in attributions}
        st.bar_chart(chart_dict)
        
    st.caption(f"**Operational Disclaimer:** {result['disclaimer']}")
