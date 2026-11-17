"""
Page 7: Business Insights & Strategic Recommendations
"""
import streamlit as st

st.title("💡 Business Insights & Strategic Recommendations")
st.markdown("Actionable operational strategies derived from empirical Big Data analytics.")

st.subheader("1. Core Empirical Findings")
st.markdown("""
1. **Afternoon Network Cascade:**
   Delays compound cumulatively throughout the day. A flight departing at 07:00 has an ~11% probability of delay, whereas an identical route departing after 17:00 exceeds a 32% delay probability. This reflects turnaround buffer deficits across rotating aircraft tail numbers.
   
2. **Hub Congestion Asymmetry:**
   Runway-constrained airports (EWR, JFK, ORD, SFO) demonstrate a 1.6x delay rate multiplier compared to secondary focus cities. Ground taxi delays and air traffic metering during adverse weather account for the majority of severe (>45m) arrival delays.
   
3. **Turnaround Vulnerability in Low-Cost Carriers:**
   Airlines operating high aircraft utilization (e.g., 30-minute turnarounds) show faster delay propagation compared to legacy carriers possessing banked spare aircraft at core hubs.
""")

st.subheader("2. Operational Recommendations for Airlines")
st.markdown("""
- **Dynamic Turnaround Padding:** Allocate an additional 12–18 minutes of turnaround slack for tail numbers arriving at congested hubs after 16:00.
- **Predictive Crew Reserve Allocation:** Stage standby flight crews at secondary hubs when inbound flights experience cascading delay probabilities exceeding 40%.
- **Proactive Passenger Communication:** Notify connecting passengers of high delay likelihoods 4 hours before departure to facilitate voluntary rebooking and reduce gate customer service queues.
""")

st.subheader("3. Recommendations for Passengers")
st.markdown("""
- **Book Morning Flights:** Choose departures before 09:30 AM whenever schedule punctuality is mission-critical.
- **Buffer Connecting Layovers:** Ensure at least 90 minutes of connection time when transiting through major hubs (ORD, DFW, ATL, JFK) during late afternoon or evening hours.
""")

st.subheader("4. Technical & Domain Limitations")
st.markdown("""
- **Correlation Does Not Equal Causation:** High delay rates at a specific hub do not necessarily mean the local airport management is at fault; external airspace weather fronts often cause incoming delays.
- **Tactical Weather & FAA Airspace Directives:** Real-time convective squalls and FAA Ground Delay Programs cannot be fully anticipated months in advance from static schedule tables alone.
""")
