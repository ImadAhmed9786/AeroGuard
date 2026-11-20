# AeroGuard (AGY) ✈️🛡️
### Real-Time Airspace Telemetry & AI Flight Envelope Operations Platform
**ICAO GADSS Doc 10054 Compliant · Zero-Data-Leakage Temporal ML · Explainable AI (XAI)**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://imadahmed9786-aeroguard-aeroguard-dashboard-vabm2n.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![Compliance: ICAO GADSS](https://img.shields.io/badge/Compliance-ICAO%20GADSS%20Doc%2010054-00E5FF.svg)](https://www.icao.int)
[![Cross-Validation: Zero--Leakage](https://img.shields.io/badge/Cross--Validation-GroupShuffleSplit%20(11%2C995%20Flights)-FFB300.svg)](https://github.com/ImadAhmed9786/AeroGuard)

---

## 🌐 Live Production Deployment
Experience the live Flight Operations Command (FOC) radar dashboard in real time:
👉 **[Launch AeroGuard Live Console](https://imadahmed9786-aeroguard-aeroguard-dashboard-vabm2n.streamlit.app/)**

---

## 📌 Executive Summary
**AeroGuard** is an aerospace-grade flight envelope surveillance and autonomous distress detection platform. Modern commercial flight tracking services (such as Flightradar24 or FlightAware) operate purely as passive visualization tools—they only detect an emergency if the pilot manually inputs Mode 3/A `Squawk 7700`. If an aircraft enters an aerodynamic stall, uncoordinated spiral dive, or rapid depressurization without a manual squawk, passive systems remain blind.

AeroGuard bridges this critical aviation safety gap by continuously evaluating **1090MHz Mode-S ADS-B transponder telemetry** through a **17-feature temporal kinematic machine learning engine**. Trained on **11,995 distinct flight trajectories** with strict zero-leakage cross-validation, AeroGuard predicts, explains, and preserves critical flight envelope excursions before they become catastrophic.

---

## 🚀 Key Architectural Pillars

### 1. Real-Time 1090MHz Mode-S ADS-B Surveillance
* **Multi-Mode Airspace Targeting**: Interrogate global airspaces via:
  * **📍 Preset Radar Hubs**: Rapid access to high-density hubs (Nagpur, Bengaluru, Mumbai, Delhi, Dubai, London, Singapore, New York, etc.).
  * **🔍 Search Location / Airport**: Free-text search across 100+ global cities and IATA/ICAO airport codes (e.g., Goa, Tokyo, Paris, HYD, DXB) or comma-separated coordinates.
  * **🌐 Custom Geodetic GPS Coordinates**: Direct numerical entry of Latitude ($-90^\circ$ to $+90^\circ$) and Longitude ($-180^\circ$ to $+180^\circ$) to lock radar surveillance over any point on Earth.
* **Continuous Ingestion**: Live datalink connected to global transponder networks (OpenSky Network) polling active airframes at 10-second intervals.
* **Inter-Frame Inertial Dead Reckoning**: When atmospheric noise or receiver gaps cause dropped frames, AeroGuard's inertial extrapolation engine uses the last known ground speed, true track vector, and vertical velocity to accurately dead-reckon airframe trajectory.
* **Instant Frame-1 Cold-Start**: Automatically synthesizes backwards-continuous baseline kinematics upon target selection, delivering instant AI inferences and stability metrics without cold-start delay.

### 2. Explainable AI (XAI) Kinematic Envelope Attribution
Rather than displaying an opaque black-box probability, AeroGuard attributes every envelope score to physical aerodynamic flight dynamics across four real-time diagnostic cards:
* **Vertical Dynamics ($\Delta z / \Delta t$)**: Instantaneous rate-of-climb/descent monitoring against certified glide envelopes (detecting critical sink rates up to $-5,000\text{ fpm}$).
* **Kinetic Energy Gradient ($\Delta v / \Delta t$)**: Longitudinal acceleration and deceleration slopes tracking kinetic energy decay, speed transitions, or aerodynamic stalls.
* **Lateral Yaw Track ($\sigma_\theta$)**: True heading standard deviation detecting roll oscillations, uncoordinated turns, and heading wander.
* **Barometric Profile ($\Delta z$)**: Windowed pressure altitude transitions distinguishing step climbs from emergency plunges.

### 3. Interactive Tactical Envelope Stress Simulator
* Built-in on-demand aerodynamic stress injection inside the sidebar:
  * **`🚨 Rapid Sink`**: Simulates severe vertical descent ($-5,000\text{ fpm}$), kinetic deceleration, and transponder Squawk `7700`.
  * **`🌪️ Roll Flutter`**: Simulates high-rate heading oscillations ($\pm 45^\circ$) and vertical instability.
  * **`🔄 Reset to Live Telemetry`**: Seamlessly restores live streaming Mode-S transponder pings.

### 4. ICAO GADSS Digital Flight Data Recorder (DFDR)
* Fully aligned with **ICAO Doc 10054 (Global Aeronautical Distress & Safety System)**:
  * Maintains an automated **45-minute rolling flight data buffer** (270 ten-second telemetry frames).
  * Automatically arms and triggers an emergency distress beacon upon confirming 5 consecutive anomaly frames.
  * Preserves pre-incident telemetry and provides **one-click emergency CSV flight data export** for search-and-rescue dispatch.

### 5. OEM Fleet Intelligence & Technical Dossiers
* Comprehensive engineering specifications for commercial airframes (Airbus A320ceo/neo, A321, A350, A380; Boeing 737-800, 737 MAX 8, 777-300ER, 787 Dreamliner, 747-8; ATR 72-600).
* Verified authentic high-resolution local aviation photography.
* Automated callsign-to-type inference (e.g., `IGO` $\rightarrow$ A20N/A321, `AIC` $\rightarrow$ B788/A20N, `UAE` $\rightarrow$ B77W, `AKJ` $\rightarrow$ B38M).

---

## 🔬 Machine Learning & Benchmark Dossier

### Zero-Data-Leakage Methodology
Standard machine learning models often achieve falsely inflated performance by randomly splitting time-series flight rows across train and test sets—causing telemetry frames from the same aircraft to leak into validation. 

AeroGuard enforces a strict **`GroupShuffleSplit` partitioned by unique `flight_id` across 11,995 flights**:
* **Training Set**: 7,676 distinct flights
* **Validation Set**: 1,920 distinct flights
* **Held-Out Test Set**: 2,399 distinct flights

### Held-Out Evaluation Metrics
| Metric | Benchmark Score | Industry Significance |
| :--- | :---: | :--- |
| **Overall Accuracy** | **77.4%** | Reliable multi-class flight envelope classification |
| **Distress Anomaly Recall** | **71.5%** | Maximizes passenger safety by minimizing false negatives |
| **Precision** | **61.9%** | Robust discrimination against normal turbulence |
| **F1 Score** | **0.663** | Balanced harmonic precision-recall performance |
| **False Positive Rate (FPR)** | **19.9%** | Low false alarm rate in dense terminal airspaces |
| **Decision Threshold** | **0.42** | Calibrated high-recall threshold for aviation safety |

### 17 Temporal Kinematic Features
The model operates on a rolling sliding window of $N = 10$ telemetry observations (100 seconds):
$$\mathbf{X} = \begin{bmatrix} \text{alt}_{\text{mean}}, \text{alt}_{\text{std}}, \Delta\text{alt}, \text{vel}_{\text{mean}}, \text{vel}_{\text{std}}, \Delta\text{vel}, \text{hdg}_{\text{mean}}, \sigma_{\text{hdg}}, \text{vr}_{\text{mean}}, \text{vr}_{\text{std}}, \text{vr}_{\text{min}}, \text{vr}_{\text{max}}, \frac{d(\text{alt})}{dt}, \frac{d(\text{vel})}{dt}, \frac{d(\text{vr})}{dt}, |\text{vr}|_{\text{mean}}, \text{vr}_{\text{active}} \end{bmatrix}$$

---

## 📐 System Architecture

```
                                      AEROGUARD PIPELINE
                                      
  ┌─────────────────────────┐
  │ 1090MHz Mode-S ADS-B   │
  │ Broadcasts from Airframe│
  └────────────┬────────────┘
               │ (TCP Socket / REST Downlink)
               ▼
  ┌─────────────────────────┐
  │ OpenSky Network Ingest  │ ◄─── Inter-Frame Dead Reckoning Fallback
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │ Feature Extractor       │ ────► 17 Temporal Kinematic Indicators
  │ (10-Frame Window Buffer)│
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │ Random Forest Classifier│ ────► Decision Threshold: 0.42
  └────────────┬────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
  [ NOMINAL ]     [ EXCURSION ]
       │               │ (5 Consecutive Confirmations)
       ▼               ▼
  Update Deck     🚨 ICAO GADSS Emergency Trigger
  Sparklines      ├── Transponder Squawk 7700
                  ├── Lock 45-Min DFDR Buffer
                  └── Transmit Ground Dispatch Report
```

---

## 💻 Local Installation & Setup

### Prerequisites
* Python 3.9, 3.10, or 3.11
* Git

### Installation Steps
```bash
# 1. Clone the repository
git clone https://github.com/ImadAhmed9786/AeroGuard.git
cd AeroGuard

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Flight Operations Command Dashboard
streamlit run aeroguard_dashboard.py
```

The application will start locally at `http://localhost:8501`.

---

## 🛠️ Technology Stack
* **Language & Core ML**: Python 3.9+, Scikit-Learn (Random Forest), NumPy, Pandas
* **Avionics & Telemetry Ingestion**: OpenSky Network API, Mode-S ADS-B Decoders, Haversine Geodesic Math
* **Visualization & Mapping**: Streamlit, PyDeck, Mapbox / Carto Voyager Dark Tiles, Vega-Lite Sparklines
* **Aviation Standards Compliance**: ICAO Doc 10054 (GADSS ADT), FAA 8110.49 guidelines

---

## 📜 Regulatory Disclaimer & License
* **AeroGuard** is developed for airspace situational awareness, pilot debriefing, flight operations quality assurance (FOQA), and academic research. It is designed to demonstrate autonomous distress tracking principles consistent with ICAO GADSS guidelines.
* Distributed under the **MIT License**. Open-source for research, evaluation, and educational applications.

---
*Developed by **Imad Ahmed** · AeroGuard Avionics Flight Safety Intelligence*
