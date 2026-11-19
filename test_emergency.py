
import joblib
import pandas as pd

from transmitter import activate_transmitter

MODEL_FILE = "aeroguard_model_10.pkl"
DATA_FILE = "data/real_training_dataset_10.csv"

REQUIRED_ANOMALIES = 5
MIN_CONFIDENCE = 85.0

FEATURES = [
    "altitude_mean",
    "altitude_std",
    "altitude_change",
    "velocity_mean",
    "velocity_std",
    "velocity_change",
    "heading_mean",
    "heading_std",
    "vertical_rate_mean",
    "vertical_rate_std",
    "vertical_rate_min",
    "vertical_rate_max"
]

print("Loading current AeroGuard model...")
model = joblib.load(MODEL_FILE)

print("Loading 10-observation dataset...")
df = pd.read_csv(DATA_FILE)

# Select a labelled emergency sample for a basic trigger-rule test.
samples = df[df["label"] == 1]

if samples.empty:
    raise ValueError("No labelled emergency samples found.")

sample = samples.iloc[0]
X = pd.DataFrame([sample[FEATURES].to_dict()], columns=FEATURES)

print("\n======================================")
print("AEROGUARD TRIGGER RULE TEST")
print("======================================")
print("Model:", MODEL_FILE)
print("This test uses one sample repeatedly.")
print("It does not validate real-world detection.")

anomaly_count = 0
triggered = False

for detection in range(1, REQUIRED_ANOMALIES + 1):
    probabilities = model.predict_proba(X)[0]
    classes = list(model.classes_)
    emergency_index = classes.index(1)
    prediction = int(model.predict(X)[0])
    confidence = float(probabilities[emergency_index]) * 100

    qualifies = prediction == 1 and confidence >= MIN_CONFIDENCE

    print(f"\nDetection {detection}/{REQUIRED_ANOMALIES}")
    print("Prediction:", "EMERGENCY" if prediction == 1 else "NORMAL")
    print(f"Emergency probability: {confidence:.2f}%")

    if qualifies:
        anomaly_count += 1
        print(f"Qualifying streak: {anomaly_count}/{REQUIRED_ANOMALIES}")
    else:
        anomaly_count = 0
        print("Streak reset: prediction or confidence threshold failed.")

    if anomaly_count >= REQUIRED_ANOMALIES:
        triggered = True
        break

print("\n======================================")
print("TEST RESULT")
print("======================================")

if triggered:
    print("Trigger rule satisfied.")
    print("Sending TEST-ONLY simulated emergency report...")

    activate_transmitter(
        aircraft_id="TEST-AIRCRAFT",
        callsign="TEST",
        issues=["TEST ONLY: simulated abnormal flight behaviour"],
        recording=None
    )

    print("Check data/ground_station/TEST-AIRCRAFT/ for the report.")
else:
    print("Trigger rule not satisfied by this sample.")
    print("No simulated transmission was sent.")

print("\nLive model file was not modified.")
