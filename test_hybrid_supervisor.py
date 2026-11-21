import os
import sys

sys.path.append(r"d:\Models\AeroGuard")
from aeroguard_hybrid_supervisor import (
    KinematicKalmanFilter,
    DualHorizonCUSUMTracker,
    OEMDeterministicSupervisor,
    classify_emergency_subsystem,
    generate_icao_executive_summary,
)

def run_tests():
    print("=================================================================")
    print("    AEROGUARD HYBRID SAFETY SUPERVISOR UNIT VERIFICATION SUITE   ")
    print("=================================================================\n")
    
    passed = 0
    total = 0

    def assert_test(name, condition, details=""):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"[PASS] {name} -> {details}")
        else:
            print(f"[FAIL] {name} -> {details}")

    # 1. KALMAN FILTER TESTS
    kf = KinematicKalmanFilter()
    # Initial frame
    r0 = kf.update(32000.0, -150.0, 440.0, 1000.0)
    assert_test("EKF Initialization", r0["filtered_alt"] == 32000.0, f"Filtered: {r0['filtered_alt']}")

    # Continuous noisy frames
    alts = [32050, 31960, 32030, 31970, 32010]
    filtered_alts = []
    for i, a in enumerate(alts):
        rf = kf.update(a, -150.0, 440.0, 1001.0 + i)
        filtered_alts.append(rf["filtered_alt"])
    variance_raw = sum((a - 32000)**2 for a in alts) / len(alts)
    variance_filtered = sum((fa - 32000)**2 for fa in filtered_alts) / len(filtered_alts)
    assert_test("EKF Noise Damping", variance_filtered < variance_raw, f"Raw var: {variance_raw:.1f} -> Filtered var: {variance_filtered:.1f}")

    # Sensor dropout test (12-second missing packet gap)
    r_drop = kf.update(31950.0, -150.0, 440.0, 1006.0 + 12.0)
    assert_test("EKF Dropout Bridging (12s gap)", r_drop["dropout_bridged"] == True and abs(r_drop["estimated_acc"]) < 5.0, f"Bridged: {r_drop['dropout_bridged']}, Acc: {r_drop['estimated_acc']} kts/s")

    # 2. DUAL-HORIZON CUSUM TRACKER TESTS
    cusum = DualHorizonCUSUMTracker(window_size=180)
    # 30 frames of normal cruise
    for i in range(30):
        _ = cusum.update(32000.0, 440.0, -150.0, 1000.0 + i)
    assert_test("CUSUM Baseline Lock", cusum.baseline_alt == 32000.0, f"Baseline: {cusum.baseline_alt} ft")

    # Inject insidious gradual descent (-10 ft per second over 100 seconds, low fpm)
    drift_flag = False
    for i in range(31, 140):
        # Aircraft slowly drops from 32,000 down to 30,500 ft (subtle -600 fpm descent)
        current_a = 32000.0 - (i - 30) * 12.0
        res = cusum.update(current_a, 440.0, -600.0, 1000.0 + i)
        if res["drift_detected"]:
            drift_flag = True
            break
    assert_test("CUSUM Insidious Slow-Drift Detection", drift_flag == True, f"Caught at CUSUM Score: {res.get('cusum_score')}")

    # 3. OEM DETERMINISTIC SUPERVISOR TESTS
    sup = OEMDeterministicSupervisor()
    a320_specs = {"service_ceiling": "39,800 ft (12,130 m / FL398)", "full_name": "Airbus A320-200"}

    # Test nominal cruise
    s_nom = sup.evaluate_envelope(32000.0, -150.0, 440.0, "3321", a320_specs)
    assert_test("OEM Supervisor Nominal State", s_nom["hard_override"] == False, "Envelope Nominal")

    # Test service ceiling breach (40,500 ft > 39,800 ft)
    s_ceil = sup.evaluate_envelope(40500.0, 0.0, 440.0, "3321", a320_specs)
    assert_test("OEM Service Ceiling Hard Override", s_ceil["hard_override"] == True and "SERVICE CEILING" in s_ceil["primary_violation"], s_ceil.get("primary_violation"))

    # Test low-altitude CFIT sink rate (-4,000 fpm at 6,000 ft)
    s_cfit = sup.evaluate_envelope(6000.0, -4200.0, 250.0, "3321", a320_specs)
    assert_test("OEM Low-Altitude CFIT Hard Override", s_cfit["hard_override"] == True and "LOW-ALTITUDE SINK RATE" in s_cfit["primary_violation"], s_cfit.get("primary_violation"))

    # Test Squawk 7700 tripwire
    s_sq = sup.evaluate_envelope(32000.0, -150.0, 440.0, "7700", a320_specs)
    assert_test("OEM Transponder 7700 Hard Override", s_sq["hard_override"] == True and "7700" in s_sq["primary_violation"], s_sq.get("primary_violation"))

    # 4. EMERGENCY SUBSYSTEM CLASSIFICATION TESTS (Feature #5)
    d_dive = classify_emergency_subsystem(25000.0, -5500.0, 410.0, 1.2, "7700")
    assert_test("Subsystem Diagnosis: Depressurization Dive", d_dive["code"] == "DEPRESS_DIVE", f"Diagnosed: {d_dive['category']}")

    d_roll = classify_emergency_subsystem(32000.0, -150.0, 430.0, 6.2, "3321")
    assert_test("Subsystem Diagnosis: Dutch Roll Yaw Flutter", d_roll["code"] == "DUTCH_ROLL", f"Diagnosed: {d_roll['category']}")

    d_nom = classify_emergency_subsystem(32000.0, -150.0, 440.0, 1.1, "3321")
    assert_test("Subsystem Diagnosis: Nominal Operations", d_nom["code"] == "NOMINAL", f"Diagnosed: {d_nom['category']}")

    # 5. ICAO EXECUTIVE NARRATIVE GENERATION (Feature #33)
    nar = generate_icao_executive_summary("AIC019", "Airbus A320", "AMD", "BOM", 32000.0, -150.0, 440.0, "3321", d_nom, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    assert_test("ICAO Executive Narrative Generation", "AIC019" in nar and "SHA-256" in nar, f"Length: {len(nar)} chars")

    print("\n" + "="*65)
    print(f"   HYBRID SUPERVISOR TEST RESULTS: {passed} / {total} PASS ({(passed/total)*100:.1f}%)")
    print("=================================================================\n")

if __name__ == "__main__":
    run_tests()
