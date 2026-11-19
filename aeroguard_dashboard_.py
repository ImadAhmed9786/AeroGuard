
from __future__ import annotations

import ast
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from opensky_api import OpenSkyApi, TokenManager
from streamlit_geolocation import streamlit_geolocation


# ============================================================
# AEROGUARD CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CREDENTIALS_FILE = ROOT / "credentials.json"
MODEL_FILE = DATA_DIR / "aeroguard_temporal_candidate.pkl"
MONITOR_FILE = ROOT / "aeroguard_single_candidate.py"

BUFFER_SECONDS = 45 * 60
POLL_SECONDS = 10
AI_WINDOW = 10
ANOMALY_REQUIRED = 5
DEFAULT_THRESHOLD = 0.42

FEATURE_NAMES = [
    "altitude_mean", "altitude_std", "altitude_change",
    "velocity_mean", "velocity_std", "velocity_change",
    "heading_mean", "heading_std", "vertical_rate_mean",
    "vertical_rate_std", "vertical_rate_min", "vertical_rate_max",
    "altitude_slope", "velocity_slope", "vertical_rate_slope",
    "vertical_rate_abs_mean", "vertical_rate_nonzero_fraction",
]

CITIES = {
    "Bengaluru": (12.9716, 77.5946),
    "Mumbai": (19.0760, 72.8777),
    "Nagpur": (21.1458, 79.0882),
    "Delhi": (28.6139, 77.2090),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Kochi": (9.9312, 76.2673),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
}

st.set_page_config(
    page_title="AeroGuard Flight Monitor",
    page_icon="✈️",
    layout="wide",
)


# ============================================================
# MODEL AND API
# ============================================================

@st.cache_resource
def load_model():
    artifact = joblib.load(MODEL_FILE)

    if isinstance(artifact, dict):
        model = next(
            (
                artifact[key]
                for key in (
                    "model", "pipeline", "classifier",
                    "candidate_model"
                )
                if hasattr(artifact.get(key), "predict_proba")
            ),
            None,
        )

        if model is None:
            estimator = artifact.get("estimator")
            if hasattr(estimator, "predict_proba"):
                model = estimator

        threshold = float(
            artifact.get(
                "threshold",
                artifact.get("model_threshold", DEFAULT_THRESHOLD),
            )
        )
        names = artifact.get(
            "feature_names",
            artifact.get("features", FEATURE_NAMES),
        )
        window = int(artifact.get("window_size", AI_WINDOW))
    else:
        model = artifact
        threshold = DEFAULT_THRESHOLD
        names = FEATURE_NAMES
        window = AI_WINDOW

    if model is None or not hasattr(model, "predict_proba"):
        raise ValueError(
            "Candidate model does not expose predict_proba()."
        )

    if list(names) != FEATURE_NAMES:
        raise ValueError("Model feature names/order do not match.")

    if window != AI_WINDOW:
        raise ValueError(
            f"Expected a {AI_WINDOW}-observation model window."
        )

    return model, threshold


@st.cache_resource
def load_feature_function():
    source = MONITOR_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    wanted = {
        "create_features",
        "heading_statistics",
        "calculate_slope",
    }

    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in wanted
    ]

    if not any(node.name == "create_features" for node in nodes):
        raise ValueError(
            "create_features() was not found in the monitor script."
        )

    namespace = {
        "np": np,
        "pd": pd,
        "FEATURE_NAMES": FEATURE_NAMES,
        "AI_WINDOW": AI_WINDOW,
    }

    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            str(MONITOR_FILE),
            "exec",
        ),
        namespace,
    )

    return namespace["create_features"]


@st.cache_resource
def load_api():
    token_manager = TokenManager.from_json_file(
        str(CREDENTIALS_FILE)
    )
    return OpenSkyApi(token_manager=token_manager)


# ============================================================
# LOCATION AND DISTANCE
# ============================================================

def distance_km(lat1, lon1, lat2, lon2):
    radius = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    value = (
        math.sin(dp / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )

    return 2 * radius * math.asin(
        min(1.0, math.sqrt(value))
    )



def discover_flights(api, lat, lon, radius_km):
    lat_delta = radius_km / 111.0
    cos_lat = max(
        0.05,
        abs(math.cos(math.radians(lat))),
    )
    lon_delta = min(180.0, radius_km / (111.0 * cos_lat))

    min_lat = max(-90.0, lat - lat_delta)
    max_lat = min(90.0, lat + lat_delta)
    min_lon = max(-180.0, lon - lon_delta)
    max_lon = min(180.0, lon + lon_delta)

    # Correct OpenSky bounding-box order:
    # minimum latitude, maximum latitude,
    # minimum longitude, maximum longitude.
    response = api.get_states(
        bbox=(min_lat, max_lat, min_lon, max_lon)
    )

    if response is None:
        raise RuntimeError(
            "OpenSky returned no response. Check API credentials, "
            "rate limits, and network connectivity."
        )

    if not response.states:
        return []

    flights = []

    for state in response.states:
        if not state.icao24:
            continue
        if state.latitude is None or state.longitude is None:
            continue
        if state.on_ground is not False:
            continue

        distance = distance_km(
            lat,
            lon,
            float(state.latitude),
            float(state.longitude),
        )

        if distance > radius_km:
            continue

        altitude = (
            state.geo_altitude
            if state.geo_altitude is not None
            else state.baro_altitude
        )

        flights.append({
            "icao24": state.icao24.lower(),
            "callsign": (state.callsign or "").strip() or "UNKNOWN",
            "latitude": float(state.latitude),
            "longitude": float(state.longitude),
            "altitude": altitude,
            "velocity": state.velocity,
            "heading": state.true_track,
            "vertical_rate": state.vertical_rate,
            "distance_km": distance,
        })

    return sorted(flights, key=lambda item: item["distance_km"])


# ============================================================
# FLIGHT ROUTE LOOKUP
# ============================================================

def lookup_route(callsign):
    clean = (callsign or "").strip().replace(" ", "")

    if not clean or clean.upper() == "UNKNOWN":
        return None

    try:
        response = requests.get(
            f"https://api.adsbdb.com/v0/callsign/{clean}",
            timeout=8,
            headers={
                "User-Agent": "AeroGuard-Research-Prototype/1.0"
            },
        )

        if response.status_code != 200:
            return None

        payload = response.json().get("response", {})
        route_data = payload.get("flightroute") or {}

        def airport(item):
            if not isinstance(item, dict):
                return None

            lat = item.get("latitude")
            lon = item.get("longitude")

            if lat is None or lon is None:
                return None

            return {
                "name": (
                    item.get("name")
                    or item.get("icao_code")
                    or item.get("iata_code")
                    or "Airport"
                ),
                "icao": item.get("icao_code") or "",
                "iata": item.get("iata_code") or "",
                "latitude": float(lat),
                "longitude": float(lon),
            }

        origin = airport(route_data.get("origin"))
        destination = airport(route_data.get("destination"))

        if origin and destination:
            return {
                "origin": origin,
                "destination": destination,
                "source": "ADSBdb",
            }

    except (
        requests.RequestException,
        ValueError,
        TypeError,
    ):
        pass

    return None


# ============================================================
# FLIGHT MAP
# ============================================================

def make_route_map(route, track):
    """Build a map that fits all displayed points and labels route data honestly."""
    layers = []
    coordinates = []

    if route:
        origin = route["origin"]
        destination = route["destination"]

        source_pos = [origin["longitude"], origin["latitude"]]
        destination_pos = [destination["longitude"], destination["latitude"]]
        coordinates.extend([source_pos, destination_pos])

        # This is a straight-line reference only, not a verified flight path.
        layers.append(
            pdk.Layer(
                "LineLayer",
                data=[{"source": source_pos, "target": destination_pos}],
                get_source_position="source",
                get_target_position="target",
                get_color=[240, 65, 75, 190],
                get_width=3,
                width_min_pixels=2,
                pickable=True,
            )
        )

        airports = [
            {
                "name": f"ORIGIN · {origin['name']}",
                "position": source_pos,
                "color": [30, 220, 90, 255],
            },
            {
                "name": f"DESTINATION · {destination['name']}",
                "position": destination_pos,
                "color": [245, 55, 65, 255],
            },
        ]

        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=airports,
                get_position="position",
                get_fill_color="color",
                get_radius=7000,
                radius_min_pixels=7,
                radius_max_pixels=14,
                pickable=True,
            )
        )
        layers.append(
            pdk.Layer(
                "TextLayer",
                data=airports,
                get_position="position",
                get_text="name",
                get_color=[255, 255, 255, 255],
                get_size=12,
                get_pixel_offset=[0, -14],
                pickable=True,
            )
        )

    valid_track = []
    for item in track:
        try:
            lat = float(item["latitude"])
            lon = float(item["longitude"])
            if (
                math.isfinite(lat)
                and math.isfinite(lon)
                and -90 <= lat <= 90
                and -180 <= lon <= 180
            ):
                valid_track.append({
                    "latitude": lat,
                    "longitude": lon,
                    "timestamp": item.get("timestamp", ""),
                })
        except (KeyError, TypeError, ValueError):
            continue

    valid_track.sort(key=lambda item: item["timestamp"])

    if valid_track:
        positions = [
            [item["longitude"], item["latitude"]]
            for item in valid_track
        ]
        coordinates.extend(positions)

        if len(positions) >= 2:
            layers.append(
                pdk.Layer(
                    "PathLayer",
                    data=[{"path": positions}],
                    get_path="path",
                    get_color=[40, 210, 115, 240],
                    get_width=4,
                    width_min_pixels=3,
                    pickable=True,
                )
            )

        # Latest observed telemetry is the live-position marker.
        current = positions[-1]
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=[{"position": current, "name": "Latest observed aircraft position"}],
                get_position="position",
                get_fill_color=[255, 145, 25, 255],
                get_line_color=[255, 255, 255, 255],
                get_line_width=2,
                stroked=True,
                get_radius=5000,
                radius_min_pixels=8,
                radius_max_pixels=14,
                pickable=True,
            )
        )

    if not coordinates:
        return None

    # Fit the view to every displayed airport and observed aircraft position.
    longitudes = [point[0] for point in coordinates]
    latitudes = [point[1] for point in coordinates]
    min_lon, max_lon = min(longitudes), max(longitudes)
    min_lat, max_lat = min(latitudes), max(latitudes)
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2

    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat
    span = max(lon_span, lat_span)

    if span < 0.25:
        zoom = 8
    elif span < 0.5:
        zoom = 7
    elif span < 1:
        zoom = 6
    elif span < 2:
        zoom = 5.5
    elif span < 5:
        zoom = 4.5
    elif span < 10:
        zoom = 3.8
    elif span < 20:
        zoom = 3
    elif span < 40:
        zoom = 2.3
    else:
        zoom = 1.5

    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=zoom,
            pitch=0,
            bearing=0,
        ),
        tooltip={"text": "{name}"},
        map_style="dark",
    )


# ============================================================
# EMERGENCY SIMULATION
# ============================================================

def save_emergency(aircraft_id, callsign, rows, issue):
    folder = DATA_DIR / "emergencies" / aircraft_id
    folder.mkdir(parents=True, exist_ok=True)

    history_file = folder / "previous_45_minutes.csv"

    pd.DataFrame(rows).drop(
        columns=["timestamp_unix"],
        errors="ignore",
    ).to_csv(history_file, index=False)

    ground_folder = DATA_DIR / "ground_station" / aircraft_id
    ground_folder.mkdir(parents=True, exist_ok=True)

    report_file = ground_folder / "emergency_report.txt"

    report_file.write_text(
        "AEROGUARD SIMULATED EMERGENCY REPORT\n"
        f"UTC: {datetime.now(timezone.utc).isoformat()}\n"
        f"Aircraft: {aircraft_id}\n"
        f"Callsign: {callsign}\n"
        f"Issue: {issue}\n"
        f"Buffered observations: {len(rows)}\n"
        f"Recording: {history_file}\n"
        "SIMULATION ONLY. NO REAL ATC CONNECTION.\n",
        encoding="utf-8",
    )

    try:
        from transmitter import activate_transmitter

        return activate_transmitter(
            aircraft_id=aircraft_id,
            callsign=callsign,
            issues=[issue],
            recording=str(history_file),
        )
    except Exception:
        return False


# ============================================================
# SESSION STATE
# ============================================================

def initialize():
    defaults = {
        "flights": [],
        "selected_id": None,
        "selected_callsign": None,
        "recording": [],
        "ai_history": [],
        "track": [],
        "last_timestamp": None,
        "last_poll": 0.0,
        "last_score": None,
        "ai_status": "Waiting for data",
        "anomaly_count": 0,
        "emergency": False,
        "emergency_issue": None,
        "route": None,
        "route_checked": False,
        "poll_status": "Waiting for first poll",
        "location_lat": None,
        "location_lon": None,
        "location_source": "Not set",
        "ground_report": None,
        "last_snapshot_time": 0.0,
        "location_attempted": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def select_flight(flight):
    st.session_state.selected_id = flight["icao24"]
    st.session_state.selected_callsign = flight["callsign"]
    st.session_state.recording = []
    st.session_state.ai_history = []
    st.session_state.track = []
    st.session_state.last_timestamp = None
    st.session_state.last_poll = 0.0
    st.session_state.last_score = None
    st.session_state.ai_status = "Waiting for 10 observations"
    st.session_state.anomaly_count = 0
    st.session_state.emergency = False
    st.session_state.emergency_issue = None
    st.session_state.route = None
    st.session_state.route_checked = False
    st.session_state.ground_report = None
    st.session_state.last_snapshot_time = 0.0


# ============================================================
# TELEMETRY AND AI
# ============================================================

def poll_flight(api, model, threshold, create_features):
    response = api.get_states(
        icao24=[st.session_state.selected_id]
    )

    if response is None or not response.states:
        return "OpenSky returned no states."

    state = next(
        (
            item for item in response.states
            if item.icao24
            and item.icao24.lower() == st.session_state.selected_id
        ),
        None,
    )

    if state is None:
        return "Selected aircraft not returned in this poll."

    if state.time_position is None:
        return "Waiting for a fresh position timestamp."

    now = time.time()
    timestamp = float(state.time_position)

    if now - timestamp > 60 or timestamp - now > 10:
        return "Stale telemetry ignored."

    previous = st.session_state.last_timestamp

    if previous is not None and timestamp <= previous:
        return "Duplicate timestamp ignored; waiting for next update."

    if state.latitude is None or state.longitude is None:
        return "Aircraft position unavailable in this update."

    record = {
        "timestamp": datetime.fromtimestamp(
            timestamp, timezone.utc
        ).isoformat(),
        "timestamp_unix": timestamp,
        "icao24": state.icao24.lower(),
        "callsign": (state.callsign or "").strip() or "UNKNOWN",
        "latitude": state.latitude,
        "longitude": state.longitude,
        "baro_altitude": state.baro_altitude,
        "geo_altitude": state.geo_altitude,
        "velocity": state.velocity,
        "heading": state.true_track,
        "vertical_rate": state.vertical_rate,
        "on_ground": state.on_ground,
        "squawk": state.squawk,
    }

    if previous is not None and timestamp - previous > 30:
        st.session_state.ai_history = []
        st.session_state.anomaly_count = 0

    st.session_state.last_timestamp = timestamp
    st.session_state.recording.append(record)

    cutoff = timestamp - BUFFER_SECONDS

    st.session_state.recording = [
        item for item in st.session_state.recording
        if item["timestamp_unix"] >= cutoff
    ]

    st.session_state.track.append({
        "latitude": float(state.latitude),
        "longitude": float(state.longitude),
        "timestamp": record["timestamp"],
    })

    st.session_state.track = st.session_state.track[-2000:]

    if st.session_state.emergency:
        post_file = (
            DATA_DIR / "ground_station"
            / st.session_state.selected_id
            / "post_trigger_telemetry.csv"
        )

        post_file.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame([record]).drop(
            columns=["timestamp_unix"],
            errors="ignore",
        ).to_csv(
            post_file,
            mode="a",
            header=not post_file.exists(),
            index=False,
        )

    st.session_state.ai_history.append(record)

    if len(st.session_state.ai_history) >= AI_WINDOW:
        batch = list(st.session_state.ai_history[:AI_WINDOW])
        st.session_state.ai_history = (
            st.session_state.ai_history[AI_WINDOW:]
        )

        features = create_features(batch)

        if isinstance(features, pd.DataFrame):
            X = features[FEATURE_NAMES]
        elif isinstance(features, dict):
            X = pd.DataFrame(
                [[features[name] for name in FEATURE_NAMES]],
                columns=FEATURE_NAMES,
            )
        else:
            X = pd.DataFrame(
                np.asarray(features).reshape(1, -1),
                columns=FEATURE_NAMES,
            )

        probabilities = model.predict_proba(X)[0]
        classes = list(model.classes_)
        index = classes.index(1) if 1 in classes else 0
        score = float(probabilities[index])

        st.session_state.last_score = score
        st.session_state.ai_status = (
            "ANOMALY CANDIDATE"
            if score >= threshold
            else "NORMAL"
        )

        if score >= threshold:
            st.session_state.anomaly_count += 1
        else:
            st.session_state.anomaly_count = 0

        if (
            st.session_state.anomaly_count >= ANOMALY_REQUIRED
            and not st.session_state.emergency
        ):
            fire_emergency("AI anomaly threshold exceeded")

    return "Telemetry updated."


def fire_emergency(issue):
    if st.session_state.emergency:
        return

    st.session_state.emergency = True
    st.session_state.emergency_issue = issue

    success = save_emergency(
        st.session_state.selected_id,
        st.session_state.selected_callsign,
        st.session_state.recording,
        issue,
    )

    st.session_state.ground_report = (
        "Transmitter simulation successful"
        if success
        else "Report saved; check transmitter output"
    )


# ============================================================
# DASHBOARD
# ============================================================

def main():
    initialize()

    st.title("✈️ AeroGuard")
    st.caption(
        "Live telemetry · 45-minute rolling buffer · "
        "AI candidate · simulated ground station"
    )

    st.warning(
        "Research prototype only. AI alerts are not verified "
        "emergencies. No real ATC connection."
    )

    with st.sidebar:
        st.header("📍 Location and flight discovery")

        st.write("**Option 1: Detect your current location**")
        st.caption(
            "Allow browser location access when prompted. "
            "Location detection requires browser permission."
        )

        try:
            location = streamlit_geolocation()

            if isinstance(location, dict):
                browser_lat = location.get("latitude")
                browser_lon = location.get("longitude")

                if browser_lat is not None and browser_lon is not None:
                    browser_lat = float(browser_lat)
                    browser_lon = float(browser_lon)

                    if (
                        -90 <= browser_lat <= 90
                        and -180 <= browser_lon <= 180
                    ):
                        st.session_state.location_lat = browser_lat
                        st.session_state.location_lon = browser_lon
                        st.session_state.location_source = (
                            "Browser GPS/location permission"
                        )
                        st.success("Your location was detected.")
        except Exception as exc:
            st.info(
                "Browser location is not available right now. "
                "Choose a location manually below."
            )

        if (
            st.session_state.location_lat is not None
            and st.session_state.location_lon is not None
        ):
            st.caption(
                f"Current reference: "
                f"{st.session_state.location_lat:.4f}, "
                f"{st.session_state.location_lon:.4f}"
            )
            st.caption(st.session_state.location_source)

        st.divider()
        st.write("**Option 2: Select a location manually**")

        city = st.selectbox(
            "Choose a city",
            ["Use detected location"] + list(CITIES.keys()) + ["Custom coordinates"],
        )

        if city in CITIES:
            if st.button("Use selected city"):
                city_lat, city_lon = CITIES[city]
                st.session_state.location_lat = city_lat
                st.session_state.location_lon = city_lon
                st.session_state.location_source = f"Manual: {city}"
                st.session_state.flights = []
                st.rerun()

        elif city == "Custom coordinates":
            custom_lat = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=(
                    float(st.session_state.location_lat)
                    if st.session_state.location_lat is not None
                    else 19.0760
                ),
                format="%.5f",
                key="custom_lat",
            )

            custom_lon = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=(
                    float(st.session_state.location_lon)
                    if st.session_state.location_lon is not None
                    else 72.8777
                ),
                format="%.5f",
                key="custom_lon",
            )

            if st.button("Use custom coordinates"):
                st.session_state.location_lat = custom_lat
                st.session_state.location_lon = custom_lon
                st.session_state.location_source = "Manual coordinates"
                st.session_state.flights = []
                st.rerun()

        st.divider()

        radius_km = st.slider(
            "Search radius (km)",
            min_value=25,
            max_value=500,
            value=150,
            step=25,
        )

        if st.button("🔎 Find nearby flights", type="primary"):
            if (
                st.session_state.location_lat is None
                or st.session_state.location_lon is None
            ):
                st.error(
                    "Allow location access or select a city first."
                )
            else:
                try:
                    with st.spinner("Searching OpenSky aircraft data..."):
                        st.session_state.flights = discover_flights(
                            load_api(),
                            st.session_state.location_lat,
                            st.session_state.location_lon,
                            radius_km,
                        )

                    if not st.session_state.flights:
                        st.warning(
                            "No airborne aircraft found within this "
                            "radius. Try a larger radius or another city."
                        )
                except Exception as exc:
                    st.error(f"Flight search failed: {exc}")

        flights = st.session_state.flights

        if flights:
            st.success(f"{len(flights)} aircraft found")

            options = {
                (
                    f"{flight['callsign']} · {flight['icao24']} · "
                    f"{flight['distance_km']:.1f} km away"
                ): flight
                for flight in flights
            }

            labels = list(options)

            chosen_label = st.selectbox(
                "Choose an aircraft",
                labels,
                key="flight_choice",
            )

            if st.button("✈️ Monitor selected flight"):
                select_flight(options[chosen_label])
                st.rerun()

            st.caption(
                "Aircraft are sorted by distance from your chosen "
                "reference location."
            )

    if not st.session_state.selected_id:
        st.info(
            "Detect your location or choose a city, find nearby "
            "flights, and select an aircraft to monitor."
        )
        return

    try:
        model, threshold = load_model()
        create_features = load_feature_function()
        api = load_api()
    except Exception as exc:
        st.error(f"Dashboard setup failed: {exc}")
        return

    st.subheader(
        f"Monitoring {st.session_state.selected_callsign} "
        f"· {st.session_state.selected_id}"
    )

    if not st.session_state.route_checked:
        st.session_state.route = lookup_route(
            st.session_state.selected_callsign
        )
        st.session_state.route_checked = True

    @st.fragment(run_every="10s")
    def live_panel():
        try:
            if time.time() - st.session_state.last_poll >= POLL_SECONDS:
                st.session_state.last_poll = time.time()
                st.session_state.poll_status = poll_flight(
                    api,
                    model,
                    threshold,
                    create_features,
                )
        except Exception as exc:
            st.session_state.poll_status = f"Polling error: {exc}"

        rows = st.session_state.recording

        coverage = (
            (
                rows[-1]["timestamp_unix"]
                - rows[0]["timestamp_unix"]
            ) / 60
            if len(rows) > 1 else 0
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Observations", len(rows))
        c2.metric("Buffer coverage", f"{coverage:.1f} / 45 min")
        c3.metric("AI threshold", f"{threshold:.0%}")
        c4.metric("AI status", st.session_state.ai_status)

        st.caption(f"Status: {st.session_state.poll_status}")

        if rows:
            latest = rows[-1]
            a, b, c = st.columns(3)

            a.metric(
                "Altitude",
                (
                    f"{latest['baro_altitude']:.0f} m"
                    if pd.notna(latest["baro_altitude"])
                    else "N/A"
                ),
            )

            b.metric(
                "Speed",
                (
                    f"{latest['velocity']:.1f} m/s"
                    if pd.notna(latest["velocity"])
                    else "N/A"
                ),
            )

            c.metric(
                "Vertical rate",
                (
                    f"{latest['vertical_rate']:.2f} m/s"
                    if pd.notna(latest["vertical_rate"])
                    else "N/A"
                ),
            )

            score = st.session_state.last_score

            st.caption(
                "AI score: "
                + (f"{score:.1%}" if score is not None
                   else "Waiting for 10 observations")
            )

            table = pd.DataFrame(rows[-10:]).drop(
                columns=["timestamp_unix"],
                errors="ignore",
            )

            columns = [
                "timestamp", "latitude", "longitude",
                "baro_altitude", "velocity", "heading",
                "vertical_rate", "squawk",
            ]

            st.dataframe(
                table[
                    [column for column in columns if column in table.columns]
                ].iloc[::-1],
                use_container_width=True,
                hide_index=True,
            )

            if (
                time.time()
                - st.session_state.get("last_snapshot_time", 0)
                >= 60
            ):
                snapshot = (
                    DATA_DIR / "recordings"
                    / st.session_state.selected_id
                    / "rolling_45_minutes.csv"
                )

                snapshot.parent.mkdir(parents=True, exist_ok=True)

                pd.DataFrame(rows).drop(
                    columns=["timestamp_unix"],
                    errors="ignore",
                ).to_csv(snapshot, index=False)

                st.session_state.last_snapshot_time = time.time()

        st.markdown("### 🗺️ Flight path")

        deck = make_route_map(
            st.session_state.route,
            st.session_state.track,
        )

        if deck:
            st.pydeck_chart(deck, use_container_width=True)
        else:
            st.info("Waiting for position data.")

        if st.session_state.route:
            origin = st.session_state.route["origin"]
            destination = st.session_state.route["destination"]

            st.caption(
                f"Route lookup: {origin['name']} → {destination['name']} "
                "(ADSBdb callsign result; not independently verified)"
            )
            st.caption(
                "🔴 Straight-line origin-to-destination reference, not an "
                "actual flight path · 🟢 GPS positions recorded since monitoring "
                "started · 🟠 Latest observed aircraft position"
            )
            st.warning(
                "The red line is only a straight-line reference. It does not "
                "represent the aircraft's actual route or prove a navigation deviation."
            )
        else:
            st.caption(
                "Airport route unavailable for this callsign. "
                "The green track records positions observed after "
                "monitoring begins."
            )

    live_panel()

    st.markdown("### 🚨 Emergency and ground station")

    if not st.session_state.emergency:
        if st.button("Trigger simulated test emergency", type="primary"):
            fire_emergency("MANUAL TEST EMERGENCY")
            st.rerun()
    else:
        st.error(f"Triggered: {st.session_state.emergency_issue}")
        st.write(st.session_state.ground_report)
        st.caption("Simulation only. No real ATC connection.")

        if st.button("Reset monitor"):
            current = {
                "icao24": st.session_state.selected_id,
                "callsign": st.session_state.selected_callsign,
            }
            select_flight(current)
            st.rerun()


main()
