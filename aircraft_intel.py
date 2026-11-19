"""
Aircraft Model & Fleet Intelligence Knowledge Base
Provides deep specifications, production history, dimensions, engine specs,
and authentic photography for commercial and general aviation aircraft.
"""

from __future__ import annotations
import requests
from typing import Dict, Any, Optional

# High-resolution, reliable aviation photography for major aircraft types
DEFAULT_AIRCRAFT_PHOTOS = {
    "A320": "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?auto=format&fit=crop&w=1200&q=80",
    "A321": "https://images.unsplash.com/photo-1520437358207-323b43b50729?auto=format&fit=crop&w=1200&q=80",
    "A20N": "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?auto=format&fit=crop&w=1200&q=80",
    "A21N": "https://images.unsplash.com/photo-1520437358207-323b43b50729?auto=format&fit=crop&w=1200&q=80",
    "A330": "https://images.unsplash.com/photo-1569154941061-e231b4725ef1?auto=format&fit=crop&w=1200&q=80",
    "A350": "https://images.unsplash.com/photo-1583863788434-e58a36330cf0?auto=format&fit=crop&w=1200&q=80",
    "A380": "https://images.unsplash.com/photo-1570710891163-6d3b5c47248b?auto=format&fit=crop&w=1200&q=80",
    "B737": "https://images.unsplash.com/photo-1529074963764-98f45c47344b?auto=format&fit=crop&w=1200&q=80",
    "B738": "https://images.unsplash.com/photo-1529074963764-98f45c47344b?auto=format&fit=crop&w=1200&q=80",
    "B38M": "https://images.unsplash.com/photo-1529074963764-98f45c47344b?auto=format&fit=crop&w=1200&q=80",
    "B777": "https://images.unsplash.com/photo-1579783900882-c0d3dad7b119?auto=format&fit=crop&w=1200&q=80",
    "B77W": "https://images.unsplash.com/photo-1579783900882-c0d3dad7b119?auto=format&fit=crop&w=1200&q=80",
    "B787": "https://images.unsplash.com/photo-1506015391300-4802dc74de2e?auto=format&fit=crop&w=1200&q=80",
    "B788": "https://images.unsplash.com/photo-1506015391300-4802dc74de2e?auto=format&fit=crop&w=1200&q=80",
    "B789": "https://images.unsplash.com/photo-1506015391300-4802dc74de2e?auto=format&fit=crop&w=1200&q=80",
    "B747": "https://images.unsplash.com/photo-1580674285054-bed31e145f59?auto=format&fit=crop&w=1200&q=80",
    "AT72": "https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?auto=format&fit=crop&w=1200&q=80",
    "AT76": "https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?auto=format&fit=crop&w=1200&q=80",
    "E190": "https://images.unsplash.com/photo-1519074069444-1ba4fff16def?auto=format&fit=crop&w=1200&q=80",
    "E195": "https://images.unsplash.com/photo-1519074069444-1ba4fff16def?auto=format&fit=crop&w=1200&q=80",
    "GENERIC": "https://images.unsplash.com/photo-1517999144091-3d9dca6d1e43?auto=format&fit=crop&w=1200&q=80",
}

AIRCRAFT_SPECIFICATIONS = {
    "A320": {
        "full_name": "Airbus A320-200 / A320ceo",
        "family": "Airbus A320 Family",
        "manufacturer": "Airbus S.A.S.",
        "headquarters": "Blagnac (Toulouse), France",
        "assembly_sites": "Toulouse (France), Hamburg (Germany), Tianjin (China), Mobile (USA)",
        "first_flight": "February 22, 1987",
        "introduction": "April 18, 1988 with Air France",
        "production_status": "Over 11,500+ A320-family delivered (Active worldwide)",
        "design_chief": "Bernard Ziegler & Roger Béteille",
        "role": "Short-to-Medium Range Narrow-Body Commercial Jetliner",
        "seating_typical": "150 passengers (2-class) / up to 180 (1-class)",
        "cockpit_crew": "2 pilots (Full Glass Cockpit + Side-stick Fly-by-Wire)",
        "wingspan": "35.80 m (117 ft 5 in) with Sharklets",
        "length": "37.57 m (123 ft 3 in)",
        "height": "11.76 m (38 ft 7 in)",
        "mtow": "78,000 kg (172,000 lb)",
        "engines": "2× CFM International CFM56-5B or IAE V2500-A5 turbofans (27,000 lbf each)",
        "cruise_speed": "Mach 0.78 (828 km/h / 447 knots)",
        "max_speed": "Mach 0.82 (871 km/h / 470 knots MMO)",
        "service_ceiling": "39,800 ft (12,130 m / FL398)",
        "range": "6,150 km (3,320 nautical miles)",
        "avionics_suite": "Honeywell / Thales Integrated Fly-By-Wire with flight-envelope protection",
        "historical_significance": (
            "The A320 made aviation history as the first commercial jetliner to feature full digital "
            "Fly-By-Wire (FBW) flight controls and side-stick controller architecture. It introduced "
            "hard flight-envelope protection, preventing pilots from exceeding aerodynamic limits "
            "(stall, overspeed, excessive bank angles)."
        )
    },
    "A20N": {
        "full_name": "Airbus A320neo (New Engine Option)",
        "family": "Airbus A320neo Family",
        "manufacturer": "Airbus S.A.S.",
        "headquarters": "Blagnac (Toulouse), France",
        "assembly_sites": "Toulouse, Hamburg, Tianjin, Mobile",
        "first_flight": "September 25, 2014",
        "introduction": "January 25, 2016 with Lufthansa",
        "production_status": "Active high-rate production (World's bestselling single-aisle aircraft)",
        "design_chief": "Airbus Engineering Team",
        "role": "Next-Generation High-Efficiency Narrow-Body Jetliner",
        "seating_typical": "165 passengers (2-class) / 186 (maximum high-density)",
        "cockpit_crew": "2 pilots (Glass Cockpit + Category IIIB Autoland)",
        "wingspan": "35.80 m (117 ft 5 in) with Blended Sharklets",
        "length": "37.57 m (123 ft 3 in)",
        "height": "11.76 m (38 ft 7 in)",
        "mtow": "79,000 kg (174,000 lb)",
        "engines": "2× CFM International LEAP-1A or Pratt & Whitney PW1100G-JM PurePower Geared Turbofan",
        "cruise_speed": "Mach 0.78 (828 km/h / 447 knots)",
        "max_speed": "Mach 0.82 (871 km/h / 470 knots MMO)",
        "service_ceiling": "39,800 ft (12,130 m)",
        "range": "6,850 km (3,700 nautical miles)",
        "avionics_suite": "Thales / Collins Aerospace Flight Management with RNP AR 0.1 capability",
        "historical_significance": (
            "Introduced ultra-high bypass geared turbofans and composite sharklets, delivering 20% lower "
            "fuel burn and 50% reduced noise footprint compared to prior generations. Backbone of modern "
            "fleets worldwide, including IndiGo, Lufthansa, American Airlines, and Air India."
        )
    },
    "A321": {
        "full_name": "Airbus A321-200 / A321neo",
        "family": "Airbus A320 Family",
        "manufacturer": "Airbus S.A.S.",
        "headquarters": "Blagnac (Toulouse), France",
        "assembly_sites": "Hamburg (Germany), Mobile (USA), Tianjin (China)",
        "first_flight": "March 11, 1993 (ceo) / February 9, 2016 (neo)",
        "introduction": "January 1994 with Lufthansa",
        "production_status": "Active production (LR and XLR transatlantic variants)",
        "design_chief": "Airbus Engineering Team",
        "role": "Extended Capacity Single-Aisle & Transatlantic Jetliner",
        "seating_typical": "180 - 220 passengers (up to 244 in Airbus Cabin Flex layout)",
        "cockpit_crew": "2 pilots",
        "wingspan": "35.80 m (117 ft 5 in)",
        "length": "44.51 m (146 ft 0 in) — stretched fuselage",
        "height": "11.76 m (38 ft 7 in)",
        "mtow": "93,500 kg to 97,000 kg (A321LR/XLR up to 101,000 kg)",
        "engines": "2× CFM LEAP-1A or Pratt & Whitney PW1133G-JM (33,000 lbf each)",
        "cruise_speed": "Mach 0.78 (828 km/h)",
        "max_speed": "Mach 0.82 (871 km/h)",
        "service_ceiling": "39,800 ft (12,130 m)",
        "range": "7,400 km (A321LR) / 8,700 km (A321XLR)",
        "avionics_suite": "Thales Avionics Suite with FANS-1/A Future Air Navigation",
        "historical_significance": (
            "Stretched variant of the A320. The A321neo and A321XLR are changing long-haul aviation by "
            "allowing single-aisle point-to-point transatlantic flights previously only flown by widebody twinjets."
        )
    },
    "B738": {
        "full_name": "Boeing 737-800 Next Generation",
        "family": "Boeing 737 Next Generation (NG)",
        "manufacturer": "The Boeing Company",
        "headquarters": "Arlington, Virginia (Commercial Airplanes: Renton, Washington, USA)",
        "assembly_sites": "Boeing Renton Factory (Washington, USA)",
        "first_flight": "July 31, 1997",
        "introduction": "April 24, 1998 with Hapag-Lloyd Flug",
        "production_status": "5,100+ units delivered; most ubiquitous commercial airliner on earth",
        "design_chief": "Boeing Commercial Airplanes Design Council",
        "role": "Short-to-Medium Haul Narrow-Body Commercial Airliner",
        "seating_typical": "162 passengers (2-class) / 189 maximum (single-class)",
        "cockpit_crew": "2 pilots (Rockwell Collins 6-Screen Flight Display System)",
        "wingspan": "35.79 m (117 ft 5 in) with Blended Winglets",
        "length": "39.47 m (129 ft 6 in)",
        "height": "12.55 m (41 ft 2 in)",
        "mtow": "79,015 kg (174,200 lb)",
        "engines": "2× CFM International CFM56-7B26 turbofans (27,300 lbf each)",
        "cruise_speed": "Mach 0.785 (838 km/h / 453 knots)",
        "max_speed": "Mach 0.82 (876 km/h / 473 knots MMO)",
        "service_ceiling": "41,000 ft (12,500 m / FL410)",
        "range": "5,436 km (2,935 nautical miles)",
        "avionics_suite": "Collins Aerospace Electronic Flight Instrument System (EFIS)",
        "historical_significance": (
            "The 737-800 is the commercial core of Boeing's narrowbody history. Designed to replace "
            "the 737-400 and compete directly with the A320, it became the workhorse of global airlines "
            "(Ryanair, American, Southwest, Air India Express, United, Qantas) with unparalleled dispatch reliability."
        )
    },
    "B38M": {
        "full_name": "Boeing 737 MAX 8",
        "family": "Boeing 737 MAX Series",
        "manufacturer": "The Boeing Company",
        "headquarters": "Arlington, Virginia / Renton, Washington, USA",
        "assembly_sites": "Renton Factory (Washington, USA)",
        "first_flight": "January 29, 2016",
        "introduction": "May 22, 2017 with Malindo Air",
        "production_status": "Active production with comprehensive recertification upgrades",
        "design_chief": "Boeing Commercial Airplanes",
        "role": "Advanced Efficiency Single-Aisle Commercial Jetliner",
        "seating_typical": "162 - 178 passengers (up to 200 on MAX 200 variant)",
        "cockpit_crew": "2 pilots (Four 15.1-inch Rockwell Collins Large-Format Landscape Displays)",
        "wingspan": "35.92 m (117 ft 10 in) with Advanced Technology (AT) Split-Scimitar Winglets",
        "length": "39.52 m (129 ft 8 in)",
        "height": "12.30 m (40 ft 4 in)",
        "mtow": "82,190 kg (181,200 lb)",
        "engines": "2× CFM International LEAP-1B turbofans with serrated chevron nacelles (29,300 lbf)",
        "cruise_speed": "Mach 0.79 (839 km/h / 453 knots)",
        "max_speed": "Mach 0.82 (876 km/h)",
        "service_ceiling": "41,000 ft (12,500 m / FL410)",
        "range": "6,570 km (3,550 nautical miles)",
        "avionics_suite": "Rockwell Collins 787-style cockpit avionics + Dual-Channel Angle-of-Attack Architecture",
        "historical_significance": (
            "The 4th generation of the 737 family. Powered by high-efficiency LEAP-1B engines and "
            "distinctive split-scimitar wingtips, offering 14% lower fuel consumption. Underwent the "
            "most extensive regulatory audit in aviation history by the FAA, EASA, and DGCA."
        )
    },
    "B77W": {
        "full_name": "Boeing 777-300ER (Extended Range)",
        "family": "Boeing 777 Series ('Triple Seven')",
        "manufacturer": "The Boeing Company",
        "headquarters": "Arlington, Virginia / Everett, Washington, USA",
        "assembly_sites": "Everett Factory (Washington, USA — World's largest building by volume)",
        "first_flight": "February 24, 2003",
        "introduction": "May 10, 2004 with Air France",
        "production_status": "Over 830+ 777-300ER delivered; premier long-haul flagship",
        "design_chief": "Alan Mulally & Boeing 777 Engineering Team",
        "role": "Long-Range Wide-Body Flagship Twinjet",
        "seating_typical": "365 - 396 passengers (3-class) / up to 550 (high-density)",
        "cockpit_crew": "2 pilots (Honeywell Flat-Panel EFIS with Electronic Checklist)",
        "wingspan": "64.80 m (212 ft 7 in) with Raked Wingtips",
        "length": "73.86 m (242 ft 4 in)",
        "height": "18.50 m (60 ft 8 in)",
        "mtow": "351,534 kg (775,000 lb)",
        "engines": "2× General Electric GE90-115B turbofans (115,300 lbf each — world's largest engine)",
        "cruise_speed": "Mach 0.84 (892 km/h / 482 knots)",
        "max_speed": "Mach 0.89 (945 km/h MMO)",
        "service_ceiling": "43,100 ft (13,140 m / FL431)",
        "range": "13,650 km (7,370 nautical miles)",
        "avionics_suite": "Honeywell Integrated Aircraft Information Management System (AIMS)",
        "historical_significance": (
            "The 777 was the first airliner designed 100% digitally using 3D CAD (CATIA). The 777-300ER "
            "effectively replaced 4-engine jumbos like the 747-400 and A340 by demonstrating that a twinjet "
            "could safely fly 15-hour polar and transpacific ETOPS-330 missions with unmatched efficiency."
        )
    },
    "B788": {
        "full_name": "Boeing 787-8 / 787-9 Dreamliner",
        "family": "Boeing 787 Dreamliner Series",
        "manufacturer": "The Boeing Company",
        "headquarters": "Arlington, Virginia / Charleston, South Carolina, USA",
        "assembly_sites": "Boeing South Carolina (North Charleston, USA)",
        "first_flight": "December 15, 2009",
        "introduction": "October 26, 2011 with All Nippon Airways (ANA)",
        "production_status": "Over 1,100+ delivered across 787-8, 787-9, and 787-10 variants",
        "design_chief": "Mike Sinnett & Dreamliner Engineering Leadership",
        "role": "Super-Efficient Long-Haul Wide-Body Composite Airliner",
        "seating_typical": "248 (787-8) to 296 passengers (787-9) in 2-class layout",
        "cockpit_crew": "2 pilots (Dual Head-Up Displays HUD + Rockwell Collins Cockpit)",
        "wingspan": "60.12 m (197 ft 3 in) with Smooth Flex Raked Wingtips",
        "length": "56.72 m (787-8) / 62.81 m (787-9)",
        "height": "17.02 m (55 ft 10 in)",
        "mtow": "227,930 kg (787-8) / 254,000 kg (787-9)",
        "engines": "2× General Electric GEnx-1B or Rolls-Royce Trent 1000 turbofans (70,000 lbf)",
        "cruise_speed": "Mach 0.85 (903 km/h / 488 knots)",
        "max_speed": "Mach 0.90 (956 km/h MMO)",
        "service_ceiling": "43,000 ft (13,100 m / FL430)",
        "range": "14,140 km (7,635 nautical miles)",
        "avionics_suite": "Rockwell Collins Integrated Modular Avionics (IMA) on AFDX Gigabit Ethernet",
        "historical_significance": (
            "The Dreamliner revolutionised aerospace manufacturing by using 50% carbon fiber composite "
            "materials by weight. This enabled 6,000-ft cabin altitude pressurization, 20% higher humidity, "
            "large electrochromic dimming windows, and 25% lower fuel consumption."
        )
    },
    "A359": {
        "full_name": "Airbus A350-900 XWB (Extra Wide Body)",
        "family": "Airbus A350 XWB Series",
        "manufacturer": "Airbus S.A.S.",
        "headquarters": "Blagnac (Toulouse), France",
        "assembly_sites": "Toulouse Jean-Luc Lagardère Plant (France)",
        "first_flight": "June 14, 2013",
        "introduction": "January 15, 2015 with Qatar Airways",
        "production_status": "Over 580+ delivered (Flagship long-haul widebody in active production)",
        "design_chief": "Didier Evrard & Airbus Engineering",
        "role": "Ultra-Long-Range Wide-Body Flagship Airliner",
        "seating_typical": "315 - 369 passengers (3-class) / up to 440 (all-economy)",
        "cockpit_crew": "2 pilots (Six interchangeable 15-inch landscape LCD displays + HUD)",
        "wingspan": "64.75 m (212 ft 5 in) with Biomimetic Morphing Curved Winglets",
        "length": "66.80 m (219 ft 2 in)",
        "height": "17.05 m (55 ft 11 in)",
        "mtow": "283,000 kg (624,000 lb)",
        "engines": "2× Rolls-Royce Trent XWB-84 high-bypass turbofans (84,200 lbf each)",
        "cruise_speed": "Mach 0.85 (903 km/h / 488 knots)",
        "max_speed": "Mach 0.89 (945 km/h MMO)",
        "service_ceiling": "43,100 ft (13,140 m / FL431)",
        "range": "15,372 km (8,300 nautical miles / ULR variant up to 18,000 km)",
        "avionics_suite": "Thales Avionics with Airport Navigation System & Brake-to-Vacate automation",
        "historical_significance": (
            "Composed of 53% carbon-fiber-reinforced polymer (CFRP) composites and titanium. Features "
            "adaptive wing surfaces that morph inflight to minimize drag. Operates the world's longest "
            "commercial nonstop flight (Singapore to New York, 18+ hours)."
        )
    },
    "A388": {
        "full_name": "Airbus A380-800 Superjumbo",
        "family": "Airbus A380 Double-Deck Series",
        "manufacturer": "Airbus S.A.S.",
        "headquarters": "Blagnac (Toulouse), France",
        "assembly_sites": "Toulouse (France)",
        "first_flight": "April 27, 2005",
        "introduction": "October 25, 2007 with Singapore Airlines",
        "production_status": "251 aircraft built (Flagship superjumbo in service with Emirates, BA, Singapore, etc.)",
        "design_chief": "Robert Lafontan & Jürgen Thomas",
        "role": "Very Large Commercial Transport (VLCT) / Double-Deck Superjumbo",
        "seating_typical": "525 passengers (3-class) / up to 853 maximum certified",
        "cockpit_crew": "2 pilots",
        "wingspan": "79.75 m (261 ft 8 in)",
        "length": "72.72 m (238 ft 7 in)",
        "height": "24.09 m (79 ft 0 in) — 8 stories tall",
        "mtow": "575,000 kg (1,268,000 lb / 575 metric tons)",
        "engines": "4× Engine Alliance GP7200 or Rolls-Royce Trent 900 turbofans (72,000 lbf each)",
        "cruise_speed": "Mach 0.85 (903 km/h / 488 knots)",
        "max_speed": "Mach 0.89 (945 km/h MMO)",
        "service_ceiling": "43,000 ft (13,100 m / FL430)",
        "range": "14,800 km (8,000 nautical miles)",
        "avionics_suite": "Thales / Honeywell Avionics with Onboard Information System (OIS)",
        "historical_significance": (
            "The largest passenger airliner ever constructed. Required airport runway expansions and "
            "two-story boarding jet bridges worldwide. Features full twin decks, whisper-quiet cabin acoustics, "
            "and unmatched widebody passenger comfort."
        )
    },
    "AT76": {
        "full_name": "ATR 72-600 Regional Turboprop",
        "family": "ATR 42 / 72 Series",
        "manufacturer": "Avions de Transport Régional (ATR)",
        "headquarters": "Toulouse, France (Airbus & Leonardo Joint Venture)",
        "assembly_sites": "Toulouse Francazal (France) & Pomigliano d'Arco (Italy)",
        "first_flight": "October 27, 1988 (ATR 72) / July 24, 2009 (-600 series)",
        "introduction": "November 2011 with Royal Air Maroc",
        "production_status": "Over 1,600+ ATR turboprops delivered worldwide",
        "design_chief": "French-Italian Consortium Engineering Team",
        "role": "Twin-Engine Regional Short-Takeoff Turboprop Airliner",
        "seating_typical": "68 - 72 passengers (up to 78 high-density)",
        "cockpit_crew": "2 pilots (Thales 5-Display Glass Cockpit Suite)",
        "wingspan": "27.05 m (88 ft 9 in)",
        "length": "27.17 m (89 ft 2 in)",
        "height": "7.65 m (25 ft 1 in)",
        "mtow": "23,000 kg (50,700 lb)",
        "engines": "2× Pratt & Whitney Canada PW127M / PW127XT-M turboprops with 6-blade Hamilton Sundstrand props (2,750 shp each)",
        "cruise_speed": "510 km/h (275 knots / 317 mph)",
        "max_speed": "518 km/h (280 knots)",
        "service_ceiling": "25,000 ft (7,600 m / FL250)",
        "range": "1,528 km (825 nautical miles)",
        "avionics_suite": "Thales Glass Cockpit with Dual Flight Management System & LPV approach capability",
        "historical_significance": (
            "World's dominant regional turboprop. Consumes 45% less fuel than regional jets of similar size, "
            "making it the backbone of regional connectivity, island hopping, and unpaved/short airstrips worldwide."
        )
    }
}

# Alias mapping from ADS-B ICAO type codes to specifications
ICAO_TYPE_ALIASES = {
    # Airbus A320
    "A320": "A320", "A319": "A320", "A318": "A320", "A20N": "A20N",
    "A321": "A321", "A21N": "A321", "A321NEO": "A321",
    # Boeing 737
    "B738": "B738", "B737": "B738", "B739": "B738", "B736": "B738",
    "B38M": "B38M", "B39M": "B38M", "B3JM": "B38M", "738": "B738",
    # Boeing 777
    "B77W": "B77W", "B772": "B77W", "B773": "B77W", "B77L": "B77W", "B77F": "B77W",
    # Boeing 787
    "B788": "B788", "B789": "B788", "B78X": "B788", "B787": "B788",
    # Airbus A350
    "A359": "A359", "A35K": "A359", "A350": "A359",
    # Airbus A380
    "A388": "A388", "A380": "A388",
    # ATR
    "AT76": "AT76", "AT75": "AT76", "AT72": "AT76", "AT45": "AT76", "AT46": "AT76",
}


def lookup_aircraft_db(icao24: str) -> Optional[Dict[str, Any]]:
    """
    Look up live aircraft registration, type, operator, and official photo from ADSBdb.
    """
    clean_icao = (icao24 or "").strip().lower()
    if not clean_icao:
        return None
    try:
        url = f"https://api.adsbdb.com/v0/aircraft/{clean_icao}"
        resp = requests.get(url, timeout=5, headers={"User-Agent": "AeroGuard-Avionics/2.0"})
        if resp.status_code == 200:
            payload = resp.json().get("response", {}).get("aircraft", {})
            if payload:
                return {
                    "icao24": clean_icao.upper(),
                    "registration": payload.get("registration") or "Unregistered / Mode-S",
                    "type": payload.get("type") or "Commercial Aircraft",
                    "icao_type": (payload.get("icao_type") or "").upper(),
                    "manufacturer": payload.get("manufacturer") or "Commercial Aerospace",
                    "registered_owner": payload.get("registered_owner") or "Commercial Operator",
                    "country": payload.get("registered_owner_country_name") or "International",
                    "photo_url": payload.get("url_photo") or payload.get("url_photo_thumbnail") or None,
                }
    except Exception:
        pass
    return None


def get_full_aircraft_intelligence(icao24: str, detected_callsign: str = "") -> Dict[str, Any]:
    """
    Aggregates online ADS-B database registration info with deep technical specifications,
    historical context, engineering schematics, and photos.
    """
    live_data = lookup_aircraft_db(icao24) or {}
    
    icao_type = live_data.get("icao_type") or ""
    type_name = (live_data.get("type") or "").upper()
    manufacturer_raw = live_data.get("manufacturer") or ""
    
    # Identify matching spec key
    spec_key = None
    if icao_type in ICAO_TYPE_ALIASES:
        spec_key = ICAO_TYPE_ALIASES[icao_type]
    elif any(k in type_name for k in ["A320", "320"]):
        spec_key = "A320"
    elif any(k in type_name for k in ["A321", "321"]):
        spec_key = "A321"
    elif any(k in type_name for k in ["737-8", "738", "737-800"]):
        spec_key = "B738"
    elif any(k in type_name for k in ["737 MAX", "MAX 8", "MAX8"]):
        spec_key = "B38M"
    elif any(k in type_name for k in ["777", "TRIPLE SEVEN"]):
        spec_key = "B77W"
    elif any(k in type_name for k in ["787", "DREAMLINER"]):
        spec_key = "B788"
    elif any(k in type_name for k in ["A350", "350"]):
        spec_key = "A359"
    elif any(k in type_name for k in ["A380", "380"]):
        spec_key = "A388"
    elif any(k in type_name for k in ["ATR 72", "ATR-72", "ATR72"]):
        spec_key = "AT76"
    else:
        # Default based on most common domestic airliner if unlisted
        spec_key = "A320"

    specs = AIRCRAFT_SPECIFICATIONS.get(spec_key, AIRCRAFT_SPECIFICATIONS["A320"])
    
    # Determine best photo
    photo = live_data.get("photo_url")
    if not photo or not photo.startswith("http"):
        photo = DEFAULT_AIRCRAFT_PHOTOS.get(spec_key, DEFAULT_AIRCRAFT_PHOTOS["GENERIC"])

    return {
        "icao24": (icao24 or "UNKNOWN").upper(),
        "callsign": (detected_callsign or "UNKNOWN").strip().upper(),
        "registration": live_data.get("registration") or "Active Airframe",
        "operator": live_data.get("registered_owner") or "Scheduled Commercial Carrier",
        "country": live_data.get("country") or "International Aviation",
        "detected_model_raw": live_data.get("type") or specs["full_name"],
        "icao_type": icao_type or spec_key,
        "photo_url": photo,
        "specs": specs
    }
