"""
Professional Telematics Data Generator for Zimbabwean Fleet Telemetry.
Generates realistic synthetic vehicle telemetry data with a 5-second refresh cadence.
"""

import numpy as np
import pandas as pd
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from string import ascii_uppercase
from typing import Optional, Tuple, Any
from zoneinfo import ZoneInfo
import os

try:
    from faker import Faker
except ImportError:
    Faker = None

_POLARS_MODULE = None
_POLARS_IMPORT_ATTEMPTED = False


def _get_polars_module():
    global _POLARS_MODULE, _POLARS_IMPORT_ATTEMPTED
    if not _POLARS_IMPORT_ATTEMPTED:
        try:
            import polars as polars_module

            _POLARS_MODULE = polars_module
        except ImportError:
            _POLARS_MODULE = None
        _POLARS_IMPORT_ATTEMPTED = True
    return _POLARS_MODULE

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

UPDATE_INTERVAL_SECONDS = 5.0
NUM_VEHICLES = 2000

# Road and city data
Road_Types = ["Highway", "Gravel", "Potholed", "Local Streets", "Urban", "Rural Tarred"]
ROAD_LIMITS = {
    "Highway": (80, 160),
    "Gravel": (20, 70),
    "Potholed": (10, 45),
    "Local Streets": (15, 60),
    "Urban": (25, 85),
    "Rural Tarred": (30, 110),
}
DIRECTIONS = ["North", "South", "East", "West", "Northeast", "Northwest", "Southeast", "Southwest"]
PUBLIC_TRANSPORT_TYPES = {"Kombi", "Minibus", "Bus"}
COMMERCIAL_SERVICE_TYPES = {"Pickup", "Van", "Light Truck", "Heavy Truck"}
PUBLIC_VEHICLE_TYPES = PUBLIC_TRANSPORT_TYPES | COMMERCIAL_SERVICE_TYPES
LOCAL_TIMEZONE = ZoneInfo("Africa/Harare")
CITY_PEAK_ACTIVITY_BOOST = {
    "Harare": 0.08,
    "Bulawayo": 0.05,
    "Mutare": 0.02,
    "Gweru": 0.02,
}
CITY_LATE_NIGHT_ACTIVITY_BOOST = {
    "Harare": 0.20,
    "Bulawayo": 0.08,
    "Mutare": 0.04,
    "Gweru": 0.03,
    "Victoria Falls": 0.03,
}

if Faker is not None:
    _faker = Faker("en_US")
    _faker.seed_instance(42)
else:
    _faker = None

CITY_COORDS = {
    "Harare": (-17.8252, 31.0335),
    "Bulawayo": (-20.1320, 28.6265),
    "Gweru": (-19.455, 29.820),
    "Mutare": (-18.9707, 32.6709),
    "Masvingo": (-20.070, 30.829),
    "Kwekwe": (-18.921, 29.816),
    "Kadoma": (-18.340, 29.900),
    "Chinhoyi": (-17.350, 30.200),
    "Victoria Falls": (-17.925, 25.856),
    "Beitbridge": (-22.217, 30.000),
    "Chegutu": (-18.130, 30.150),
    "Redcliff": (-19.033, 29.783),
    "Zvishavane": (-20.330, 30.046),
    "Shurugwi": (-19.667, 30.000),
    "Plumtree": (-20.478, 27.797),
    "Macheke": (-18.139, 31.849),
    "Mvuma": (-19.283, 30.533),
    "Gwanda": (-20.939, 29.005),
}
CITIES = list(CITY_COORDS.keys())
CITY_SELECTION_WEIGHTS = np.array(
    [0.19, 0.13, 0.07, 0.07, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02],
    dtype=float,
)
CITY_SELECTION_WEIGHTS /= CITY_SELECTION_WEIGHTS.sum()


def _scaled_city_selection_weights(multiplier_map: dict[str, float], default_multiplier: float = 1.0) -> np.ndarray:
    weights = CITY_SELECTION_WEIGHTS.copy()
    multipliers = np.array([multiplier_map.get(city, default_multiplier) for city in CITIES], dtype=float)
    weights = weights * multipliers
    weights /= weights.sum()
    return weights


PUBLIC_TRANSPORT_CITY_SELECTION_WEIGHTS = _scaled_city_selection_weights(
    {
        "Harare": 2.70,
        "Bulawayo": 2.00,
        "Mutare": 1.15,
        "Gweru": 1.10,
        "Masvingo": 0.92,
        "Kwekwe": 0.90,
        "Kadoma": 0.88,
        "Chinhoyi": 0.88,
    },
    default_multiplier=0.72,
)
COMMERCIAL_SERVICE_CITY_SELECTION_WEIGHTS = _scaled_city_selection_weights(
    {
        "Harare": 1.85,
        "Bulawayo": 1.45,
        "Mutare": 1.12,
        "Gweru": 1.10,
        "Kwekwe": 1.05,
        "Kadoma": 1.04,
    },
    default_multiplier=0.92,
)
PRIVATE_CITY_SELECTION_WEIGHTS = _scaled_city_selection_weights(
    {
        "Harare": 0.95,
        "Bulawayo": 0.96,
    },
    default_multiplier=1.0,
)

CITY_WEATHER = {
    "highveld": {
        "day": {
            "wet":   {"Sunny": 0.45, "Hot / Scorching": 0.05, "Partly Cloudy": 0.25, "Thundery / Heavy Rain": 0.15, "Hazy": 0.10},
            "dry":   {"Sunny": 0.65, "Hot / Scorching": 0.10, "Partly Cloudy": 0.15, "Thundery / Heavy Rain": 0.00, "Hazy": 0.10}
        },
        "night": {
            "wet":   {"Clear": 0.55, "Cold / Freezing": 0.00, "Mild / Balmy": 0.30, "Windy / Gusty": 0.05, "Misty / Foggy": 0.10},
            "dry":   {"Clear": 0.70, "Cold / Freezing": 0.10, "Mild / Balmy": 0.05, "Windy / Gusty": 0.05, "Misty / Foggy": 0.10}
        }
    },
    "eastern_highlands": {
        "day": {
            "wet":   {"Sunny": 0.30, "Hot": 0.02, "Partly Cloudy": 0.25, "Heavy Rain": 0.28, "Hazy": 0.15},
            "dry":   {"Sunny": 0.50, "Hot": 0.05, "Partly Cloudy": 0.20, "Heavy Rain": 0.00, "Hazy": 0.25}
        },
        "night": {
            "wet":   {"Clear": 0.40, "Cold": 0.10, "Mild": 0.20, "Windy": 0.10, "Misty": 0.20},
            "dry":   {"Clear": 0.55, "Cold": 0.15, "Mild": 0.05, "Windy": 0.10, "Misty": 0.15}
        }
    },
    "lowveld": {
        "day": {
            "wet":   {"Sunny": 0.55, "Hot": 0.15, "Partly Cloudy": 0.15, "Heavy Rain": 0.10, "Hazy": 0.05},
            "dry":   {"Sunny": 0.75, "Hot": 0.15, "Partly Cloudy": 0.05, "Heavy Rain": 0.00, "Hazy": 0.05}
        },
        "night": {
            "wet":   {"Clear": 0.60, "Cold": 0.00, "Mild": 0.30, "Windy": 0.05, "Misty": 0.05},
            "dry":   {"Clear": 0.80, "Cold": 0.00, "Mild": 0.15, "Windy": 0.03, "Misty": 0.02}
        }
    },
    "matabeleland": {
        "day": {
            "wet":   {"Sunny": 0.60, "Hot": 0.08, "Partly Cloudy": 0.18, "Heavy Rain": 0.08, "Hazy": 0.06},
            "dry":   {"Sunny": 0.70, "Hot": 0.15, "Partly Cloudy": 0.10, "Heavy Rain": 0.00, "Hazy": 0.05}
        },
        "night": {
            "wet":   {"Clear": 0.65, "Cold": 0.02, "Mild": 0.20, "Windy": 0.05, "Misty": 0.08},
            "dry":   {"Clear": 0.75, "Cold": 0.10, "Mild": 0.05, "Windy": 0.05, "Misty": 0.05}
        }
    },
    "transition": {
        "day": {
            "wet":   {"Sunny": 0.50, "Hot": 0.08, "Partly Cloudy": 0.22, "Heavy Rain": 0.12, "Hazy": 0.08},
            "dry":   {"Sunny": 0.68, "Hot": 0.12, "Partly Cloudy": 0.12, "Heavy Rain": 0.00, "Hazy": 0.08}
        },
        "night": {
            "wet":   {"Clear": 0.60, "Cold": 0.02, "Mild": 0.25, "Windy": 0.05, "Misty": 0.08},
            "dry":   {"Clear": 0.72, "Cold": 0.12, "Mild": 0.06, "Windy": 0.05, "Misty": 0.05}
        }
    }
}

CITY_WEATHER_ZONES = {
    "Harare": "highveld", "Bulawayo": "matabeleland", "Mutare": "eastern_highlands", "Gweru": "highveld",
    "Masvingo": "lowveld", "Kwekwe": "transition", "Kadoma": "transition", "Chinhoyi": "transition",
    "Victoria Falls": "matabeleland", "Beitbridge": "lowveld", "Chegutu": "transition", "Redcliff": "transition",
    "Zvishavane": "lowveld", "Shurugwi": "transition", "Plumtree": "matabeleland", "Macheke": "highveld",
    "Mvuma": "transition", "Gwanda": "matabeleland",
}

CITY_ROAD_PROFILES = {
    "Harare": {"Highway": 0.18, "Urban": 0.30, "Local Streets": 0.22, "Rural Tarred": 0.12, "Gravel": 0.08, "Potholed": 0.10},
    "Bulawayo": {"Highway": 0.20, "Urban": 0.26, "Local Streets": 0.20, "Rural Tarred": 0.14, "Gravel": 0.10, "Potholed": 0.10},
    "Gweru": {"Highway": 0.16, "Urban": 0.24, "Local Streets": 0.24, "Rural Tarred": 0.14, "Gravel": 0.11, "Potholed": 0.11},
    "Mutare": {"Highway": 0.14, "Urban": 0.22, "Local Streets": 0.20, "Rural Tarred": 0.18, "Gravel": 0.14, "Potholed": 0.12},
    "Masvingo": {"Highway": 0.14, "Urban": 0.18, "Local Streets": 0.18, "Rural Tarred": 0.20, "Gravel": 0.16, "Potholed": 0.14},
    "Kwekwe": {"Highway": 0.15, "Urban": 0.20, "Local Streets": 0.22, "Rural Tarred": 0.18, "Gravel": 0.14, "Potholed": 0.11},
    "Kadoma": {"Highway": 0.16, "Urban": 0.22, "Local Streets": 0.22, "Rural Tarred": 0.16, "Gravel": 0.13, "Potholed": 0.11},
    "Chinhoyi": {"Highway": 0.14, "Urban": 0.20, "Local Streets": 0.22, "Rural Tarred": 0.18, "Gravel": 0.14, "Potholed": 0.12},
    "Victoria Falls": {"Highway": 0.24, "Urban": 0.18, "Local Streets": 0.18, "Rural Tarred": 0.16, "Gravel": 0.12, "Potholed": 0.12},
    "Beitbridge": {"Highway": 0.26, "Urban": 0.14, "Local Streets": 0.16, "Rural Tarred": 0.18, "Gravel": 0.12, "Potholed": 0.14},
    "Chegutu": {"Highway": 0.16, "Urban": 0.20, "Local Streets": 0.22, "Rural Tarred": 0.18, "Gravel": 0.14, "Potholed": 0.10},
    "Redcliff": {"Highway": 0.16, "Urban": 0.20, "Local Streets": 0.20, "Rural Tarred": 0.18, "Gravel": 0.15, "Potholed": 0.11},
    "Zvishavane": {"Highway": 0.18, "Urban": 0.18, "Local Streets": 0.18, "Rural Tarred": 0.20, "Gravel": 0.14, "Potholed": 0.12},
    "Shurugwi": {"Highway": 0.16, "Urban": 0.18, "Local Streets": 0.18, "Rural Tarred": 0.20, "Gravel": 0.16, "Potholed": 0.12},
    "Plumtree": {"Highway": 0.22, "Urban": 0.16, "Local Streets": 0.18, "Rural Tarred": 0.18, "Gravel": 0.14, "Potholed": 0.12},
    "Macheke": {"Highway": 0.14, "Urban": 0.24, "Local Streets": 0.24, "Rural Tarred": 0.16, "Gravel": 0.12, "Potholed": 0.10},
    "Mvuma": {"Highway": 0.18, "Urban": 0.18, "Local Streets": 0.20, "Rural Tarred": 0.18, "Gravel": 0.13, "Potholed": 0.13},
    "Gwanda": {"Highway": 0.20, "Urban": 0.14, "Local Streets": 0.16, "Rural Tarred": 0.20, "Gravel": 0.16, "Potholed": 0.14},
}
DEFAULT_CITY_ROAD_PROFILE = {"Highway": 0.16, "Urban": 0.22, "Local Streets": 0.22, "Rural Tarred": 0.17, "Gravel": 0.13, "Potholed": 0.10}

TIME_OF_DAY_ROAD_BIAS = {
    "Dawn": np.array([1.05, 1.00, 0.95, 1.05, 1.00, 0.95], dtype=float),
    "Morning Rush": np.array([1.20, 1.10, 1.00, 0.90, 0.95, 0.85], dtype=float),
    "Mid-Day": np.array([1.00, 1.10, 1.10, 1.10, 0.95, 0.95], dtype=float),
    "Evening Rush": np.array([1.20, 1.10, 1.00, 0.90, 0.95, 0.85], dtype=float),
    "Night": np.array([1.15, 1.05, 1.00, 0.95, 0.90, 0.90], dtype=float),
    "Late Night": np.array([1.25, 0.90, 0.90, 0.90, 0.85, 0.80], dtype=float),
}

VEHICLE_CATALOG = pd.DataFrame([
    # Sedans (private usage, common imports)
    {"type": "Sedan", "brand": "Toyota", "model": "Corolla/Axio", "cls": "Compact", "cc": 1500, "price": 6500, "usage": "Private", "weight": 0.12},
    {"type": "Sedan", "brand": "Toyota", "model": "Premio", "cls": "Mid-size", "cc": 1500, "price": 8500, "usage": "Private", "weight": 0.08},
    {"type": "Sedan", "brand": "Honda", "model": "Civic", "cls": "Compact", "cc": 1800, "price": 7500, "usage": "Private", "weight": 0.05},
    {"type": "Sedan", "brand": "Nissan", "model": "Sunny", "cls": "Compact", "cc": 1500, "price": 5800, "usage": "Private", "weight": 0.06},
    {"type": "Sedan", "brand": "Mercedes-Benz", "model": "C180", "cls": "Luxury", "cc": 1800, "price": 12000, "usage": "Private", "weight": 0.02},
    
    # Hatchbacks
    {"type": "Hatchback", "brand": "Honda", "model": "Fit", "cls": "Compact", "cc": 1300, "price": 5500, "usage": "Private", "weight": 0.10},
    {"type": "Hatchback", "brand": "Toyota", "model": "Vitz", "cls": "Compact", "cc": 1000, "price": 4800, "usage": "Private", "weight": 0.09},
    {"type": "Hatchback", "brand": "Nissan", "model": "Note", "cls": "Compact", "cc": 1200, "price": 5200, "usage": "Private", "weight": 0.06},
    
    # Station Wagons
    {"type": "Station Wagon", "brand": "Toyota", "model": "Fielder", "cls": "Compact", "cc": 1500, "price": 7200, "usage": "Private", "weight": 0.07},
    {"type": "Station Wagon", "brand": "Subaru", "model": "Legacy", "cls": "Mid-size", "cc": 2000, "price": 9000, "usage": "Private", "weight": 0.03},
    
    # SUVs
    {"type": "SUV", "brand": "Toyota", "model": "Rav4", "cls": "Compact", "cc": 2000, "price": 12000, "usage": "Private", "weight": 0.08},
    {"type": "SUV", "brand": "Toyota", "model": "Fortuner", "cls": "Mid-size", "cc": 2400, "price": 25000, "usage": "Private", "weight": 0.04},
    {"type": "SUV", "brand": "Toyota", "model": "Land Cruiser Prado", "cls": "Large", "cc": 3000, "price": 35000, "usage": "Private", "weight": 0.02},
    {"type": "SUV", "brand": "Nissan", "model": "X-Trail", "cls": "Compact", "cc": 2000, "price": 10500, "usage": "Private", "weight": 0.05},
    {"type": "SUV", "brand": "Ford", "model": "Everest", "cls": "Mid-size", "cc": 2500, "price": 28000, "usage": "Private", "weight": 0.03},
    
    # Pickups (very common for both private and commercial)
    {"type": "Pickup", "brand": "Toyota", "model": "Hilux", "cls": "Light Commercial", "cc": 2400, "price": 18000, "usage": "Commercial", "weight": 0.10},
    {"type": "Pickup", "brand": "Nissan", "model": "NP300", "cls": "Light Commercial", "cc": 2500, "price": 16000, "usage": "Commercial", "weight": 0.06},
    {"type": "Pickup", "brand": "Ford", "model": "Ranger", "cls": "Light Commercial", "cc": 2200, "price": 17000, "usage": "Commercial", "weight": 0.05},
    {"type": "Pickup", "brand": "Isuzu", "model": "D-Max", "cls": "Light Commercial", "cc": 2500, "price": 16500, "usage": "Commercial", "weight": 0.05},
    
    # Vans (commercial deliveries)
    {"type": "Van", "brand": "Toyota", "model": "Hiace Panel Van", "cls": "Commercial", "cc": 2800, "price": 20000, "usage": "Commercial", "weight": 0.05},
    {"type": "Van", "brand": "Nissan", "model": "NV350", "cls": "Commercial", "cc": 2500, "price": 19000, "usage": "Commercial", "weight": 0.04},
    
    # Kombi / Minibus (public transport)
    {"type": "Kombi", "brand": "Toyota", "model": "Hiace Commuter", "cls": "Public Service", "cc": 2700, "price": 15000, "usage": "Commercial", "weight": 0.07},
    {"type": "Minibus", "brand": "Nissan", "model": "Caravan", "cls": "Public Service", "cc": 3000, "price": 14000, "usage": "Commercial", "weight": 0.04},
    {"type": "Minibus", "brand": "Toyota", "model": "Coaster", "cls": "Public Service", "cc": 4200, "price": 35000, "usage": "Commercial", "weight": 0.02},
    
    # Buses
    {"type": "Bus", "brand": "Yutong", "model": "ZK6729", "cls": "Public Service", "cc": 5200, "price": 55000, "usage": "Commercial", "weight": 0.01},
    {"type": "Bus", "brand": "Marcopolo", "model": "Volare", "cls": "Public Service", "cc": 4000, "price": 48000, "usage": "Commercial", "weight": 0.01},
    
    # Light Trucks
    {"type": "Light Truck", "brand": "Isuzu", "model": "NQR", "cls": "Commercial", "cc": 4800, "price": 45000, "usage": "Commercial", "weight": 0.03},
    {"type": "Light Truck", "brand": "Toyota", "model": "Dyna", "cls": "Commercial", "cc": 4000, "price": 38000, "usage": "Commercial", "weight": 0.03},
    
    # Heavy Trucks
    {"type": "Heavy Truck", "brand": "Howo", "model": "A7", "cls": "Heavy Commercial", "cc": 9700, "price": 75000, "usage": "Commercial", "weight": 0.02},
    {"type": "Heavy Truck", "brand": "Sinotruk", "model": "Howo 371", "cls": "Heavy Commercial", "cc": 10000, "price": 82000, "usage": "Commercial", "weight": 0.02},
])

VEHICLE_YEAR_RANGES = {
    "Sedan": (2013, 2025),
    "Hatchback": (2014, 2025),
    "Station Wagon": (2011, 2024),
    "SUV": (2012, 2025),
    "Pickup": (2011, 2023),
    "Van": (2010, 2022),
    "Kombi": (2008, 2020),
    "Minibus": (2008, 2019),
    "Bus": (2007, 2018),
    "Light Truck": (2009, 2021),
    "Heavy Truck": (2006, 2018),
}

VEHICLE_FUEL_EFFICIENCY_BASE = {
    "Sedan": 7.2,
    "Hatchback": 6.6,
    "Station Wagon": 7.5,
    "SUV": 9.6,
    "Pickup": 10.8,
    "Van": 11.4,
    "Kombi": 12.2,
    "Minibus": 13.5,
    "Bus": 24.0,
    "Light Truck": 16.5,
    "Heavy Truck": 27.0,
}

STATIC_VEHICLE_COLUMNS = [
    "Plate", "Type", "Make", "Model", "Class", "Year", "Engine_CC", "Base_Price_USD",
    "City", "Usage",
    "Policy_Status", "Policy_Start_Date", "Effective_Date", "Policy_End_Date", "Expiry_Date",
    "Premium_Payment_Status", "Outstanding_Balance",
]

LIVE_DYNAMIC_COLUMNS = [
    "Trip_ID", "Road_Type", "Weather", "Status", "Direction",
    "Trip_Distance_km", "Trip_Duration_Hour", "Speed_kmh", "Acceleration_mps2", "RPM", "Throttle_pct",
    "Engine_Load_pct", "Coolant_Temp_C", "MAF_gs", "Battery_V", "GPS_Latitude", "GPS_Longitude",
    "Harsh_Brake_Count", "Harsh_Accel_Count", "Harsh_Corner_Count", "Total_Harsh_Events_per_Day", "Fuel_Efficiency_L_per_100km", "Last_Update",
    "Speeding_Flag", "Status_Change_Timestamp", "Time_of_Day", "Day_of_Week"
]

LIVE_RAW_COLUMNS = STATIC_VEHICLE_COLUMNS + LIVE_DYNAMIC_COLUMNS

LIVE_PREPROCESSED_ENGINEERED_COLUMNS = [
    "Max_Speed_kmh", "Speed_Variance", "Max_Acceleration_mps2", "Max_Deceleration_mps2",
    "Battery_Health_Score", "Recent_Harsh_Events", "Recent_Avg_Speed", "Recent_Speeding_Ratio", "Recent_Night_Distance",
    "Avg_Speed_Last_7_Days", "Total_Harsh_Events_Last_30_Days", "Speeding_Ratio_x_Night_Driving_Ratio",
    "Potholed_Ratio", "Gravel_Ratio", "Highway_Ratio", "Urban_Ratio", "Vehicle_Age_Years", "Trip_Duration_Minutes",
    "Weather_Risk_Score", "Driving_Event_Score", "Aggressive_Driving_Score",
    "Fatigue_Risk_Score", "Night_Driving_Flag", "Base_Road_Max_Speed_kmh", "Speeding_Excess_kmh", "Coolant_Overheat_Flag",
    "Update_Year", "Update_Month", "Update_Day", "Update_Hour", "Update_Minute",
]

LIVE_PREPROCESSED_ALL_COLUMNS = LIVE_RAW_COLUMNS + LIVE_PREPROCESSED_ENGINEERED_COLUMNS

DIRECTION_BEARINGS = {
    "North": 0.0,
    "Northeast": 45.0,
    "East": 90.0,
    "Southeast": 135.0,
    "South": 180.0,
    "Southwest": 225.0,
    "West": 270.0,
    "Northwest": 315.0,
}

PLATE_PREFIX_POOL = np.array(
    [f"A{first}{second}" for first in ascii_uppercase for second in ascii_uppercase],
    dtype=object,
)

ACTIVE_POLICY_STATUS_VALUES = np.array(["Active", "In Force", "Valid", "Renewed"], dtype=object)
ACTIVE_POLICY_STATUS_WEIGHTS = np.array([0.50, 0.22, 0.18, 0.10], dtype=float)
ACTIVE_PAYMENT_STATUS_VALUES = np.array(["Paid", "Up to Date", "Current"], dtype=object)
ACTIVE_PAYMENT_STATUS_WEIGHTS = np.array([0.58, 0.27, 0.15], dtype=float)

# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def normalize_weather_label(label: str) -> str:
    return str(label).strip()

def _current_local_time() -> datetime:
    return datetime.now(LOCAL_TIMEZONE).replace(tzinfo=None)


def _hour_fraction(hour: int, minute: int = 0, second: int = 0) -> float:
    return (float(hour) + (float(minute) / 60.0) + (float(second) / 3600.0)) % 24.0


def _datetime_hour_fraction(dt: datetime) -> float:
    return _hour_fraction(dt.hour, dt.minute, dt.second)


def _hourly_activity_intensity(hour_of_day: float) -> float:
    hour_points = np.array(
        [0.0, 2.0, 4.5, 5.0, 6.0, 7.5, 9.0, 10.0, 12.5, 15.0, 17.0, 18.0, 19.5, 20.5, 22.0, 23.5, 24.0],
        dtype=float,
    )
    intensity_points = np.array(
        [0.12, 0.08, 0.16, 0.30, 0.58, 1.00, 0.86, 0.62, 0.44, 0.38, 0.72, 0.95, 0.82, 0.52, 0.24, 0.16, 0.12],
        dtype=float,
    )
    return float(np.interp(hour_of_day % 24.0, hour_points, intensity_points))


def _generate_policy_fields(
    num_vehicles: int,
    reference_time: datetime,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Generate active policy metadata that stays valid for the current date."""
    num_vehicles = max(int(num_vehicles), 0)
    if num_vehicles == 0:
        empty = np.array([], dtype=object)
        return {
            "Policy_Status": empty,
            "Policy_Start_Date": empty,
            "Effective_Date": empty,
            "Policy_End_Date": empty,
            "Expiry_Date": empty,
            "Premium_Payment_Status": empty,
            "Outstanding_Balance": np.array([], dtype=float),
        }

    policy_status = rng.choice(
        ACTIVE_POLICY_STATUS_VALUES,
        size=num_vehicles,
        p=ACTIVE_POLICY_STATUS_WEIGHTS,
    ).astype(object)
    payment_status = rng.choice(
        ACTIVE_PAYMENT_STATUS_VALUES,
        size=num_vehicles,
        p=ACTIVE_PAYMENT_STATUS_WEIGHTS,
    ).astype(object)

    start_offsets = rng.integers(7, 320, size=num_vehicles)
    renewed_mask = policy_status == "Renewed"
    if renewed_mask.any():
        start_offsets[renewed_mask] = rng.integers(0, 45, size=int(renewed_mask.sum()))

    remaining_days = rng.integers(30, 360, size=num_vehicles)
    if renewed_mask.any():
        remaining_days[renewed_mask] = rng.integers(300, 390, size=int(renewed_mask.sum()))

    start_dates = np.array(
        [(reference_time - timedelta(days=int(days))).date().isoformat() for days in start_offsets],
        dtype=object,
    )
    end_dates = np.array(
        [(reference_time + timedelta(days=int(days))).date().isoformat() for days in remaining_days],
        dtype=object,
    )

    outstanding_balance = np.zeros(num_vehicles, dtype=float)
    up_to_date_mask = payment_status == "Up to Date"
    current_mask = payment_status == "Current"
    if up_to_date_mask.any():
        outstanding_balance[up_to_date_mask] = np.round(
            rng.uniform(0.0, 12.0, size=int(up_to_date_mask.sum())),
            2,
        )
    if current_mask.any():
        outstanding_balance[current_mask] = np.round(
            rng.uniform(0.0, 28.0, size=int(current_mask.sum())),
            2,
        )

    zero_mask = rng.random(num_vehicles) < 0.78
    outstanding_balance[zero_mask] = 0.0

    return {
        "Policy_Status": policy_status,
        "Policy_Start_Date": start_dates,
        "Effective_Date": start_dates.copy(),
        "Policy_End_Date": end_dates,
        "Expiry_Date": end_dates.copy(),
        "Premium_Payment_Status": payment_status,
        "Outstanding_Balance": np.round(outstanding_balance, 2),
    }


def get_time_of_day(hour: int, minute: int = 0, second: int = 0) -> str:
    hour_of_day = _hour_fraction(hour, minute, second)
    if 5.0 <= hour_of_day < 10.0:
        return "Morning Rush"
    if 10.0 <= hour_of_day < 17.0:
        return "Mid-Day"
    if 17.0 <= hour_of_day < 20.0:
        return "Evening Rush"
    if 20.0 <= hour_of_day < 22.0:
        return "Night"
    if hour_of_day >= 22.0 or hour_of_day < 2.0:
        return "Late Night"
    return "Dawn"

def get_day_of_week(dt: datetime) -> str:
    return dt.strftime("%A")

def generate_placeholder_plates(size: int, start_index: int = 0) -> np.ndarray:
    size = max(int(size), 0)
    if size == 0:
        return np.array([], dtype=object)

    indices = np.arange(start_index, start_index + size, dtype=int)
    prefix_indices = (indices // 9000) % len(PLATE_PREFIX_POOL)
    suffix_values = (indices % 9000) + 1000
    prefixes = PLATE_PREFIX_POOL[prefix_indices].astype(str)
    suffixes = np.char.zfill(suffix_values.astype(str), 4)
    return np.char.add(prefixes, suffixes).astype(object)

def fill_missing_plate_values(plates: pd.Series) -> pd.Series:
    plate_series = plates.copy().astype(object)
    missing_mask = plate_series.isna() | plate_series.astype(str).str.strip().eq("")
    if missing_mask.any():
        plate_series.loc[missing_mask] = generate_placeholder_plates(int(missing_mask.sum()))
    return plate_series.astype(str)

def generate_plates(rng: np.random.Generator, size: int) -> np.ndarray:
    prefixes = rng.choice(PLATE_PREFIX_POOL, size=size, replace=True).astype(str)
    suffixes = np.char.zfill(rng.integers(1000, 10000, size=size).astype(str), 4)
    return np.char.add(prefixes, suffixes).astype(object)

def _direction_to_bearing(direction: str) -> float:
    return DIRECTION_BEARINGS.get(direction, 0.0)

def _move_gps(lat: float, lon: float, direction: str, speed: float, update_interval_seconds: float) -> Tuple[float, float]:
    if speed <= 0.0:
        return lat + np.random.uniform(-0.00008, 0.00008), lon + np.random.uniform(-0.00008, 0.00008)
    distance_km = speed * (update_interval_seconds / 3600.0)
    bearing = np.deg2rad(_direction_to_bearing(direction))
    lat_move = distance_km / 111.0 * np.cos(bearing)
    lon_move = distance_km / max(1e-6, 111.0 * np.cos(np.deg2rad(lat))) * np.sin(bearing)
    return (
        float(np.clip(lat + lat_move + np.random.uniform(-0.00012, 0.00012), -90.0, 90.0)),
        float(np.clip(lon + lon_move + np.random.uniform(-0.00012, 0.00012), -180.0, 180.0)),
    )


@lru_cache(maxsize=None)
def _road_profile_weights(city: str, usage: str, time_of_day: str, vehicle_type: str = "") -> Tuple[float, ...]:
    profile = CITY_ROAD_PROFILES.get(city, DEFAULT_CITY_ROAD_PROFILE)
    weights = np.array([profile.get(rt, 0.1) for rt in Road_Types], dtype=float)
    weights /= weights.sum()
    if usage == "Commercial" and vehicle_type not in PUBLIC_TRANSPORT_TYPES:
        weights *= np.array([1.1 if rt in ["Highway", "Gravel"] else 0.9 for rt in Road_Types], dtype=float)
    if vehicle_type in PUBLIC_TRANSPORT_TYPES:
        weights *= np.array([1.02, 0.88, 0.94, 1.16, 1.18, 0.92], dtype=float)
    elif vehicle_type in {"Pickup", "Light Truck", "Heavy Truck"}:
        weights *= np.array([1.16, 1.08, 0.92, 0.86, 0.90, 1.10], dtype=float)
    if time_of_day == "Night":
        weights *= np.array([0.9, 1.05, 1.10, 0.95, 1.0, 0.9], dtype=float)
    weights = np.clip(weights, 0.01, None)
    weights /= weights.sum()
    return tuple(weights.tolist())


def _vehicle_service_masks(vehicle_types: np.ndarray, usage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vehicle_types = np.asarray(vehicle_types, dtype=object).astype(str)
    usage = np.asarray(usage, dtype=object).astype(str)
    public_transport_mask = np.isin(vehicle_types, list(PUBLIC_TRANSPORT_TYPES))
    commercial_service_mask = (usage == "Commercial") | np.isin(vehicle_types, list(COMMERCIAL_SERVICE_TYPES))
    return public_transport_mask, commercial_service_mask


def _map_city_factor(city_codes: np.ndarray, factor_map: dict[str, float], default: float = 0.0) -> np.ndarray:
    return np.array([factor_map.get(str(city), default) for city in city_codes], dtype=float)


def _sample_city_codes_for_fleet(
    vehicle_types: np.ndarray,
    usage: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    public_transport_mask, commercial_service_mask = _vehicle_service_masks(vehicle_types, usage)
    commercial_only_mask = commercial_service_mask & ~public_transport_mask
    private_mask = ~(public_transport_mask | commercial_only_mask)
    city_codes = np.empty(len(vehicle_types), dtype=object)

    if np.any(public_transport_mask):
        city_codes[public_transport_mask] = rng.choice(
            CITIES,
            size=int(public_transport_mask.sum()),
            p=PUBLIC_TRANSPORT_CITY_SELECTION_WEIGHTS,
        )
    if np.any(commercial_only_mask):
        city_codes[commercial_only_mask] = rng.choice(
            CITIES,
            size=int(commercial_only_mask.sum()),
            p=COMMERCIAL_SERVICE_CITY_SELECTION_WEIGHTS,
        )
    if np.any(private_mask):
        city_codes[private_mask] = rng.choice(
            CITIES,
            size=int(private_mask.sum()),
            p=PRIVATE_CITY_SELECTION_WEIGHTS,
        )

    return city_codes


def _sample_trip_distance_targets(
    daily_distance_target_km: np.ndarray,
    public_transport_mask: np.ndarray,
    commercial_service_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    vehicle_count = len(daily_distance_target_km)
    trip_fraction = np.where(
        public_transport_mask,
        rng.uniform(0.18, 0.34, size=vehicle_count),
        np.where(
            commercial_service_mask,
            rng.uniform(0.22, 0.38, size=vehicle_count),
            rng.uniform(0.28, 0.52, size=vehicle_count),
        ),
    )
    base_targets = daily_distance_target_km * trip_fraction
    base_targets += np.where(
        public_transport_mask,
        rng.normal(8.0, 3.0, size=vehicle_count),
        np.where(commercial_service_mask, rng.normal(5.0, 2.5, size=vehicle_count), rng.normal(2.0, 1.8, size=vehicle_count)),
    )
    return np.clip(base_targets, 4.0, 120.0).astype(float)


def _sample_status_target_minutes(
    is_driving: np.ndarray,
    drive_session_minutes: np.ndarray,
    park_session_minutes: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    base_minutes = np.where(is_driving, drive_session_minutes, park_session_minutes).astype(float)
    target_minutes = base_minutes * rng.uniform(0.72, 1.38, size=len(base_minutes)) + rng.normal(0.0, 4.0, size=len(base_minutes))
    return np.where(
        is_driving,
        np.clip(target_minutes, 12.0, 360.0),
        np.clip(target_minutes, 10.0, 480.0),
    ).astype(float)


def _calculate_desired_driving_probability(
    city_codes: np.ndarray,
    driving_propensity: np.ndarray,
    public_transport_mask: np.ndarray,
    commercial_service_mask: np.ndarray,
    time_of_day: str,
    peak_hour_bias: np.ndarray,
    late_night_bias: np.ndarray,
    hour_of_day: Optional[float] = None,
) -> np.ndarray:
    if time_of_day == "Morning Rush":
        desired = np.where(public_transport_mask, 0.88, np.where(commercial_service_mask, 0.78, 0.66))
    elif time_of_day == "Evening Rush":
        desired = np.where(public_transport_mask, 0.80, np.where(commercial_service_mask, 0.70, 0.58))
    elif time_of_day == "Mid-Day":
        desired = np.where(public_transport_mask, 0.54, np.where(commercial_service_mask, 0.42, 0.24))
    elif time_of_day == "Night":
        desired = np.where(public_transport_mask, 0.36, np.where(commercial_service_mask, 0.22, 0.11))
    elif time_of_day == "Late Night":
        desired = np.where(public_transport_mask, 0.28, np.where(commercial_service_mask, 0.14, 0.05))
    else:
        desired = np.where(public_transport_mask, 0.20, np.where(commercial_service_mask, 0.12, 0.05))

    desired += 0.18 * (driving_propensity - 0.25)

    if hour_of_day is None:
        category_reference = {
            "Morning Rush": 0.78,
            "Mid-Day": 0.46,
            "Evening Rush": 0.82,
            "Night": 0.44,
            "Late Night": 0.18,
            "Dawn": 0.22,
        }
        actual_intensity = category_reference.get(time_of_day, 0.40)
    else:
        actual_intensity = _hourly_activity_intensity(hour_of_day)

    category_reference = {
        "Morning Rush": 0.78,
        "Mid-Day": 0.46,
        "Evening Rush": 0.82,
        "Night": 0.44,
        "Late Night": 0.18,
        "Dawn": 0.22,
    }
    intensity_delta = actual_intensity - category_reference.get(time_of_day, 0.40)
    desired += intensity_delta * np.where(public_transport_mask, 0.24, np.where(commercial_service_mask, 0.18, 0.15))

    if time_of_day in {"Morning Rush", "Evening Rush"}:
        desired += (peak_hour_bias - 1.0) * 0.18
        desired += _map_city_factor(city_codes, CITY_PEAK_ACTIVITY_BOOST, default=0.01) * np.where(
            public_transport_mask,
            1.0,
            np.where(commercial_service_mask, 0.75, 0.60),
        )
    elif time_of_day == "Night":
        desired += (late_night_bias - 1.0) * np.where(public_transport_mask, 0.16, np.where(commercial_service_mask, 0.10, 0.04))
        desired += _map_city_factor(city_codes, CITY_LATE_NIGHT_ACTIVITY_BOOST, default=0.0) * np.where(
            public_transport_mask,
            0.90,
            np.where(commercial_service_mask, 0.65, 0.30),
        )
    elif time_of_day == "Late Night":
        desired += (late_night_bias - 1.0) * np.where(public_transport_mask, 0.18, np.where(commercial_service_mask, 0.12, 0.05))
        desired += _map_city_factor(city_codes, CITY_LATE_NIGHT_ACTIVITY_BOOST, default=0.0) * np.where(
            public_transport_mask,
            1.15,
            np.where(commercial_service_mask, 0.80, 0.35),
        )
    else:
        desired += (late_night_bias - 1.0) * np.where(public_transport_mask, 0.10, np.where(commercial_service_mask, 0.06, 0.03))
        desired += _map_city_factor(city_codes, CITY_LATE_NIGHT_ACTIVITY_BOOST, default=0.0) * np.where(
            public_transport_mask,
            0.55,
            np.where(commercial_service_mask, 0.45, 0.20),
        )

    return np.clip(desired, 0.02, 0.96).astype(float)


def _sample_vehicle_years(vehicle_types: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    years = np.zeros(len(vehicle_types), dtype=np.int32)
    for vehicle_type, (min_year, max_year) in VEHICLE_YEAR_RANGES.items():
        mask = np.asarray(vehicle_types, dtype=object) == vehicle_type
        if np.any(mask):
            years[mask] = rng.integers(min_year, max_year + 1, int(mask.sum()), dtype=np.int32)
    fallback_mask = years == 0
    if np.any(fallback_mask):
        years[fallback_mask] = rng.integers(2012, 2026, int(fallback_mask.sum()), dtype=np.int32)
    return years


def _build_vehicle_latent_profiles(
    vehicle_types: np.ndarray,
    usage: np.ndarray,
    years: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    public_transport_mask, commercial_service_mask = _vehicle_service_masks(vehicle_types, usage)
    vehicle_age = (2026 - years.astype(float)).clip(min=0.0)
    vehicle_count = len(vehicle_types)

    risk_profile = rng.beta(2.4, 10.5, size=vehicle_count)
    risk_profile += np.where(commercial_service_mask, 0.03, 0.0)
    risk_profile += np.where(public_transport_mask, 0.06, 0.0)
    risk_profile += np.clip((vehicle_age - 8.0) / 80.0, 0.0, 0.08)

    tail_probability = np.where(public_transport_mask, 0.08, np.where(commercial_service_mask, 0.03, 0.005))
    tail_mask = rng.random(vehicle_count) < tail_probability
    risk_profile += np.where(tail_mask, rng.uniform(0.10, 0.18, size=vehicle_count), 0.0)
    risk_profile = np.clip(risk_profile, 0.10, 0.88)

    driving_propensity = (
        np.where(public_transport_mask, 0.46, np.where(commercial_service_mask, 0.32, 0.18))
        + 0.18 * risk_profile
        + rng.normal(0.0, 0.03, size=vehicle_count)
    )
    driving_propensity = np.clip(driving_propensity, 0.08, 0.82)

    target_speed_ratio = (
        np.where(public_transport_mask, 0.82, np.where(commercial_service_mask, 0.78, 0.72))
        + 0.28 * (risk_profile - 0.25)
        + rng.normal(0.0, 0.04, size=vehicle_count)
    )
    target_speed_ratio = np.clip(target_speed_ratio, 0.55, 1.08)

    harsh_event_probability = (
        0.006
        + 0.028 * risk_profile
        + np.where(public_transport_mask, 0.010, 0.0)
        + np.where(commercial_service_mask & ~public_transport_mask, 0.005, 0.0)
    )
    harsh_event_probability = np.clip(harsh_event_probability, 0.003, 0.055)

    maintenance_risk = (
        0.10
        + 0.45 * np.clip(vehicle_age / 20.0, 0.0, 1.0)
        + np.where(public_transport_mask, 0.10, 0.0)
        + rng.normal(0.0, 0.04, size=vehicle_count)
    )
    maintenance_risk = np.clip(maintenance_risk, 0.02, 0.95)

    daily_distance_target_km = np.where(
        public_transport_mask,
        rng.normal(170.0, 38.0, size=vehicle_count),
        np.where(commercial_service_mask, rng.normal(102.0, 30.0, size=vehicle_count), rng.normal(46.0, 15.0, size=vehicle_count)),
    )
    low_distance_mask = rng.random(vehicle_count) < np.where(public_transport_mask, 0.20, np.where(commercial_service_mask, 0.26, 0.34))
    daily_distance_target_km = np.where(
        low_distance_mask,
        daily_distance_target_km * rng.uniform(0.52, 0.78, size=vehicle_count),
        daily_distance_target_km * rng.uniform(0.95, 1.26, size=vehicle_count),
    )
    daily_distance_target_km += np.where(public_transport_mask, 25.0, np.where(commercial_service_mask, 12.0, 0.0)) * np.clip(driving_propensity, 0.10, 0.90)
    daily_distance_target_km = np.clip(daily_distance_target_km, 18.0, 320.0)

    drive_session_minutes = np.where(
        public_transport_mask,
        rng.normal(108.0, 28.0, size=vehicle_count),
        np.where(commercial_service_mask, rng.normal(74.0, 24.0, size=vehicle_count), rng.normal(34.0, 14.0, size=vehicle_count)),
    )
    drive_session_minutes += np.clip(
        daily_distance_target_km - np.where(public_transport_mask, 150.0, np.where(commercial_service_mask, 90.0, 45.0)),
        -30.0,
        90.0,
    ) * 0.25
    drive_session_minutes = np.clip(drive_session_minutes, 15.0, 300.0)

    park_session_minutes = np.where(
        public_transport_mask,
        rng.normal(26.0, 10.0, size=vehicle_count),
        np.where(commercial_service_mask, rng.normal(56.0, 18.0, size=vehicle_count), rng.normal(128.0, 42.0, size=vehicle_count)),
    )
    park_session_minutes -= np.clip(
        daily_distance_target_km - np.where(public_transport_mask, 150.0, np.where(commercial_service_mask, 90.0, 45.0)),
        -35.0,
        80.0,
    ) * 0.10
    park_session_minutes = np.clip(park_session_minutes, 12.0, 420.0)

    peak_hour_bias = np.clip(
        rng.normal(np.where(public_transport_mask, 1.10, np.where(commercial_service_mask, 1.05, 1.00)), 0.08, size=vehicle_count),
        0.82,
        1.28,
    )
    late_night_bias = np.clip(
        rng.normal(np.where(public_transport_mask, 1.20, np.where(commercial_service_mask, 1.06, 0.82)), 0.10, size=vehicle_count),
        0.65,
        1.40,
    )

    return {
        "risk_profile": risk_profile.astype(float),
        "driving_propensity": driving_propensity.astype(float),
        "target_speed_ratio": target_speed_ratio.astype(float),
        "harsh_event_probability": harsh_event_probability.astype(float),
        "maintenance_risk": maintenance_risk.astype(float),
        "daily_distance_target_km": daily_distance_target_km.astype(float),
        "drive_session_minutes": drive_session_minutes.astype(float),
        "park_session_minutes": park_session_minutes.astype(float),
        "peak_hour_bias": peak_hour_bias.astype(float),
        "late_night_bias": late_night_bias.astype(float),
        "public_transport_flag": public_transport_mask.astype(int),
        "commercial_service_flag": commercial_service_mask.astype(int),
    }


def _weather_speed_adjustment(weather: np.ndarray) -> np.ndarray:
    factors = np.ones(len(weather), dtype=float)
    weather = np.asarray(weather, dtype=object).astype(str)
    slowdown_map = {
        "Thundery / Heavy Rain": 0.82,
        "Heavy Rain": 0.82,
        "Misty / Foggy": 0.86,
        "Misty": 0.88,
        "Windy / Gusty": 0.94,
        "Windy": 0.95,
        "Hazy": 0.96,
    }
    for label, factor in slowdown_map.items():
        factors[weather == label] = factor
    return factors


def _base_fuel_efficiency(vehicle_types: np.ndarray, maintenance_risk: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    base = np.array([VEHICLE_FUEL_EFFICIENCY_BASE.get(str(vehicle_type), 8.5) for vehicle_type in vehicle_types], dtype=float)
    adjusted = base + (maintenance_risk * np.where(base >= 12.0, 2.0, 1.0)) + rng.normal(0.0, 0.35, len(base))
    return np.clip(adjusted, 4.8, 34.0)


def _cap_harsh_event_totals(
    harsh_brakes: np.ndarray,
    harsh_accels: np.ndarray,
    harsh_corners: np.ndarray,
    vehicle_types: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = harsh_brakes + harsh_accels + harsh_corners
    public_transport_mask, commercial_service_mask = _vehicle_service_masks(vehicle_types, np.where(np.isin(vehicle_types, list(PUBLIC_VEHICLE_TYPES)), "Commercial", "Private"))
    caps = np.where(public_transport_mask, 8, np.where(commercial_service_mask, 6, 4)).astype(int)
    over_mask = total > caps
    if np.any(over_mask):
        scale = np.divide(caps[over_mask], np.maximum(total[over_mask], 1), dtype=float)
        harsh_brakes[over_mask] = np.floor(harsh_brakes[over_mask] * scale).astype(int)
        harsh_accels[over_mask] = np.floor(harsh_accels[over_mask] * scale).astype(int)
        remaining = caps[over_mask] - harsh_brakes[over_mask] - harsh_accels[over_mask]
        harsh_corners[over_mask] = np.clip(remaining, 0, None).astype(int)
    return harsh_brakes, harsh_accels, harsh_corners


def _weather_for_cycle(current_time: datetime, rng: np.random.Generator) -> dict[str, str]:
    weather_by_city: dict[str, str] = {}
    period = "day" if 6 <= current_time.hour < 18 else "night"
    season = "wet" if current_time.month in (11, 12, 1, 2, 3) else "dry"
    for city in CITIES:
        region = CITY_WEATHER_ZONES.get(city, "highveld")
        options = CITY_WEATHER.get(region, {}).get(period, {}).get(season, {})
        if options:
            labels = list(options.keys())
            probs = np.array(list(options.values()), dtype=float)
            probs /= probs.sum()
            weather_by_city[city] = normalize_weather_label(rng.choice(labels, p=probs))
        else:
            weather_by_city[city] = "Sunny" if period == "day" else "Clear"
    return weather_by_city


def _road_limits_for_types(road_types: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    road_min = np.full(len(road_types), 25.0, dtype=float)
    road_max = np.full(len(road_types), 85.0, dtype=float)
    for road_name, (minimum, maximum) in ROAD_LIMITS.items():
        mask = road_types == road_name
        if np.any(mask):
            road_min[mask] = minimum
            road_max[mask] = maximum
    return road_min, road_max


def _sample_road_types(
    city_codes: np.ndarray,
    usage: np.ndarray,
    vehicle_types: np.ndarray,
    time_of_day: str,
    active_mask: np.ndarray,
    rng: np.random.Generator,
    fallback: np.ndarray,
) -> np.ndarray:
    road_types = fallback.copy()
    if not np.any(active_mask):
        return road_types

    keep_mask = active_mask & np.isin(fallback, Road_Types) & (rng.random(len(fallback)) < 0.90)
    resample_mask = active_mask & ~keep_mask
    if not np.any(resample_mask):
        return road_types

    active_indices = np.flatnonzero(resample_mask)
    for city in np.unique(city_codes[active_indices]):
        city_mask = resample_mask & (city_codes == city)
        for usage_value in np.unique(usage[city_mask]):
            usage_mask = city_mask & (usage == usage_value)
            for vehicle_type in np.unique(vehicle_types[usage_mask]):
                combo_mask = usage_mask & (vehicle_types == vehicle_type)
                weights = np.array(_road_profile_weights(str(city), str(usage_value), time_of_day, str(vehicle_type)), dtype=float)
                road_types[combo_mask] = rng.choice(Road_Types, size=int(combo_mask.sum()), p=weights)
    return road_types


def _sample_directions(previous_directions: np.ndarray, active_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    directions = previous_directions.copy()
    if not np.any(active_mask):
        return directions

    keep_mask = active_mask & (rng.random(len(previous_directions)) < 0.88)
    change_mask = active_mask & ~keep_mask
    for direction in DIRECTIONS:
        mask = change_mask & (previous_directions == direction)
        if np.any(mask):
            alternatives = [item for item in DIRECTIONS if item != direction]
            directions[mask] = rng.choice(alternatives, size=int(mask.sum()))

    missing_mask = active_mask & ~np.isin(directions, DIRECTIONS)
    if np.any(missing_mask):
        directions[missing_mask] = rng.choice(DIRECTIONS, size=int(missing_mask.sum()))
    return directions


def _move_gps_batch(latitudes: np.ndarray, longitudes: np.ndarray, directions: np.ndarray, speeds: np.ndarray, update_interval_seconds: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    new_latitudes = latitudes.astype(float, copy=True)
    new_longitudes = longitudes.astype(float, copy=True)
    stationary_mask = speeds <= 0.0
    moving_mask = ~stationary_mask

    if np.any(stationary_mask):
        new_latitudes[stationary_mask] += rng.uniform(-0.00008, 0.00008, int(stationary_mask.sum()))
        new_longitudes[stationary_mask] += rng.uniform(-0.00008, 0.00008, int(stationary_mask.sum()))

    if np.any(moving_mask):
        distance_km = speeds[moving_mask] * (update_interval_seconds / 3600.0)
        direction_subset = directions[moving_mask]
        bearings_deg = np.zeros(direction_subset.shape[0], dtype=float)
        for direction_name, degrees in DIRECTION_BEARINGS.items():
            bearings_deg[direction_subset == direction_name] = degrees
        bearings = np.deg2rad(bearings_deg)
        lat_move = distance_km / 111.0 * np.cos(bearings)
        lon_denom = np.maximum(1e-6, 111.0 * np.cos(np.deg2rad(latitudes[moving_mask])))
        lon_move = distance_km / lon_denom * np.sin(bearings)
        new_latitudes[moving_mask] = np.clip(
            latitudes[moving_mask] + lat_move + rng.uniform(-0.00012, 0.00012, int(moving_mask.sum())),
            -90.0,
            90.0,
        )
        new_longitudes[moving_mask] = np.clip(
            longitudes[moving_mask] + lon_move + rng.uniform(-0.00012, 0.00012, int(moving_mask.sum())),
            -180.0,
            180.0,
        )

    return new_latitudes, new_longitudes

def generate_live_status(row: pd.Series, current_time: datetime, rng: np.random.Generator) -> str:
    prev_status = str(row.get('Status', 'Parked'))
    usage = str(row.get('Usage', 'Private'))
    vehicle_type = str(row.get('Type', 'Sedan'))
    city = str(row.get('City', 'Harare'))
    tod = get_time_of_day(current_time.hour, current_time.minute, current_time.second)
    hour_of_day = _datetime_hour_fraction(current_time)
    public_transport_mask = np.array([vehicle_type in PUBLIC_TRANSPORT_TYPES], dtype=bool)
    commercial_service_mask = np.array([(usage == 'Commercial') or (vehicle_type in COMMERCIAL_SERVICE_TYPES)], dtype=bool)
    desired_drive_prob = _calculate_desired_driving_probability(
        np.array([city], dtype=object),
        np.array([float(row.get('_Driving_Propensity', 0.18))], dtype=float),
        public_transport_mask,
        commercial_service_mask,
        tod,
        np.array([float(row.get('_Peak_Hour_Bias', 1.0))], dtype=float),
        np.array([float(row.get('_Late_Night_Bias', 1.0))], dtype=float),
        hour_of_day=hour_of_day,
    )[0]
    if prev_status == 'Driving':
        keep_prob = np.clip(0.18 + (0.76 * desired_drive_prob), 0.08, 0.98)
        return 'Driving' if rng.random() < keep_prob else 'Parked'
    return 'Driving' if rng.random() < desired_drive_prob else 'Parked'

def generate_live_road_type(row: pd.Series, rng: np.random.Generator) -> str:
    city = str(row.get('City', 'Harare'))
    usage = str(row.get('Usage', 'Private'))
    tod = str(row.get('Time_of_Day', 'Mid-Day'))
    vehicle_type = str(row.get('Type', 'Sedan'))
    weights = np.array(_road_profile_weights(city, usage, tod, vehicle_type), dtype=float)
    return str(rng.choice(Road_Types, p=weights))

def generate_live_direction(row: pd.Series, rng: np.random.Generator) -> str:
    prev_direction = str(row.get('Direction', 'North'))
    options = DIRECTIONS.copy()
    if prev_direction in DIRECTIONS:
        options = [prev_direction] + [d for d in DIRECTIONS if d != prev_direction]
    probs = [0.35] + [0.65 / (len(options) - 1)] * (len(options) - 1)
    return str(rng.choice(options, p=probs))

def _generate_weather_for_cities(cities: pd.Series, current_time: datetime, rng: np.random.Generator) -> np.ndarray:
    period = 'day' if 6 <= current_time.hour < 18 else 'night'
    season = 'wet' if current_time.month in (11, 12, 1, 2, 3) else 'dry'
    weather_list = []
    for city in cities:
        region = CITY_WEATHER_ZONES.get(city, 'highveld')
        options = CITY_WEATHER.get(region, {}).get(period, {}).get(season, {})
        if options:
            labels = list(options.keys())
            probs = np.array(list(options.values()), dtype=float)
            probs /= probs.sum()
            weather_list.append(normalize_weather_label(rng.choice(labels, p=probs)))
        else:
            weather_list.append('Sunny' if period == 'day' else 'Clear')
    return np.array(weather_list, dtype=object)

def _ensure_live_columns_populated(df: pd.DataFrame, mode: str = 'raw') -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=(LIVE_PREPROCESSED_ALL_COLUMNS if mode == 'preprocessed' else LIVE_RAW_COLUMNS))
    df = df.copy()
    current_time = _current_local_time()
    now_ts = current_time.strftime('%Y-%m-%d %H:%M:%S')
    
    for col in STATIC_VEHICLE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df['Plate'] = fill_missing_plate_values(df['Plate'])
    df['Type'] = df['Type'].fillna('Sedan')
    df['Make'] = df['Make'].fillna('Unknown')
    df['Model'] = df['Model'].fillna('Unknown')
    df['Class'] = df['Class'].fillna('Unknown')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(current_time.year).astype(int)
    df['Engine_CC'] = pd.to_numeric(df['Engine_CC'], errors='coerce').fillna(1500).astype(int)
    df['Base_Price_USD'] = pd.to_numeric(df['Base_Price_USD'], errors='coerce').fillna(600.0).astype(float)
    df['City'] = df['City'].fillna('Harare')
    df['Usage'] = df['Usage'].fillna('Private')

    default_policy_start = (current_time.date() - timedelta(days=120)).isoformat()
    default_policy_end = (current_time.date() + timedelta(days=245)).isoformat()
    df['Policy_Status'] = df.get('Policy_Status', pd.Series('Active', index=df.index)).fillna('Active').astype(str)
    df['Policy_Start_Date'] = df.get('Policy_Start_Date', pd.Series(default_policy_start, index=df.index)).fillna(default_policy_start).astype(str)
    df['Effective_Date'] = df.get('Effective_Date', df['Policy_Start_Date']).fillna(df['Policy_Start_Date']).astype(str)
    df['Policy_End_Date'] = df.get('Policy_End_Date', pd.Series(default_policy_end, index=df.index)).fillna(default_policy_end).astype(str)
    df['Expiry_Date'] = df.get('Expiry_Date', df['Policy_End_Date']).fillna(df['Policy_End_Date']).astype(str)
    df['Premium_Payment_Status'] = (
        df.get('Premium_Payment_Status', pd.Series('Paid', index=df.index))
        .fillna('Paid')
        .astype(str)
    )
    df['Outstanding_Balance'] = pd.to_numeric(
        df.get('Outstanding_Balance', pd.Series(0.0, index=df.index)),
        errors='coerce',
    ).fillna(0.0).clip(lower=0.0).round(2).astype(float)

    for col in LIVE_DYNAMIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    ts_short = current_time.strftime('%H%M%S')
    if 'Trip_ID' in df.columns:
        try:
            mask = df['Trip_ID'].isna()
            if mask.any():
                df.loc[mask, 'Trip_ID'] = df.loc[mask, 'Plate'].astype(str).apply(lambda p: f"IDLE-{p}-{ts_short}")
        except Exception:
            df['Trip_ID'] = df['Trip_ID'].fillna('IDLE-UNKNOWN-%s' % ts_short)

    df['Road_Type'] = df['Road_Type'].fillna('Urban')
    df['Weather'] = df['Weather'].fillna('Sunny')
    df['Status'] = df['Status'].fillna('Parked')
    df['Direction'] = df['Direction'].fillna('North')
    df['Trip_Distance_km'] = pd.to_numeric(df['Trip_Distance_km'], errors='coerce').fillna(0.0).astype(float)
    df['Trip_Duration_Hour'] = pd.to_numeric(df['Trip_Duration_Hour'], errors='coerce').fillna(0.0).astype(float)
    df['Speed_kmh'] = pd.to_numeric(df['Speed_kmh'], errors='coerce').fillna(0.0).astype(float)
    df['Acceleration_mps2'] = pd.to_numeric(df['Acceleration_mps2'], errors='coerce').fillna(0.0).astype(float)
    df['RPM'] = pd.to_numeric(df['RPM'], errors='coerce').fillna(700.0).astype(float)
    df['Throttle_pct'] = pd.to_numeric(df['Throttle_pct'], errors='coerce').fillna(0.0).astype(float)
    df['Engine_Load_pct'] = pd.to_numeric(df['Engine_Load_pct'], errors='coerce').fillna(5.0).astype(float)
    df['Coolant_Temp_C'] = pd.to_numeric(df['Coolant_Temp_C'], errors='coerce').fillna(75.0).astype(float)
    df['MAF_gs'] = pd.to_numeric(df['MAF_gs'], errors='coerce').fillna(1.0).astype(float)
    df['Battery_V'] = pd.to_numeric(df['Battery_V'], errors='coerce').fillna(12.6).astype(float)

    lat_default = df['City'].map(lambda c: CITY_COORDS.get(c, CITY_COORDS.get('Harare'))[0] if pd.notna(c) else CITY_COORDS['Harare'][0])
    lon_default = df['City'].map(lambda c: CITY_COORDS.get(c, CITY_COORDS.get('Harare'))[1] if pd.notna(c) else CITY_COORDS['Harare'][1])
    df['GPS_Latitude'] = pd.to_numeric(df['GPS_Latitude'], errors='coerce').fillna(lat_default).astype(float)
    df['GPS_Longitude'] = pd.to_numeric(df['GPS_Longitude'], errors='coerce').fillna(lon_default).astype(float)

    df['Harsh_Brake_Count'] = pd.to_numeric(df['Harsh_Brake_Count'], errors='coerce').fillna(0).astype(int)
    df['Harsh_Accel_Count'] = pd.to_numeric(df['Harsh_Accel_Count'], errors='coerce').fillna(0).astype(int)
    df['Harsh_Corner_Count'] = pd.to_numeric(df['Harsh_Corner_Count'], errors='coerce').fillna(0).astype(int)
    df['Total_Harsh_Events_per_Day'] = pd.to_numeric(df['Total_Harsh_Events_per_Day'], errors='coerce').fillna(df['Harsh_Brake_Count'] + df['Harsh_Accel_Count'] + df['Harsh_Corner_Count']).astype(int)
    df['Fuel_Efficiency_L_per_100km'] = pd.to_numeric(df['Fuel_Efficiency_L_per_100km'], errors='coerce').fillna(8.5).astype(float)
    df['Last_Update'] = df['Last_Update'].fillna(now_ts)
    df['Speeding_Flag'] = pd.to_numeric(df['Speeding_Flag'], errors='coerce').fillna(0).astype(int)
    df['Status_Change_Timestamp'] = df['Status_Change_Timestamp'].fillna(now_ts)
    df['Time_of_Day'] = df['Time_of_Day'].fillna(get_time_of_day(current_time.hour, current_time.minute, current_time.second))
    df['Day_of_Week'] = df['Day_of_Week'].fillna(get_day_of_week(current_time))

    if mode == 'preprocessed':
        for col in LIVE_PREPROCESSED_ENGINEERED_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0

        df['Recent_Harsh_Events'] = pd.to_numeric(df.get('Recent_Harsh_Events', pd.Series(np.nan, index=df.index)), errors='coerce').fillna(df['Total_Harsh_Events_per_Day']).astype(float)
        df['Recent_Avg_Speed'] = pd.to_numeric(df.get('Recent_Avg_Speed', pd.Series(np.nan, index=df.index)), errors='coerce').fillna(df['Speed_kmh'] * 0.95).astype(float)
        fallback_night = pd.Series(np.where(df['Night_Driving_Flag'] == 1, df['Trip_Distance_km'] * 0.45, df['Trip_Distance_km'] * 0.08), index=df.index)
        df['Recent_Night_Distance'] = pd.to_numeric(df.get('Recent_Night_Distance', pd.Series(np.nan, index=df.index)), errors='coerce').fillna(fallback_night).astype(float)
        df['Battery_Health_Score'] = pd.to_numeric(df.get('Battery_Health_Score', pd.Series(np.nan, index=df.index)), errors='coerce').fillna((df['Battery_V'] - 11.5) * 50.0).clip(0.0, 100.0).astype(float)

    return df

# ──────────────────────────────────────────────────────────────────────────────
# FLEET MANAGER
# ──────────────────────────────────────────────────────────────────────────────

class FleetManager:
    def __init__(self, num_vehicles: int = NUM_VEHICLES, update_interval_seconds: float = UPDATE_INTERVAL_SECONDS):
        self.lock = threading.Lock()
        self.num_vehicles = num_vehicles
        self.update_interval_seconds = update_interval_seconds
        self._last_update_monotonic = None
        self._snapshot_version = 0
        
        # Initialize fleet
        self.fleet = self._generate_initial_fleet()
        self.rng = np.random.default_rng(seed=42)
        
    def _generate_initial_fleet(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        catalog_weights = VEHICLE_CATALOG['weight'].to_numpy(dtype=float, copy=False)
        catalog_weights = catalog_weights / catalog_weights.sum()
        catalog_indices = rng.choice(len(VEHICLE_CATALOG), size=self.num_vehicles, p=catalog_weights)
        catalog = VEHICLE_CATALOG.iloc[catalog_indices].reset_index(drop=True)
        
        vehicle_types = catalog['type'].to_numpy(dtype=object, copy=False)
        usage = catalog['usage'].to_numpy(dtype=object, copy=False)
        public_transport_mask, commercial_service_mask = _vehicle_service_masks(vehicle_types, usage)
        city_codes = _sample_city_codes_for_fleet(vehicle_types, usage, rng)
        years = _sample_vehicle_years(vehicle_types, rng)
        latent_profiles = _build_vehicle_latent_profiles(vehicle_types, usage, years, rng)
        
        now = _current_local_time()
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
        policy_fields = _generate_policy_fields(self.num_vehicles, now, rng)
        time_of_day = get_time_of_day(now.hour, now.minute, now.second)
        day_of_week = get_day_of_week(now)
        hour_of_day = _datetime_hour_fraction(now)

        desired_drive_prob = _calculate_desired_driving_probability(
            city_codes,
            latent_profiles['driving_propensity'],
            public_transport_mask,
            commercial_service_mask,
            time_of_day,
            latent_profiles['peak_hour_bias'],
            latent_profiles['late_night_bias'],
            hour_of_day=hour_of_day,
        )
        statuses = np.where(rng.random(self.num_vehicles) < desired_drive_prob, 'Driving', 'Parked')
        current_status_target_minutes = _sample_status_target_minutes(
            statuses == 'Driving',
            latent_profiles['drive_session_minutes'],
            latent_profiles['park_session_minutes'],
            rng,
        )
        elapsed_status_minutes = current_status_target_minutes * rng.uniform(0.08, 0.88, size=self.num_vehicles)
        status_change_times = (
            pd.Timestamp(now) - pd.to_timedelta(elapsed_status_minutes, unit='m')
        ).strftime('%Y-%m-%d %H:%M:%S')
        current_trip_target_km = _sample_trip_distance_targets(
            latent_profiles['daily_distance_target_km'],
            public_transport_mask,
            commercial_service_mask,
            rng,
        )

        weather = _generate_weather_for_cities(pd.Series(city_codes), now, rng)
        road_types = np.array(
            [
                generate_live_road_type(
                    pd.Series({'City': c, 'Usage': u, 'Type': t, 'Time_of_Day': time_of_day, 'Weather': w}),
                    rng,
                ) if s == 'Driving' else 'Urban'
                for c, u, t, w, s in zip(city_codes, usage, vehicle_types, weather, statuses)
            ],
            dtype=object,
        )
        directions = np.where(statuses == 'Driving', rng.choice(DIRECTIONS, size=self.num_vehicles), 'North')

        road_min, road_max = _road_limits_for_types(road_types)
        weather_speed_factor = _weather_speed_adjustment(weather)
        target_speed_ratio = latent_profiles['target_speed_ratio'] * weather_speed_factor
        allowed_excess = np.where(public_transport_mask, 10.0, np.where(commercial_service_mask, 6.0, 3.0))
        base_speed = road_max * target_speed_ratio
        speeds = np.where(
            statuses == 'Driving',
            np.clip(base_speed + rng.normal(0.0, 3.0, self.num_vehicles), road_min * 0.60, road_max + allowed_excess),
            0.0,
        )

        elapsed_status_hours = elapsed_status_minutes / 60.0
        trip_progress_speed = np.clip(
            speeds * rng.normal(0.74, 0.08, self.num_vehicles),
            road_min * 0.50,
            np.maximum(road_max * 0.92, road_min + 8.0),
        )
        trip_distances = np.where(
            statuses == 'Driving',
            np.round(
                np.clip(
                    np.minimum(current_trip_target_km, trip_progress_speed * elapsed_status_hours),
                    2.0,
                    current_trip_target_km * 1.05,
                ),
                2,
            ),
            0.0,
        )
        trip_durations = np.where(
            statuses == 'Driving',
            np.round(
                np.maximum(
                    np.divide(trip_distances, np.clip(trip_progress_speed, 15.0, 105.0)),
                    elapsed_status_hours * 0.35,
                ),
                3,
            ),
            0.0,
        )
        
        harsh_lambda = (
            0.20
            + 1.20 * latent_profiles['risk_profile']
            + np.where(public_transport_mask, 0.65, 0.0)
            + np.where(commercial_service_mask & ~public_transport_mask, 0.25, 0.0)
        )
        harsh_counts = rng.poisson(harsh_lambda, size=self.num_vehicles).astype(int)
        harsh_counts += np.where(public_transport_mask & (latent_profiles['risk_profile'] > 0.55), rng.integers(0, 3, self.num_vehicles), 0)
        harsh_counts = np.clip(harsh_counts, 0, 8)
        harsh_brakes = np.round(harsh_counts * 0.4).astype(int)
        harsh_accels = np.round(harsh_counts * 0.35).astype(int)
        harsh_corners = np.maximum(harsh_counts - harsh_brakes - harsh_accels, 0).astype(int)
        harsh_brakes, harsh_accels, harsh_corners = _cap_harsh_event_totals(harsh_brakes, harsh_accels, harsh_corners, vehicle_types)
        
        gps_lats = np.array([CITY_COORDS[city][0] for city in city_codes], dtype=float) + rng.uniform(-0.15, 0.15, self.num_vehicles)
        gps_lons = np.array([CITY_COORDS[city][1] for city in city_codes], dtype=float) + rng.uniform(-0.15, 0.15, self.num_vehicles)
        maintenance_risk = latent_profiles['maintenance_risk']
        acceleration = np.where(statuses == 'Driving', np.clip(rng.normal(0.0, 0.35 + 0.65 * latent_profiles['risk_profile'], self.num_vehicles), -2.2, 2.2), 0.0)
        rpm = np.where(statuses == 'Driving', 900.0 + (speeds / np.maximum(road_max, 1.0)) * 2600.0 + rng.normal(0.0, 120.0, self.num_vehicles), 700.0)
        throttle_pct = np.where(statuses == 'Driving', np.clip(18.0 + np.abs(acceleration) * 18.0 + latent_profiles['risk_profile'] * 20.0, 6.0, 72.0), 0.0)
        engine_load = np.where(statuses == 'Driving', np.clip(18.0 + (speeds / np.maximum(road_max, 1.0)) * 42.0 + maintenance_risk * 8.0, 8.0, 85.0), 5.0)
        coolant_temp = np.where(statuses == 'Driving', np.clip(73.0 + engine_load * 0.24 + rng.normal(0.0, 1.0, self.num_vehicles), 70.0, 101.0), 74.0)
        maf_gs = np.where(statuses == 'Driving', np.clip(1.4 + engine_load * 0.05 + rng.normal(0.0, 0.18, self.num_vehicles), 1.0, 7.5), 1.0)
        battery_voltage = np.clip(12.65 - (maintenance_risk * 0.55) - np.where(statuses == 'Driving', np.abs(acceleration) * 0.03, 0.0), 11.8, 12.8)
        fuel_efficiency = _base_fuel_efficiency(vehicle_types, maintenance_risk, rng)
        fuel_efficiency = np.where(statuses == 'Driving', fuel_efficiency * np.where((speeds >= 35.0) & (speeds <= 95.0), 1.0, 1.08), fuel_efficiency)
        speeding_flag = (speeds > (road_max + 3.0)).astype(int)
        
        fleet_data = {
            'Plate': generate_plates(rng, self.num_vehicles),
            'Type': vehicle_types,
            'Make': catalog['brand'].to_numpy(dtype=object, copy=False),
            'Model': catalog['model'].to_numpy(dtype=object, copy=False),
            'Class': catalog['cls'].to_numpy(dtype=object, copy=False),
            'Year': years,
            'Engine_CC': catalog['cc'].to_numpy(dtype=np.int32, copy=False),
            'Base_Price_USD': catalog['price'].to_numpy(dtype=float, copy=False),
            'City': city_codes,
            'Usage': usage,
            'Policy_Status': policy_fields['Policy_Status'],
            'Policy_Start_Date': policy_fields['Policy_Start_Date'],
            'Effective_Date': policy_fields['Effective_Date'],
            'Policy_End_Date': policy_fields['Policy_End_Date'],
            'Expiry_Date': policy_fields['Expiry_Date'],
            'Premium_Payment_Status': policy_fields['Premium_Payment_Status'],
            'Outstanding_Balance': policy_fields['Outstanding_Balance'],
            'Trip_ID': np.full(self.num_vehicles, None, dtype=object),
            'Trip_Distance_km': trip_distances,
            'Trip_Duration_Hour': trip_durations,
            'Road_Type': road_types,
            'Direction': directions,
            'Weather': weather,
            'Time_of_Day': np.full(self.num_vehicles, time_of_day, dtype=object),
            'Day_of_Week': np.full(self.num_vehicles, day_of_week, dtype=object),
            'Status': statuses,
            'Status_Change_Timestamp': status_change_times.to_numpy(dtype=object, copy=False),
            'Speed_kmh': np.round(speeds, 1),
            'Acceleration_mps2': np.round(acceleration, 2),
            'RPM': np.round(rpm, 0),
            'Throttle_pct': np.round(throttle_pct, 1),
            'Engine_Load_pct': np.round(engine_load, 1),
            'Coolant_Temp_C': np.round(coolant_temp, 1),
            'MAF_gs': np.round(maf_gs, 2),
            'Battery_V': np.round(battery_voltage, 2),
            'GPS_Latitude': gps_lats,
            'GPS_Longitude': gps_lons,
            'Harsh_Brake_Count': harsh_brakes,
            'Harsh_Accel_Count': harsh_accels,
            'Harsh_Corner_Count': harsh_corners,
            'Total_Harsh_Events_per_Day': harsh_brakes + harsh_accels + harsh_corners,
            'Fuel_Efficiency_L_per_100km': np.round(fuel_efficiency, 2),
            'Last_Update': np.full(self.num_vehicles, timestamp, dtype=object),
            'Speeding_Flag': speeding_flag,
            '_Risk_Profile': latent_profiles['risk_profile'],
            '_Driving_Propensity': latent_profiles['driving_propensity'],
            '_Target_Speed_Ratio': latent_profiles['target_speed_ratio'],
            '_Harsh_Event_Probability': latent_profiles['harsh_event_probability'],
            '_Maintenance_Risk': latent_profiles['maintenance_risk'],
            '_Daily_Distance_Target_km': latent_profiles['daily_distance_target_km'],
            '_Drive_Session_Minutes': latent_profiles['drive_session_minutes'],
            '_Park_Session_Minutes': latent_profiles['park_session_minutes'],
            '_Peak_Hour_Bias': latent_profiles['peak_hour_bias'],
            '_Late_Night_Bias': latent_profiles['late_night_bias'],
            '_Current_Status_Target_Minutes': current_status_target_minutes,
            '_Current_Trip_Target_km': current_trip_target_km,
            '_Public_Transport_Flag': latent_profiles['public_transport_flag'],
            '_Commercial_Service_Flag': latent_profiles['commercial_service_flag'],
        }
        polars_module = _get_polars_module()
        if polars_module is not None:
            try:
                return polars_module.DataFrame(fleet_data).to_pandas()
            except Exception:
                pass
        return pd.DataFrame(fleet_data)
    
    def maybe_update(self, force: bool = False, full_refresh: bool = False) -> None:
        """Update fleet telemetry data every UPDATE_INTERVAL_SECONDS or if forced."""
        with self.lock:
            now_mono = time.monotonic()
            if self._last_update_monotonic is None:
                self._last_update_monotonic = now_mono
            
            elapsed = now_mono - self._last_update_monotonic
            if not force and elapsed < self.update_interval_seconds:
                return
            
            self._update_live_telemetry()
            self._last_update_monotonic = now_mono
            self._snapshot_version += 1
    
    def _update_live_telemetry(self) -> None:
        """Update vehicle telemetry for all vehicles every cycle."""
        now = _current_local_time()
        now_ts = now.strftime('%Y-%m-%d %H:%M:%S')
        time_of_day = get_time_of_day(now.hour, now.minute, now.second)
        day_of_week = get_day_of_week(now)
        hour_of_day = _datetime_hour_fraction(now)
        fleet = self.fleet
        vehicle_count = len(fleet)
        if vehicle_count == 0:
            return

        weather_by_city = _weather_for_cycle(now, self.rng)
        interval_seconds = max(float(self.update_interval_seconds), 1.0)

        plates = fill_missing_plate_values(fleet['Plate']).to_numpy(copy=True)
        city_codes = fleet['City'].fillna('Harare').astype(str).to_numpy(copy=True)
        vehicle_types = fleet['Type'].fillna('Sedan').astype(str).to_numpy(copy=True)
        usage = fleet['Usage'].fillna('Private').astype(str).to_numpy(copy=True)
        prev_status = fleet['Status'].fillna('Parked').astype(str).to_numpy(copy=True)
        prev_direction = fleet['Direction'].fillna('North').astype(str).to_numpy(copy=True)
        road_types = fleet['Road_Type'].fillna('Urban').astype(str).to_numpy(copy=True)
        risk_profile = pd.to_numeric(fleet.get('_Risk_Profile', pd.Series(0.18, index=fleet.index)), errors='coerce').fillna(0.18).to_numpy(dtype=float, copy=True)
        driving_propensity = pd.to_numeric(fleet.get('_Driving_Propensity', pd.Series(0.18, index=fleet.index)), errors='coerce').fillna(0.18).to_numpy(dtype=float, copy=True)
        target_speed_ratio = pd.to_numeric(fleet.get('_Target_Speed_Ratio', pd.Series(0.74, index=fleet.index)), errors='coerce').fillna(0.74).to_numpy(dtype=float, copy=True)
        harsh_event_probability = pd.to_numeric(fleet.get('_Harsh_Event_Probability', pd.Series(0.01, index=fleet.index)), errors='coerce').fillna(0.01).to_numpy(dtype=float, copy=True)
        maintenance_risk = pd.to_numeric(fleet.get('_Maintenance_Risk', pd.Series(0.18, index=fleet.index)), errors='coerce').fillna(0.18).to_numpy(dtype=float, copy=True)
        daily_distance_target_km = pd.to_numeric(fleet.get('_Daily_Distance_Target_km', pd.Series(55.0, index=fleet.index)), errors='coerce').fillna(55.0).to_numpy(dtype=float, copy=True)
        drive_session_minutes = pd.to_numeric(fleet.get('_Drive_Session_Minutes', pd.Series(40.0, index=fleet.index)), errors='coerce').fillna(40.0).to_numpy(dtype=float, copy=True)
        park_session_minutes = pd.to_numeric(fleet.get('_Park_Session_Minutes', pd.Series(120.0, index=fleet.index)), errors='coerce').fillna(120.0).to_numpy(dtype=float, copy=True)
        peak_hour_bias = pd.to_numeric(fleet.get('_Peak_Hour_Bias', pd.Series(1.0, index=fleet.index)), errors='coerce').fillna(1.0).to_numpy(dtype=float, copy=True)
        late_night_bias = pd.to_numeric(fleet.get('_Late_Night_Bias', pd.Series(1.0, index=fleet.index)), errors='coerce').fillna(1.0).to_numpy(dtype=float, copy=True)
        current_status_target_minutes = pd.to_numeric(fleet.get('_Current_Status_Target_Minutes', pd.Series(60.0, index=fleet.index)), errors='coerce').fillna(60.0).to_numpy(dtype=float, copy=True)
        current_trip_target_km = pd.to_numeric(fleet.get('_Current_Trip_Target_km', pd.Series(18.0, index=fleet.index)), errors='coerce').fillna(18.0).to_numpy(dtype=float, copy=True)
        public_transport_flag = pd.to_numeric(fleet.get('_Public_Transport_Flag', pd.Series(0, index=fleet.index)), errors='coerce').fillna(0).to_numpy(dtype=int, copy=True).astype(bool)
        commercial_service_flag = pd.to_numeric(fleet.get('_Commercial_Service_Flag', pd.Series(0, index=fleet.index)), errors='coerce').fillna(0).to_numpy(dtype=int, copy=True).astype(bool)

        prev_speed = pd.to_numeric(fleet['Speed_kmh'], errors='coerce').fillna(0.0).to_numpy(dtype=float, copy=True)
        prev_trip_distance = pd.to_numeric(fleet['Trip_Distance_km'], errors='coerce').fillna(0.0).to_numpy(dtype=float, copy=True)
        prev_trip_duration = pd.to_numeric(fleet['Trip_Duration_Hour'], errors='coerce').fillna(0.0).to_numpy(dtype=float, copy=True)
        prev_rpm = pd.to_numeric(fleet['RPM'], errors='coerce').fillna(700.0).to_numpy(dtype=float, copy=True)
        prev_engine_load = pd.to_numeric(fleet['Engine_Load_pct'], errors='coerce').fillna(0.0).to_numpy(dtype=float, copy=True)
        prev_coolant = pd.to_numeric(fleet['Coolant_Temp_C'], errors='coerce').fillna(75.0).to_numpy(dtype=float, copy=True)
        prev_lat = pd.to_numeric(fleet['GPS_Latitude'], errors='coerce').fillna(CITY_COORDS['Harare'][0]).to_numpy(dtype=float, copy=True)
        prev_lon = pd.to_numeric(fleet['GPS_Longitude'], errors='coerce').fillna(CITY_COORDS['Harare'][1]).to_numpy(dtype=float, copy=True)
        harsh_brakes = pd.to_numeric(fleet['Harsh_Brake_Count'], errors='coerce').fillna(0).to_numpy(dtype=int, copy=True)
        harsh_accels = pd.to_numeric(fleet['Harsh_Accel_Count'], errors='coerce').fillna(0).to_numpy(dtype=int, copy=True)
        harsh_corners = pd.to_numeric(fleet['Harsh_Corner_Count'], errors='coerce').fillna(0).to_numpy(dtype=int, copy=True)
        fuel_efficiency = pd.to_numeric(fleet['Fuel_Efficiency_L_per_100km'], errors='coerce').fillna(8.5).to_numpy(dtype=float, copy=True)
        trip_ids = fleet['Trip_ID'].to_numpy(dtype=object, copy=True)
        status_change_timestamps = fleet['Status_Change_Timestamp'].fillna(now_ts).to_numpy(dtype=object, copy=True)

        weather = np.array([weather_by_city.get(city, 'Sunny') for city in city_codes], dtype=object)

        status_change_dt = pd.to_datetime(pd.Series(status_change_timestamps, index=fleet.index), errors='coerce').fillna(pd.Timestamp(now))
        elapsed_status_minutes = (
            (pd.Timestamp(now) - status_change_dt).dt.total_seconds().clip(lower=0.0) / 60.0
        ).to_numpy(dtype=float, copy=True)
        eligible_to_change = elapsed_status_minutes >= current_status_target_minutes

        desired_drive_prob = _calculate_desired_driving_probability(
            city_codes,
            driving_propensity,
            public_transport_flag,
            commercial_service_flag,
            time_of_day,
            peak_hour_bias,
            late_night_bias,
            hour_of_day=hour_of_day,
        )
        distance_progress = np.divide(
            prev_trip_distance,
            np.maximum(current_trip_target_km, 1.0),
            out=np.zeros(vehicle_count, dtype=float),
            where=current_trip_target_km > 0,
        )
        continue_drive_prob = np.clip(
            0.14
            + (0.72 * desired_drive_prob)
            + (0.34 * np.clip(1.0 - distance_progress, 0.0, 1.0))
            + np.clip((daily_distance_target_km - 50.0) / 300.0, 0.0, 0.14),
            0.04,
            0.985,
        )
        start_drive_prob = np.clip(
            0.02
            + (1.04 * desired_drive_prob)
            + np.clip((daily_distance_target_km - 50.0) / 320.0, 0.0, 0.10),
            0.02,
            0.965,
        )

        eligible_drive_mask = (prev_status == 'Driving') & eligible_to_change
        eligible_park_mask = (prev_status == 'Parked') & eligible_to_change
        new_status = prev_status.copy()
        stop_mask = eligible_drive_mask & (self.rng.random(vehicle_count) >= continue_drive_prob)
        start_mask = eligible_park_mask & (self.rng.random(vehicle_count) < start_drive_prob)
        new_status[stop_mask] = 'Parked'
        new_status[start_mask] = 'Driving'
        active_mask = new_status == 'Driving'

        transitioned_mask = prev_status != new_status
        status_change_timestamps[transitioned_mask] = now_ts

        decision_extended_mask = eligible_to_change & (~transitioned_mask)
        if np.any(decision_extended_mask):
            extension_minutes = _sample_status_target_minutes(
                new_status == 'Driving',
                drive_session_minutes,
                park_session_minutes,
                self.rng,
            )
            current_status_target_minutes[decision_extended_mask] = (
                elapsed_status_minutes[decision_extended_mask] + extension_minutes[decision_extended_mask]
            )
        if np.any(transitioned_mask):
            current_status_target_minutes[transitioned_mask] = _sample_status_target_minutes(
                new_status[transitioned_mask] == 'Driving',
                drive_session_minutes[transitioned_mask],
                park_session_minutes[transitioned_mask],
                self.rng,
            )

        new_trip_mask = (prev_status == 'Parked') & active_mask
        ended_trip_mask = (prev_status == 'Driving') & (~active_mask)
        if np.any(new_trip_mask):
            trip_stamp = now.strftime('%Y%m%d_%H%M%S')
            trip_ids[new_trip_mask] = [f"TRIP-{plate}-{trip_stamp}" for plate in plates[new_trip_mask]]
            current_trip_target_km[new_trip_mask] = _sample_trip_distance_targets(
                daily_distance_target_km[new_trip_mask],
                public_transport_flag[new_trip_mask],
                commercial_service_flag[new_trip_mask],
                self.rng,
            )
        if np.any(ended_trip_mask):
            current_trip_target_km[ended_trip_mask] = 0.0

        trip_distance = prev_trip_distance.copy()
        trip_duration = prev_trip_duration.copy()
        trip_distance[new_trip_mask] = 0.0
        trip_duration[new_trip_mask] = 0.0

        road_types = _sample_road_types(city_codes, usage, vehicle_types, time_of_day, active_mask, self.rng, road_types)
        directions = _sample_directions(prev_direction, active_mask, self.rng)

        road_min, road_max = _road_limits_for_types(road_types)
        weather_speed_factor = _weather_speed_adjustment(weather)
        if time_of_day in ['Morning Rush', 'Evening Rush']:
            congestion_pressure = max(0.0, _hourly_activity_intensity(hour_of_day) - 0.70)
            time_speed_factor = float(np.clip(0.98 - (0.10 * congestion_pressure), 0.90, 0.98))
        elif time_of_day in ['Night', 'Late Night']:
            quiet_road_bonus = max(0.0, 0.42 - _hourly_activity_intensity(hour_of_day))
            time_speed_factor = float(np.clip(0.97 + (0.05 * quiet_road_bonus), 0.97, 1.02))
        else:
            time_speed_factor = 1.0
        allowed_excess = np.where(
            public_transport_flag,
            np.where(risk_profile > 0.58, 10.0, 5.0),
            np.where(commercial_service_flag, np.where(risk_profile > 0.64, 7.0, 4.0), np.where(risk_profile > 0.78, 5.0, 2.0)),
        )
        target_ratio = np.clip(target_speed_ratio * weather_speed_factor * time_speed_factor, 0.52, 1.08)
        speed_target = np.clip(road_max * target_ratio, road_min * 0.65, road_max + allowed_excess)
        speed_kmh = np.zeros(vehicle_count, dtype=float)
        if np.any(active_mask):
            speed_noise = self.rng.normal(0.0, 0.6 + (1.0 * risk_profile[active_mask]), int(active_mask.sum()))
            speed_kmh[active_mask] = np.clip(
                prev_speed[active_mask] + 0.12 * (speed_target[active_mask] - prev_speed[active_mask]) + speed_noise,
                road_min[active_mask] * 0.60,
                road_max[active_mask] + allowed_excess[active_mask],
            )
        if np.any(new_trip_mask):
            start_speed = speed_target[new_trip_mask] * self.rng.normal(0.72, 0.06, int(new_trip_mask.sum()))
            speed_kmh[new_trip_mask] = np.clip(
                start_speed,
                road_min[new_trip_mask] * 0.55,
                road_max[new_trip_mask] + allowed_excess[new_trip_mask],
            )
        speed_kmh = np.round(speed_kmh, 1)

        acceleration = np.zeros(vehicle_count, dtype=float)
        if np.any(active_mask):
            delta_speed_mps = (speed_kmh[active_mask] - prev_speed[active_mask]) / 3.6
            active_acceleration = 0.55 * (delta_speed_mps / interval_seconds) + self.rng.normal(
                0.0,
                0.18 + (0.35 * risk_profile[active_mask]),
                int(active_mask.sum()),
            )
            spike_probability = harsh_event_probability[active_mask] * np.where(public_transport_flag[active_mask], 0.45, 0.30)
            spike_mask = self.rng.random(int(active_mask.sum())) < spike_probability
            if np.any(spike_mask):
                spike_values = self.rng.choice(np.array([-1.8, -1.5, 1.5, 1.9]), size=int(spike_mask.sum()))
                active_acceleration[spike_mask] += spike_values
            acceleration[active_mask] = np.clip(active_acceleration, -4.2, 4.2)

        throttle_pct = np.zeros(vehicle_count, dtype=float)
        rpm = np.maximum(700.0, prev_rpm - 40.0)
        engine_load = np.maximum(5.0, prev_engine_load - 1.2)
        coolant_temp = np.clip(prev_coolant - 0.2, 68.0, 104.0)
        maf_gs = np.full(vehicle_count, 1.0, dtype=float)

        if np.any(active_mask):
            speed_share = speed_kmh[active_mask] / np.maximum(road_max[active_mask], 1.0)
            throttle_pct[active_mask] = np.clip(
                16.0 + (speed_share * 26.0) + (np.abs(acceleration[active_mask]) * 10.0) + (risk_profile[active_mask] * 8.0),
                5.0,
                78.0,
            )
            rpm[active_mask] = np.clip(
                820.0 + (speed_share * 2500.0) + (np.abs(acceleration[active_mask]) * 240.0) + self.rng.normal(0.0, 80.0, int(active_mask.sum())),
                700.0,
                4300.0,
            )
            engine_load[active_mask] = np.clip(
                18.0 + (speed_share * 48.0) + (maintenance_risk[active_mask] * 8.0) + self.rng.normal(0.0, 1.2, int(active_mask.sum())),
                8.0,
                88.0,
            )
            maf_gs[active_mask] = np.clip(1.3 + (engine_load[active_mask] * 0.05) + self.rng.normal(0.0, 0.12, int(active_mask.sum())), 1.0, 7.5)
            coolant_delta = ((engine_load[active_mask] - 35.0) * 0.04) + (maintenance_risk[active_mask] * 0.6)
            coolant_temp[active_mask] = np.clip(
                prev_coolant[active_mask] + coolant_delta + self.rng.normal(0.0, 0.3, int(active_mask.sum())),
                70.0,
                103.0,
            )

        if np.any(active_mask):
            active_indices = np.flatnonzero(active_mask)
            accel_events = self.rng.random(len(active_indices)) < (
                harsh_event_probability[active_indices] * np.where(public_transport_flag[active_indices], 0.22, 0.16)
            )
            brake_events = self.rng.random(len(active_indices)) < (
                harsh_event_probability[active_indices] * np.where(public_transport_flag[active_indices], 0.18, 0.13)
            )
            corner_multiplier = np.where(np.isin(road_types[active_indices], ['Urban', 'Local Streets', 'Potholed']), 0.20, 0.10)
            corner_events = self.rng.random(len(active_indices)) < (harsh_event_probability[active_indices] * corner_multiplier)
            harsh_accels[active_indices[accel_events]] += 1
            harsh_brakes[active_indices[brake_events]] += 1
            harsh_corners[active_indices[corner_events]] += 1

        cooldown_mask = (~active_mask) & (self.rng.random(vehicle_count) < 0.03)
        harsh_brakes[cooldown_mask] = np.maximum(harsh_brakes[cooldown_mask] - 1, 0)
        harsh_accels[cooldown_mask] = np.maximum(harsh_accels[cooldown_mask] - 1, 0)
        harsh_corners[cooldown_mask] = np.maximum(harsh_corners[cooldown_mask] - 1, 0)
        harsh_brakes, harsh_accels, harsh_corners = _cap_harsh_event_totals(harsh_brakes, harsh_accels, harsh_corners, vehicle_types)

        speeding_flag = np.zeros(vehicle_count, dtype=int)
        speeding_flag[active_mask] = (speed_kmh[active_mask] > (road_max[active_mask] + 2.5)).astype(int)

        if np.any(active_mask):
            trip_distance[active_mask] += (speed_kmh[active_mask] / 60.0) * (self.update_interval_seconds / 60.0)
            trip_duration[active_mask] += self.update_interval_seconds / 3600.0

            base_efficiency = _base_fuel_efficiency(vehicle_types[active_mask], maintenance_risk[active_mask], self.rng)
            speed_factor = np.where((speed_kmh[active_mask] >= 35.0) & (speed_kmh[active_mask] <= 95.0), 1.0, 1.08)
            aggressive_factor = np.where(np.abs(acceleration[active_mask]) > 2.2, 1.06, 1.0)
            fuel_efficiency[active_mask] = np.clip(base_efficiency * speed_factor * aggressive_factor, 4.8, 34.0)

        battery_voltage = np.clip(12.70 - (maintenance_risk * 0.45) - (np.abs(acceleration) * 0.02) - (speeding_flag * 0.03), 11.8, 12.8)
        total_harsh_events = harsh_brakes + harsh_accels + harsh_corners
        gps_latitudes = prev_lat.copy()
        gps_longitudes = prev_lon.copy()
        if np.any(active_mask):
            moved_lat, moved_lon = _move_gps_batch(
                prev_lat[active_mask],
                prev_lon[active_mask],
                directions[active_mask],
                speed_kmh[active_mask],
                self.update_interval_seconds,
                self.rng,
            )
            gps_latitudes[active_mask] = moved_lat
            gps_longitudes[active_mask] = moved_lon

        fleet['Time_of_Day'] = time_of_day
        fleet['Day_of_Week'] = day_of_week
        fleet['Last_Update'] = now_ts
        fleet['Weather'] = weather
        fleet['Status'] = new_status
        fleet['Status_Change_Timestamp'] = status_change_timestamps
        fleet['Trip_ID'] = trip_ids
        fleet['Trip_Distance_km'] = trip_distance
        fleet['Trip_Duration_Hour'] = trip_duration
        fleet['Road_Type'] = road_types
        fleet['Direction'] = directions
        fleet['GPS_Latitude'] = gps_latitudes
        fleet['GPS_Longitude'] = gps_longitudes
        fleet['Speed_kmh'] = speed_kmh
        fleet['Acceleration_mps2'] = acceleration
        fleet['Throttle_pct'] = throttle_pct
        fleet['RPM'] = rpm
        fleet['Engine_Load_pct'] = engine_load
        fleet['MAF_gs'] = maf_gs
        fleet['Coolant_Temp_C'] = coolant_temp
        fleet['Harsh_Brake_Count'] = harsh_brakes
        fleet['Harsh_Accel_Count'] = harsh_accels
        fleet['Harsh_Corner_Count'] = harsh_corners
        fleet['Total_Harsh_Events_per_Day'] = total_harsh_events
        fleet['Speeding_Flag'] = speeding_flag
        fleet['Battery_V'] = np.round(battery_voltage, 2)
        fleet['Fuel_Efficiency_L_per_100km'] = np.round(fuel_efficiency, 2)
        fleet['_Current_Status_Target_Minutes'] = current_status_target_minutes
        fleet['_Current_Trip_Target_km'] = current_trip_target_km

    
    def get_fleet_snapshot(self) -> pd.DataFrame:
        """Return current fleet snapshot."""
        with self.lock:
            public_columns = [column for column in self.fleet.columns if not str(column).startswith('_')]
            return self.fleet.loc[:, public_columns].copy()

    def get_fleet_snapshot_with_version(self) -> tuple[int, pd.DataFrame]:
        """Return the current public fleet snapshot together with its update version."""
        with self.lock:
            public_columns = [column for column in self.fleet.columns if not str(column).startswith('_')]
            return int(self._snapshot_version), self.fleet.loc[:, public_columns].copy()

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL FLEET INSTANCE & API FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

_fleet_manager = None
_LIVE_SNAPSHOT_CACHE_LOCK = threading.Lock()
_LIVE_SNAPSHOT_CACHE: dict[str, Any] = {
    "version": None,
    "raw_df": None,
    "live_df": None,
}

def _get_fleet_manager() -> FleetManager:
    global _fleet_manager
    if _fleet_manager is None:
        _fleet_manager = FleetManager(num_vehicles=NUM_VEHICLES, update_interval_seconds=UPDATE_INTERVAL_SECONDS)
    return _fleet_manager


def _get_cached_live_snapshot_pair(force_update: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cached raw/preprocessed live snapshots until the fleet version changes."""
    fm = _get_fleet_manager()
    fm.maybe_update(force=force_update, full_refresh=False)
    snapshot_version, public_snapshot = fm.get_fleet_snapshot_with_version()

    with _LIVE_SNAPSHOT_CACHE_LOCK:
        if (
            _LIVE_SNAPSHOT_CACHE.get("version") == snapshot_version
            and isinstance(_LIVE_SNAPSHOT_CACHE.get("raw_df"), pd.DataFrame)
            and isinstance(_LIVE_SNAPSHOT_CACHE.get("live_df"), pd.DataFrame)
        ):
            return (
                _LIVE_SNAPSHOT_CACHE["raw_df"].copy(),
                _LIVE_SNAPSHOT_CACHE["live_df"].copy(),
            )

    raw_df = _ensure_live_columns_populated(public_snapshot, mode='raw')
    live_df = _generate_preprocessed_telematics_dataset(raw_df)

    with _LIVE_SNAPSHOT_CACHE_LOCK:
        _LIVE_SNAPSHOT_CACHE["version"] = snapshot_version
        _LIVE_SNAPSHOT_CACHE["raw_df"] = raw_df.copy()
        _LIVE_SNAPSHOT_CACHE["live_df"] = live_df.copy()

    return raw_df, live_df

def get_raw_dataset(num_sample: Optional[int] = None) -> pd.DataFrame:
    """Get raw telemetry dataset with forced live update."""
    df, _ = _get_cached_live_snapshot_pair(force_update=False)
    if num_sample and num_sample < len(df):
        return df.sample(n=num_sample, random_state=None).reset_index(drop=True)
    return df

def get_preprocessed_dataset() -> pd.DataFrame:
    """Get preprocessed telemetry dataset with engineered features."""
    _, live_df = _get_cached_live_snapshot_pair(force_update=False)
    return live_df

def get_live_data_snapshot() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get live raw and preprocessed snapshots. Returns (raw_df, live_df)."""
    return _get_cached_live_snapshot_pair(force_update=False)

def get_live_preprocessed_data() -> pd.DataFrame:
    """Get live preprocessed data with engineering."""
    _, live_df = _get_cached_live_snapshot_pair(force_update=False)
    return live_df


def capture_telematics_history(
    cycles: int = 12,
    sleep_seconds: float = 0.0,
    num_sample: Optional[int] = None,
) -> pd.DataFrame:
    """Capture a short telemetry event history for rolling-statistics analysis."""
    fm = _get_fleet_manager()
    snapshots: list[pd.DataFrame] = []
    for cycle_index in range(max(1, int(cycles))):
        fm.maybe_update(force=True, full_refresh=False)
        snapshot = _ensure_live_columns_populated(fm.get_fleet_snapshot(), mode='raw')
        if num_sample and num_sample < len(snapshot):
            snapshot = snapshot.sample(n=num_sample, random_state=None).reset_index(drop=True)
        snapshot = snapshot.copy()
        snapshot['Snapshot_Index'] = cycle_index
        snapshot['Snapshot_Captured_At'] = _current_local_time().strftime('%Y-%m-%d %H:%M:%S')
        snapshots.append(snapshot)
        if sleep_seconds > 0 and cycle_index < int(cycles) - 1:
            time.sleep(sleep_seconds)
    return pd.concat(snapshots, axis=0, ignore_index=True) if snapshots else pd.DataFrame()


@dataclass(frozen=True)
class TelemetryAggregationConfig:
    driver_id_col: str = "Plate"
    timestamp_col: str = "Last_Update"
    lookback_rows: Optional[int] = 120
    max_workers: int = max(2, min(os.cpu_count() or 4, 8))


def _aggregation_safe_numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    return (
        pd.to_numeric(frame.get(column, pd.Series(default, index=frame.index)), errors="coerce")
        .fillna(default)
        .astype(float)
    )


def _prepare_driver_statistics_history(
    history_df: pd.DataFrame,
    config: TelemetryAggregationConfig,
) -> pd.DataFrame:
    if history_df is None or history_df.empty:
        return pd.DataFrame(columns=[config.driver_id_col, config.timestamp_col])

    prepared = history_df.copy()
    if config.driver_id_col not in prepared.columns:
        raise KeyError(f"Missing driver identifier column: {config.driver_id_col}")

    prepared[config.driver_id_col] = prepared[config.driver_id_col].astype(str)
    prepared[config.timestamp_col] = pd.to_datetime(prepared.get(config.timestamp_col), errors="coerce")
    prepared = prepared.dropna(subset=[config.timestamp_col]).sort_values(
        [config.driver_id_col, config.timestamp_col]
    ).reset_index(drop=True)

    if config.lookback_rows is not None:
        prepared = (
            prepared.groupby(config.driver_id_col, group_keys=False)
            .tail(int(config.lookback_rows))
            .reset_index(drop=True)
        )

    harsh_total = (
        _aggregation_safe_numeric(prepared, "Harsh_Brake_Count")
        + _aggregation_safe_numeric(prepared, "Harsh_Accel_Count")
        + _aggregation_safe_numeric(prepared, "Harsh_Corner_Count")
    )
    if "Recent_Harsh_Events" in prepared.columns:
        harsh_total = np.maximum(
            harsh_total.to_numpy(dtype=float),
            _aggregation_safe_numeric(prepared, "Recent_Harsh_Events").to_numpy(dtype=float),
        )

    prepared["_harsh_total"] = harsh_total.astype(float)
    prepared["_trip_distance_total"] = _aggregation_safe_numeric(prepared, "Trip_Distance_km")
    prepared["_night_flag"] = pd.to_numeric(
        prepared.get(
            "Night_Driving_Flag",
            prepared.get("Time_of_Day", pd.Series("", index=prepared.index)).isin(
                ["Night", "Late Night", "Dawn"]
            ).astype(int),
        ),
        errors="coerce",
    ).fillna(0).astype(int)
    prepared["_speeding_flag"] = pd.to_numeric(
        prepared.get("Speeding_Flag", pd.Series(0, index=prepared.index)),
        errors="coerce",
    ).fillna(0).astype(int)
    prepared["_speed_kmh"] = _aggregation_safe_numeric(prepared, "Speed_kmh")
    prepared["_acceleration_mps2"] = _aggregation_safe_numeric(prepared, "Acceleration_mps2")
    prepared["_engine_load_pct"] = _aggregation_safe_numeric(prepared, "Engine_Load_pct")
    prepared["_coolant_temp_c"] = _aggregation_safe_numeric(prepared, "Coolant_Temp_C")

    prepared["_harsh_increment"] = (
        prepared.groupby(config.driver_id_col)["_harsh_total"]
        .diff()
        .clip(lower=0)
        .fillna(prepared["_harsh_total"])
    )
    prepared["_distance_increment"] = (
        prepared.groupby(config.driver_id_col)["_trip_distance_total"]
        .diff()
        .clip(lower=0)
        .fillna(prepared["_trip_distance_total"])
    )
    prepared["_night_distance_increment"] = prepared["_distance_increment"] * prepared["_night_flag"].astype(float)
    return prepared


def _aggregate_driver_statistics_partition(
    frame: pd.DataFrame,
    config: TelemetryAggregationConfig,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    group = frame.groupby(config.driver_id_col, dropna=False)
    aggregated = group.agg(
        observation_count=(config.driver_id_col, "size"),
        window_start=(config.timestamp_col, "min"),
        window_end=(config.timestamp_col, "max"),
        latest_city=("City", "last"),
        latest_status=("Status", "last"),
        avg_speed_kmh=("_speed_kmh", "mean"),
        speed_std_kmh=("_speed_kmh", "std"),
        max_speed_kmh=("_speed_kmh", "max"),
        avg_acceleration_mps2=("_acceleration_mps2", "mean"),
        harsh_event_total=("_harsh_increment", "sum"),
        latest_harsh_event_total=("_harsh_total", "max"),
        speeding_event_rate=("_speeding_flag", "mean"),
        night_distance_km=("_night_distance_increment", "sum"),
        trip_distance_km=("_distance_increment", "sum"),
        avg_engine_load_pct=("_engine_load_pct", "mean"),
        max_coolant_temp_c=("_coolant_temp_c", "max"),
    ).reset_index()

    aggregated["speed_std_kmh"] = aggregated["speed_std_kmh"].fillna(0.0)
    numeric_columns = [
        "avg_speed_kmh",
        "speed_std_kmh",
        "max_speed_kmh",
        "avg_acceleration_mps2",
        "harsh_event_total",
        "latest_harsh_event_total",
        "speeding_event_rate",
        "night_distance_km",
        "trip_distance_km",
        "avg_engine_load_pct",
        "max_coolant_temp_c",
    ]
    aggregated[numeric_columns] = aggregated[numeric_columns].round(4)
    return aggregated


def compute_driver_statistics(
    history_df: pd.DataFrame,
    config: Optional[TelemetryAggregationConfig] = None,
) -> pd.DataFrame:
    cfg = config or TelemetryAggregationConfig()
    prepared = _prepare_driver_statistics_history(history_df, cfg)
    return _aggregate_driver_statistics_partition(prepared, cfg)


def compute_driver_statistics_parallel(
    history_df: pd.DataFrame,
    config: Optional[TelemetryAggregationConfig] = None,
) -> pd.DataFrame:
    cfg = config or TelemetryAggregationConfig()
    prepared = _prepare_driver_statistics_history(history_df, cfg)
    if prepared.empty:
        return pd.DataFrame()

    driver_ids = prepared[cfg.driver_id_col].drop_duplicates().to_numpy()
    if len(driver_ids) <= 1:
        return _aggregate_driver_statistics_partition(prepared, cfg)

    worker_count = max(1, min(cfg.max_workers, len(driver_ids)))
    driver_slices = [driver_slice for driver_slice in np.array_split(driver_ids, worker_count) if len(driver_slice) > 0]
    partitions = [
        prepared[prepared[cfg.driver_id_col].isin(driver_slice)].copy()
        for driver_slice in driver_slices
    ]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(lambda frame: _aggregate_driver_statistics_partition(frame, cfg), partitions))

    aggregated = pd.concat(results, axis=0, ignore_index=True)
    return aggregated.sort_values(cfg.driver_id_col).reset_index(drop=True)


def compute_driver_statistics_spark(spark_df, driver_id_col: str = "Plate", timestamp_col: str = "Last_Update"):
    from pyspark.sql import functions as F
    from pyspark.sql import window as W

    window = W.Window.partitionBy(driver_id_col).orderBy(F.to_timestamp(F.col(timestamp_col)))
    harsh_total = (
        F.coalesce(F.col("Harsh_Brake_Count").cast("double"), F.lit(0.0))
        + F.coalesce(F.col("Harsh_Accel_Count").cast("double"), F.lit(0.0))
        + F.coalesce(F.col("Harsh_Corner_Count").cast("double"), F.lit(0.0))
    )
    distance_total = F.coalesce(F.col("Trip_Distance_km").cast("double"), F.lit(0.0))
    night_flag = F.coalesce(F.col("Night_Driving_Flag").cast("int"), F.lit(0))

    prepared = (
        spark_df
        .withColumn("_event_ts", F.to_timestamp(F.col(timestamp_col)))
        .withColumn("_harsh_total", harsh_total)
        .withColumn("_trip_distance_total", distance_total)
        .withColumn(
            "_harsh_increment",
            F.greatest(F.lit(0.0), F.col("_harsh_total") - F.coalesce(F.lag("_harsh_total").over(window), F.lit(0.0))),
        )
        .withColumn(
            "_distance_increment",
            F.greatest(
                F.lit(0.0),
                F.col("_trip_distance_total") - F.coalesce(F.lag("_trip_distance_total").over(window), F.lit(0.0)),
            ),
        )
        .withColumn("_night_distance_increment", F.col("_distance_increment") * night_flag.cast("double"))
        .withColumn("_speed_kmh", F.coalesce(F.col("Speed_kmh").cast("double"), F.lit(0.0)))
        .withColumn("_acceleration_mps2", F.coalesce(F.col("Acceleration_mps2").cast("double"), F.lit(0.0)))
        .withColumn("_engine_load_pct", F.coalesce(F.col("Engine_Load_pct").cast("double"), F.lit(0.0)))
        .withColumn("_coolant_temp_c", F.coalesce(F.col("Coolant_Temp_C").cast("double"), F.lit(0.0)))
        .withColumn("_speeding_flag", F.coalesce(F.col("Speeding_Flag").cast("double"), F.lit(0.0)))
    )

    return prepared.groupBy(driver_id_col).agg(
        F.count(F.lit(1)).alias("observation_count"),
        F.min("_event_ts").alias("window_start"),
        F.max("_event_ts").alias("window_end"),
        F.last("City", ignorenulls=True).alias("latest_city"),
        F.last("Status", ignorenulls=True).alias("latest_status"),
        F.avg("_speed_kmh").alias("avg_speed_kmh"),
        F.stddev_pop("_speed_kmh").alias("speed_std_kmh"),
        F.max("_speed_kmh").alias("max_speed_kmh"),
        F.avg("_acceleration_mps2").alias("avg_acceleration_mps2"),
        F.sum("_harsh_increment").alias("harsh_event_total"),
        F.max("_harsh_total").alias("latest_harsh_event_total"),
        F.avg("_speeding_flag").alias("speeding_event_rate"),
        F.sum("_night_distance_increment").alias("night_distance_km"),
        F.sum("_distance_increment").alias("trip_distance_km"),
        F.avg("_engine_load_pct").alias("avg_engine_load_pct"),
        F.max("_coolant_temp_c").alias("max_coolant_temp_c"),
    )


def get_parallel_driver_statistics(
    history_df: Optional[pd.DataFrame] = None,
    cycles: int = 12,
    sleep_seconds: float = 0.0,
    num_sample: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    """Return scalable rolling driver statistics from captured or supplied telemetry history."""
    history = history_df if history_df is not None else capture_telematics_history(
        cycles=cycles,
        sleep_seconds=sleep_seconds,
        num_sample=num_sample,
    )
    default_workers = TelemetryAggregationConfig().max_workers
    config = TelemetryAggregationConfig(max_workers=max_workers if max_workers is not None else default_workers)
    return compute_driver_statistics_parallel(history, config=config)


def get_parallel_driver_statistics_spark(history_df: Optional[pd.DataFrame] = None, spark_session: Any = None):
    """Return rolling driver statistics as a Spark DataFrame for distributed batch workloads."""
    history = history_df if history_df is not None else capture_telematics_history()
    spark = spark_session or _get_spark_session()
    spark_df = spark.createDataFrame(history)
    return compute_driver_statistics_spark(spark_df)

def get_fleet_manager() -> FleetManager:
    """Get the global fleet manager instance."""
    return _get_fleet_manager()


def get_raw_dataset_polars(num_sample: Optional[int] = None):
    """Return the raw telemetry snapshot as a Polars DataFrame."""
    polars_module = _get_polars_module()
    if polars_module is None:
        raise ImportError("Polars is not installed. Install `polars` to use the Polars adapter.")
    return polars_module.from_pandas(get_raw_dataset(num_sample=num_sample))


def get_preprocessed_dataset_polars():
    """Return the engineered telemetry dataset as a Polars DataFrame."""
    polars_module = _get_polars_module()
    if polars_module is None:
        raise ImportError("Polars is not installed. Install `polars` to use the Polars adapter.")
    return polars_module.from_pandas(get_preprocessed_dataset())


def get_live_data_snapshot_polars():
    """Return the live raw and engineered snapshots as Polars DataFrames."""
    polars_module = _get_polars_module()
    if polars_module is None:
        raise ImportError("Polars is not installed. Install `polars` to use the Polars adapter.")
    raw_df, live_df = get_live_data_snapshot()
    return polars_module.from_pandas(raw_df), polars_module.from_pandas(live_df)


def get_raw_dataset_dask(num_sample: Optional[int] = None, npartitions: int = 4):
    """Return the raw telemetry snapshot as a Dask DataFrame for offline scaling."""
    try:
        import dask.dataframe as dd
    except ImportError as exc:
        raise ImportError("Dask is not installed. Install `dask[dataframe]` to use the Dask adapter.") from exc

    raw_df = get_raw_dataset(num_sample=num_sample)
    return dd.from_pandas(raw_df, npartitions=max(1, npartitions))


def get_preprocessed_dataset_dask(npartitions: int = 4):
    """Return the engineered telemetry dataset as a Dask DataFrame for offline scaling."""
    try:
        import dask.dataframe as dd
    except ImportError as exc:
        raise ImportError("Dask is not installed. Install `dask[dataframe]` to use the Dask adapter.") from exc

    preprocessed_df = get_preprocessed_dataset()
    return dd.from_pandas(preprocessed_df, npartitions=max(1, npartitions))


_spark_session = None


def _get_spark_session():
    global _spark_session
    if _spark_session is None:
        try:
            from pyspark.sql import SparkSession
        except ImportError as exc:
            raise ImportError("PySpark is not installed. Install `pyspark` to use the Spark adapter.") from exc
        worker_count = max(2, min(os.cpu_count() or 4, 8))
        _spark_session = (
            SparkSession.builder.master("local[*]")
            .appName("InsurtechTelematics")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
            .config("spark.sql.shuffle.partitions", str(worker_count))
            .config("spark.default.parallelism", str(worker_count))
            .config("spark.driver.memory", "2g")
            .getOrCreate()
        )
        _spark_session.sparkContext.setLogLevel("ERROR")
    return _spark_session


def get_raw_dataset_spark(num_sample: Optional[int] = None, spark_session: Any = None):
    """Return the raw telemetry snapshot as a Spark DataFrame for batch analytics."""
    raw_df = get_raw_dataset(num_sample=num_sample)
    spark = spark_session or _get_spark_session()
    return spark.createDataFrame(raw_df)


def get_preprocessed_dataset_spark(spark_session: Any = None):
    """Return the engineered telemetry dataset as a Spark DataFrame for batch analytics."""
    preprocessed_df = get_preprocessed_dataset()
    spark = spark_session or _get_spark_session()
    return spark.createDataFrame(preprocessed_df)

def _generate_preprocessed_telematics_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Generate preprocessed dataset with engineered features."""
    df = raw_df.copy()
    
    # Ensure Trip_ID exists
    try:
        ts = _current_local_time().strftime('%H%M%S')
        if 'Trip_ID' not in df.columns:
            df['Trip_ID'] = None
        mask = df['Trip_ID'].isna()
        if mask.any():
            df.loc[mask, 'Trip_ID'] = df.loc[mask, 'Plate'].astype(str).apply(lambda p: f"IDLE-{p}-{ts}")
    except Exception:
        pass
    
    for col in LIVE_PREPROCESSED_BASE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    
    df['Night_Driving_Flag'] = df['Time_of_Day'].isin(['Night', 'Late Night', 'Dawn']).astype(int)
    df['Vehicle_Age_Years'] = (2026 - df['Year'].fillna(2026).astype(float)).clip(lower=0.0)
    df['Potholed_Ratio'] = (df['Road_Type'] == 'Potholed').astype(float)
    df['Gravel_Ratio'] = (df['Road_Type'] == 'Gravel').astype(float)
    df['Highway_Ratio'] = (df['Road_Type'] == 'Highway').astype(float)
    df['Urban_Ratio'] = (df['Road_Type'] == 'Urban').astype(float)
    
    road_max_map = {road: limits[1] for road, limits in ROAD_LIMITS.items()}
    df['Base_Road_Max_Speed_kmh'] = df['Road_Type'].map(road_max_map).fillna(80.0).astype(float)
    df['Speeding_Excess_kmh'] = np.clip(df['Speed_kmh'].fillna(0.0).astype(float) - df['Base_Road_Max_Speed_kmh'], 0.0, None)
    
    df['Recent_Harsh_Events'] = df[['Harsh_Brake_Count', 'Harsh_Accel_Count', 'Harsh_Corner_Count']].fillna(0).astype(float).sum(axis=1).clip(0.0, 10.0)
    df['Recent_Avg_Speed'] = np.clip(df['Speed_kmh'].fillna(0.0).astype(float) * 0.95, 0.0, 180.0)
    df['Recent_Speeding_Ratio'] = np.where(df['Trip_Distance_km'].fillna(0.0).astype(float) > 0, df['Speeding_Flag'].fillna(0).astype(float) / (df['Trip_Distance_km'].fillna(0.0).astype(float) / 100.0), 0.0)
    df['Recent_Night_Distance'] = np.where(df['Night_Driving_Flag'] == 1, df['Trip_Distance_km'].fillna(0.0).astype(float) * 0.45, df['Trip_Distance_km'].fillna(0.0).astype(float) * 0.08)
    
    df['Avg_Speed_Last_7_Days'] = df['Recent_Avg_Speed'].copy()
    df['Total_Harsh_Events_Last_30_Days'] = np.round(df['Recent_Harsh_Events'].fillna(0.0).astype(float) * 30.0, 1)
    
    speeding_ratio = np.where(df['Base_Road_Max_Speed_kmh'].fillna(1.0).astype(float) > 0, np.clip(df['Speeding_Excess_kmh'].fillna(0.0).astype(float) / df['Base_Road_Max_Speed_kmh'].fillna(1.0).astype(float), 0.0, 1.0), 0.0)
    night_ratio = df['Night_Driving_Flag'].fillna(0).astype(float)
    df['Speeding_Ratio_x_Night_Driving_Ratio'] = speeding_ratio * night_ratio
    
    weather_risk_map = {
        'Sunny': 0.8, 'Partly Cloudy': 0.9, 'Hazy': 1.0, 'Light Rain': 1.1,
        'Thundery / Heavy Rain': 1.3, 'Foggy': 1.3, 'Clear': 0.8, 'Hot': 0.9,
        'Hot / Scorching': 1.1, 'Cold': 0.8, 'Cold / Freezing': 0.7, 'Mild': 0.85,
        'Mild / Balmy': 0.85, 'Windy': 1.0, 'Windy / Gusty': 1.15, 'Misty': 1.2, 'Misty / Foggy': 1.25,
    }
    
    df['Battery_Health_Score'] = np.clip((df['Battery_V'].fillna(12.6).astype(float) - 11.5) * 50.0, 0.0, 100.0)
    df['Driving_Event_Score'] = np.clip(0.25 * (df['Recent_Harsh_Events'] / 5.0) + 0.25 * df['Speeding_Flag'].fillna(0).astype(float), 0.0, 1.0)
    df['Weather_Risk_Score'] = df['Weather'].map(weather_risk_map).fillna(1.0).astype(float)
    df['Aggressive_Driving_Score'] = np.clip(0.35 * (df['Recent_Harsh_Events'] / 5.0) + 0.35 * df['Speeding_Flag'].fillna(0).astype(float) + 0.30 * np.clip(np.abs(df['Acceleration_mps2'].fillna(0.0).astype(float)) / 6.0, 0.0, 1.0), 0.0, 1.0)
    df['Fatigue_Risk_Score'] = np.clip(0.4 * df['Night_Driving_Flag'] + 0.2 * np.minimum(df['Trip_Duration_Hour'].fillna(0.0).astype(float) / 5.0, 1.0) + 0.2 * df['Day_of_Week'].isin(['Saturday', 'Sunday']).astype(float), 0.0, 1.0)
    
    df['Coolant_Overheat_Flag'] = (df['Coolant_Temp_C'].fillna(0.0).astype(float) > 95.0).astype(int)
    
    # Engineered columns
    rng = np.random.default_rng(123)
    df['Max_Speed_kmh'] = np.maximum(df['Speed_kmh'].fillna(0.0).astype(float), df['Speed_kmh'].fillna(0.0).astype(float) + np.clip(rng.normal(1.2, 1.0, len(df)), 0.0, 8.0))
    df['Speed_Variance'] = np.clip(np.abs(rng.normal(2.0, 1.0, len(df))), 0.0, 20.0)
    df['Max_Acceleration_mps2'] = np.clip(np.abs(df['Acceleration_mps2'].fillna(0.0).astype(float)) + rng.uniform(0.2, 1.2, len(df)), 0.0, 6.0)
    df['Max_Deceleration_mps2'] = np.clip(np.abs(df['Acceleration_mps2'].fillna(0.0).astype(float)) * 0.6 + rng.uniform(0.1, 0.9, len(df)), 0.0, 6.0)
    
    # Time parsing
    now = pd.Timestamp(_current_local_time())
    last_update_dt = pd.to_datetime(df['Last_Update'], errors='coerce').fillna(now)
    df['Update_Year'] = last_update_dt.dt.year.astype(int)
    df['Update_Month'] = last_update_dt.dt.month.astype(int)
    df['Update_Day'] = last_update_dt.dt.day.astype(int)
    df['Update_Hour'] = last_update_dt.dt.hour.astype(int)
    df['Update_Minute'] = last_update_dt.dt.minute.astype(int)
    
    for col in LIVE_PREPROCESSED_ENGINEERED_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    
    df = df.drop(columns=["Harsh_Events_Per_Km", "Calculated_Risk_Score", "Risk_Band", "Monthly_Premium_USD"], errors='ignore')
    df = _ensure_live_columns_populated(df, mode='preprocessed')
    
    return df

LIVE_PREPROCESSED_BASE_COLUMNS = LIVE_RAW_COLUMNS.copy()

# ──────────────────────────────────────────────────────────────────────────────
# ADDITIONAL ANALYTICS & PRICING FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def update_preprocessed_dataset_live_columns(
    df: pd.DataFrame,
    live_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Overlay the latest live-backed telemetry columns onto an existing dataset."""
    if df is None or df.empty:
        return df

    result = df.copy()
    source_frame = live_df.copy() if isinstance(live_df, pd.DataFrame) and not live_df.empty else get_live_preprocessed_data()
    if source_frame is None or source_frame.empty:
        return result

    live_columns = [col for col in LIVE_PREPROCESSED_ALL_COLUMNS if col in source_frame.columns]
    if not live_columns:
        return result

    if "Plate" in result.columns and "Plate" in source_frame.columns:
        result["Plate"] = fill_missing_plate_values(result["Plate"])
        source_frame["Plate"] = fill_missing_plate_values(source_frame["Plate"])

        source_lookup = (
            source_frame
            .drop_duplicates(subset=["Plate"], keep="last")
            .set_index("Plate")
        )
        result_indexed = result.set_index("Plate", drop=False)
        overlap = result_indexed.index.intersection(source_lookup.index)
        if overlap.empty:
            return result

        for col in live_columns:
            if col == "Plate":
                continue
            aligned_live_col = source_lookup[col].reindex(result_indexed.index)
            if col not in result_indexed.columns:
                result_indexed[col] = aligned_live_col
                continue
            try:
                result_indexed.loc[overlap, col] = aligned_live_col.loc[overlap].to_numpy()
            except (TypeError, ValueError):
                result_indexed[col] = result_indexed[col].astype(object)
                result_indexed.loc[overlap, col] = aligned_live_col.loc[overlap].to_numpy()

        return result_indexed.reset_index(drop=True)

    row_count = min(len(result), len(source_frame))
    if row_count == 0:
        return result

    for col in live_columns:
        if col == "Plate" or col not in source_frame.columns:
            continue
        if col not in result.columns:
            result[col] = pd.Series([pd.NA] * len(result), index=result.index, dtype=object)
        try:
            result.iloc[:row_count, result.columns.get_loc(col)] = source_frame.iloc[:row_count][col].to_numpy()
        except (TypeError, ValueError):
            result[col] = result[col].astype(object)
            result.iloc[:row_count, result.columns.get_loc(col)] = source_frame.iloc[:row_count][col].to_numpy()

    return result

def calculate_expected_claim(row: pd.Series) -> float:
    """Calculate expected claim value for a vehicle based on risk profile."""
    base_claim = 2500.0
    
    aggressive_mult = 1.0 + (row.get('Aggressive_Driving_Score', 0.0) * 0.8)
    speeding_mult = 1.0 + (row.get('Speeding_Ratio_x_Night_Driving_Ratio', 0.0) * 1.2)
    weather_mult = row.get('Weather_Risk_Score', 1.0)
    fatigue_mult = 1.0 + (row.get('Fatigue_Risk_Score', 0.0) * 0.6)
    
    harsh_events = row.get('Recent_Harsh_Events', 0.0)
    harsh_mult = 1.0 + (min(harsh_events, 10.0) / 10.0) * 0.5
    
    expected_claim = base_claim * aggressive_mult * speeding_mult * weather_mult * fatigue_mult * harsh_mult
    return float(np.clip(expected_claim, 800.0, 15000.0))

def get_premium_kpis(df: pd.DataFrame) -> dict:
    """Calculate premium KPIs from preprocessed dataset."""
    if df is None or df.empty:
        return {
            'total_premium': 0.0,
            'avg_premium': 0.0,
            'median_premium': 0.0,
            'high_risk_count': 0,
            'low_risk_count': 0,
            'expected_claims': 0.0,
            'loss_ratio': 0.0,
        }
    
    df = df.copy()
    if 'Monthly_Premium_USD' not in df.columns:
        df['Monthly_Premium_USD'] = 50.0
    
    total_premium = float(df['Monthly_Premium_USD'].sum())
    avg_premium = float(df['Monthly_Premium_USD'].mean())
    median_premium = float(df['Monthly_Premium_USD'].median())
    
    high_risk = 0
    low_risk = 0
    if 'Risk_Band' in df.columns:
        high_risk = len(df[df['Risk_Band'] == 'Critical'])
        low_risk = len(df[df['Risk_Band'] == 'Low'])
    
    df['Expected_Claim'] = df.apply(calculate_expected_claim, axis=1)
    expected_claims = float(df['Expected_Claim'].sum())
    
    loss_ratio = expected_claims / total_premium if total_premium > 0 else 0.0
    
    return {
        'total_premium': total_premium,
        'avg_premium': avg_premium,
        'median_premium': median_premium,
        'high_risk_count': high_risk,
        'low_risk_count': low_risk,
        'expected_claims': expected_claims,
        'loss_ratio': float(np.clip(loss_ratio, 0.0, 2.0)),
    }

def build_premium_analysis_by_city(df: pd.DataFrame) -> pd.DataFrame:
    """Build premium analysis aggregated by city."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    if 'Monthly_Premium_USD' not in df.columns:
        df['Monthly_Premium_USD'] = 50.0
    if 'City' not in df.columns:
        df['City'] = 'Harare'
    
    analysis = df.groupby('City', as_index=False).agg({
        'Monthly_Premium_USD': ['sum', 'mean', 'count'],
        'Aggressive_Driving_Score': 'mean',
        'Speed_kmh': 'mean',
    }).reset_index(drop=True)

    analysis.columns = [
        'City',
        'Total_Monthly_Premium_USD',
        'Avg_Monthly_Premium_USD',
        'Vehicle_Count',
        'Avg_Aggressive_Score',
        'Avg_Speed_kmh',
    ]
    analysis = analysis.fillna(0.0)

    # Backward-compatible aliases for any older views still expecting the shorter names.
    analysis['Total_Premium'] = analysis['Total_Monthly_Premium_USD']
    analysis['Avg_Premium'] = analysis['Avg_Monthly_Premium_USD']
    analysis['Avg_Speed'] = analysis['Avg_Speed_kmh']
    return analysis

def build_premium_analysis_by_risk_band(df: pd.DataFrame) -> pd.DataFrame:
    """Build premium analysis by risk band."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    if 'Monthly_Premium_USD' not in df.columns:
        df['Monthly_Premium_USD'] = 50.0
    if 'Risk_Band' not in df.columns:
        df['Risk_Band'] = 'Medium'
    
    analysis = df.groupby('Risk_Band', as_index=False).agg({
        'Monthly_Premium_USD': ['sum', 'mean', 'count'],
        'Aggressive_Driving_Score': 'mean',
    }).reset_index(drop=True)
    
    analysis.columns = ['Risk_Band', 'Total_Premium', 'Avg_Premium', 'Vehicle_Count', 'Avg_Aggressive_Score']
    return analysis.fillna(0.0)

def detect_fraud_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Detect potential fraud flags in telemetry data."""
    if df is None or df.empty:
        return df.copy()
    
    df = df.copy()
    df['Fraud_Flag'] = 0
    
    # Flag 1: Impossible speed jumps
    if 'Speed_kmh' in df.columns:
        speed_jumps = df['Speed_kmh'].fillna(0.0).astype(float) > 180.0
        df.loc[speed_jumps, 'Fraud_Flag'] = 1
    
    # Flag 2: Unrealistic acceleration
    if 'Acceleration_mps2' in df.columns:
        extreme_accel = np.abs(df['Acceleration_mps2'].fillna(0.0).astype(float)) > 7.0
        df.loc[extreme_accel, 'Fraud_Flag'] = 1
    
    # Flag 3: Battery impossible values
    if 'Battery_V' in df.columns:
        battery_bad = (df['Battery_V'].fillna(12.6).astype(float) < 10.5) | (df['Battery_V'].fillna(12.6).astype(float) > 15.0)
        df.loc[battery_bad, 'Fraud_Flag'] = 1
    
    # Flag 4: GPS jumps >500km
    if 'GPS_Latitude' in df.columns and 'GPS_Longitude' in df.columns:
        lat_range = df['GPS_Latitude'].max() - df['GPS_Latitude'].min()
        lon_range = df['GPS_Longitude'].max() - df['GPS_Longitude'].min()
        if lat_range * 111 > 500 or lon_range * 111 > 500:
            df['Fraud_Flag'] = 1
    
    return df

def build_fraud_detection_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build fraud detection analysis table."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = detect_fraud_flags(df.copy())
    
    flagged = df[df['Fraud_Flag'] == 1][['Plate', 'Speed_kmh', 'Acceleration_mps2', 'Battery_V', 'Trip_Distance_km']].copy()
    
    if flagged.empty:
        return pd.DataFrame({'Status': ['No fraud detected']})
    
    flagged['Severity'] = 'Medium'
    flagged.loc[flagged['Speed_kmh'] > 180, 'Severity'] = 'Critical'
    flagged.loc[np.abs(flagged['Acceleration_mps2']) > 7, 'Severity'] = 'Critical'
    
    return flagged.reset_index(drop=True)
