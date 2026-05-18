"""
Risk Scoring Logic Module
Handles all risk calculations, risk band assignments, and risk-related mappings.
"""

import numpy as np
import pandas as pd

# Risk scoring constants and thresholds
RISK_MONITORING_THRESHOLD = 0.45
RISK_SURCHARGE_THRESHOLD = 0.65
RISK_MANUAL_REVIEW_THRESHOLD = 0.85
RISK_BAND_RULESET_VERSION = "2026-05-16-risk-band-v1"
RISK_BAND_MEDIUM_MIN = 0.31
RISK_BAND_HIGH_MIN = 0.51
RISK_BAND_CRITICAL_MIN = 0.81

# Risk multipliers by band
RISK_MULT = {"Low": 0.62, "Medium": 1.00, "High": 1.42, "Critical": 1.95}
EXPECTED_CLAIM = {"Low": 750, "Medium": 2200, "High": 6200, "Critical": 10000}
PUBLIC_TRANSPORT_TYPES = {"Kombi", "Minibus", "Bus"}
COMMERCIAL_RISK_TYPES = {"Pickup", "Van", "Light Truck", "Heavy Truck"}

# Weather risk mapping
WEATHER_RISK_MAP = {
    "Sunny": 0.10,
    "Cloudy": 0.20,
    "Windy": 0.45,
    "Rainy": 0.65,
    "Hot": 0.60,
    "Snowy": 0.80,
    "Stormy": 0.85,
}

# Road type risk mapping
ROAD_TYPE_RISK_MAP = {
    "Highway": 0.20,
    "Urban": 0.35,
    "Local Streets": 0.35,
    "Rural Tarred": 0.25,
    "Gravel": 0.60,
    "Potholed": 0.75,
    "Unknown": 0.40,
}

# Time of day risk mapping
TIME_OF_DAY_RISK_MAP = {
    "Morning Rush": 0.35,
    "Mid-Day": 0.20,
    "Evening Rush": 0.40,
    "Night": 0.70,
    "Late Night": 0.75,
    "Dawn": 0.30,
}

# Day of week risk mapping
DAY_OF_WEEK_RISK_MAP = {
    "Monday": 0.25,
    "Tuesday": 0.25,
    "Wednesday": 0.25,
    "Thursday": 0.25,
    "Friday": 0.35,
    "Saturday": 0.40,
    "Sunday": 0.35,
}

# Risk bins for various metrics
SPEED_RATIO_RISK_BINS = [
    (0.0, 1.01, 0.00),
    (1.01, 1.10, 0.20),
    (1.10, 1.20, 0.45),
    (1.20, 1.30, 0.70),
    (1.30, None, 0.90),
]

SPEED_EXCESS_RISK_BINS = [
    (0.0, 5.1, 0.10),
    (5.1, 10.1, 0.30),
    (10.1, 20.1, 0.60),
    (20.1, 30.1, 0.85),
    (30.1, None, 0.98),
]

HARSH_EVENTS_PER_HOUR_RISK_BINS = [
    (0.0, 2.0, 0.10),
    (2.0, 4.0, 0.30),
    (4.0, 6.0, 0.55),
    (6.0, 8.0, 0.80),
    (8.0, None, 0.95),
]

HARSH_EVENTS_PER_KM_RISK_BINS = [
    (0.0, 0.5, 0.10),
    (0.5, 1.0, 0.30),
    (1.0, 2.0, 0.55),
    (2.0, 3.0, 0.80),
    (3.0, None, 0.95),
]

HARSH_COUNT_RISK_BINS = [
    (0, 2, 0.05),
    (2, 5, 0.20),
    (5, 10, 0.45),
    (10, 15, 0.70),
    (15, None, 0.90),
]

ENGINE_STRESS_RISK_BINS = [
    (0.0, 20.0, 0.05),
    (20.0, 40.0, 0.20),
    (40.0, 60.0, 0.45),
    (60.0, 80.0, 0.70),
    (80.0, None, 0.95),
]

COOLANT_TEMP_RISK_BINS = [
    (0.0, 90.0, 0.05),
    (90.0, 100.0, 0.20),
    (100.0, 105.0, 0.50),
    (105.0, 110.0, 0.80),
    (110.0, None, 0.98),
]

BATTERY_V_RISK_BINS = [
    (0.0, 11.8, 0.90),
    (11.8, 12.2, 0.60),
    (12.2, 12.4, 0.35),
    (12.4, 12.6, 0.15),
    (12.6, None, 0.05),
]


def get_risk_score_from_bins(value: float, bins: list) -> float:
    """Get risk score for a scalar value based on bins."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.5
    for min_val, max_val, risk in bins:
        if min_val <= value and (max_val is None or value < max_val):
            return risk
    return bins[-1][2]


def get_risk_scores_from_bins(values: pd.Series, bins: list, default: float = 0.5) -> pd.Series:
    """Vectorized bin scoring for a numeric Series."""
    values = pd.to_numeric(values, errors='coerce')
    result = pd.Series(default, index=values.index, dtype=float)
    if values.isna().any():
        result.loc[values.isna()] = default

    for min_val, max_val, risk in bins:
        if max_val is None:
            mask = values >= min_val
        else:
            mask = (values >= min_val) & (values < max_val)
        result.loc[mask] = risk

    return result


def calculate_speeding_risk_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized speeding risk for a DataFrame."""
    speed_ratio = df.get('Speed_to_Road_Limit_Ratio', pd.Series(0.8, index=df.index)).fillna(0.8)
    speeding_excess = df.get('Speeding_Excess_kmh', pd.Series(0.0, index=df.index)).fillna(0.0)
    ratio_risk = get_risk_scores_from_bins(speed_ratio, SPEED_RATIO_RISK_BINS, default=0.0)
    excess_risk = get_risk_scores_from_bins(speeding_excess, SPEED_EXCESS_RISK_BINS, default=0.0)
    return 0.6 * ratio_risk + 0.4 * excess_risk


def calculate_harsh_events_risk_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized harsh events risk for a DataFrame."""
    if 'Harsh_Events_Per_Hour' in df.columns:
        harsh_per_hour = pd.to_numeric(df['Harsh_Events_Per_Hour'], errors='coerce').fillna(1.0)
    else:
        duration = pd.to_numeric(df.get('Trip_Duration_Hour', pd.Series(1.0, index=df.index)), errors='coerce').replace(0, 0.1).fillna(1.0)
        harsh_per_hour = pd.to_numeric(df.get('Recent_Harsh_Events', pd.Series(0.0, index=df.index)), errors='coerce').fillna(0.0) / duration

    harsh_per_km = pd.to_numeric(df.get('Harsh_Events_Per_Km', pd.Series(0.1, index=df.index)), errors='coerce').fillna(0.1)
    harsh_count = pd.to_numeric(df.get('Recent_Harsh_Events', pd.Series(0.0, index=df.index)), errors='coerce').fillna(0.0)

    hour_risk = get_risk_scores_from_bins(harsh_per_hour, HARSH_EVENTS_PER_HOUR_RISK_BINS, default=0.0)
    km_risk = get_risk_scores_from_bins(harsh_per_km, HARSH_EVENTS_PER_KM_RISK_BINS, default=0.0)
    count_risk = get_risk_scores_from_bins(harsh_count, HARSH_COUNT_RISK_BINS, default=0.0)
    return 0.4 * hour_risk + 0.3 * km_risk + 0.3 * count_risk


def calculate_engine_risk_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized engine and temperature risk for a DataFrame."""
    coolant_temp = pd.to_numeric(df.get('Coolant_Temp_C', pd.Series(95.0, index=df.index)), errors='coerce').fillna(95.0)
    battery_v = pd.to_numeric(df.get('Battery_V', pd.Series(12.5, index=df.index)), errors='coerce').fillna(12.5)
    engine_load = pd.to_numeric(df.get('Engine_Load_pct', pd.Series(40.0, index=df.index)), errors='coerce').fillna(40.0)
    rpm = pd.to_numeric(df.get('RPM', pd.Series(2500.0, index=df.index)), errors='coerce').fillna(2500.0)

    coolant_risk = get_risk_scores_from_bins(coolant_temp, COOLANT_TEMP_RISK_BINS, default=0.0)
    battery_risk = get_risk_scores_from_bins(battery_v, BATTERY_V_RISK_BINS, default=0.0)
    stress_score = (engine_load / 100.0) * (rpm / 7000.0) * 100.0
    stress_risk = get_risk_scores_from_bins(stress_score, ENGINE_STRESS_RISK_BINS, default=0.0)
    return 0.5 * coolant_risk + 0.25 * battery_risk + 0.25 * stress_risk


def normalize_numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """Convert a column to numeric and fill missing values with a default."""
    return pd.to_numeric(df.get(col, pd.Series(default, index=df.index)), errors='coerce').fillna(default)


def calculate_road_type_risk_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized road type risk score based on road type or existing score."""
    road_risk = pd.to_numeric(df.get('Road_Type_Risk_Score', pd.Series(np.nan, index=df.index)), errors='coerce')
    if road_risk.isna().all():
        road_risk = df.get('Road_Type', pd.Series('Unknown', index=df.index)).map(ROAD_TYPE_RISK_MAP).fillna(0.40)
    return road_risk.clip(0.0, 1.0)


def calculate_time_of_day_risk_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized time-of-day risk score based on time buckets or existing score."""
    time_risk = pd.to_numeric(df.get('Time_of_Day_Risk_Score', pd.Series(np.nan, index=df.index)), errors='coerce')
    if time_risk.isna().all():
        time_risk = df.get('Time_of_Day', pd.Series('Mid-Day', index=df.index)).map(TIME_OF_DAY_RISK_MAP).fillna(0.30)
    return time_risk.clip(0.0, 1.0)


def calculate_day_of_week_risk_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized day-of-week risk score based on day or existing score."""
    day_risk = pd.to_numeric(df.get('Day_of_Week_Risk_Score', pd.Series(np.nan, index=df.index)), errors='coerce')
    if day_risk.isna().all():
        day_risk = df.get('Day_of_Week', pd.Series('Monday', index=df.index)).map(DAY_OF_WEEK_RISK_MAP).fillna(0.30)
    return day_risk.clip(0.0, 1.0)


def calculate_fatigue_risk_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized fatigue risk based on an existing fatigue score."""
    return normalize_numeric_series(df, 'Fatigue_Risk_Score', 0.0).clip(0.0, 1.0)


def calculate_weather_risk_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized weather risk score using existing score if normalized or mapping from conditions."""
    weather_score = pd.to_numeric(df.get('Weather_Risk_Score', pd.Series(np.nan, index=df.index)), errors='coerce')
    if weather_score.isna().all():
        weather_score = df.get('Weather', pd.Series('Sunny', index=df.index)).map(WEATHER_RISK_MAP).fillna(0.30)
    else:
        weather_score = weather_score.where(weather_score.between(0.0, 1.0), weather_score / 1.3)
    return weather_score.clip(0.0, 1.0)


CITY_RISK_SCORE_MAP = {
    'Harare': 0.50,
    'Bulawayo': 0.45,
    'Mutare': 0.40,
    'Gweru': 0.35,
    'Masvingo': 0.30,
    'Kwekwe': 0.20,
    'Kadoma': 0.18,
    'Chinhoyi': 0.18,
    'Victoria Falls': 0.18,
    'Beitbridge': 0.18,
    'Chegutu': 0.18,
    'Redcliff': 0.18,
    'Zvishavane': 0.18,
    'Shurugwi': 0.18,
    'Plumtree': 0.18,
    'Macheke': 0.18,
    'Mvuma': 0.18,
    'Gwanda': 0.18,
}


def calculate_city_risk_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized city risk bias for major cities."""
    return df.get('City', pd.Series('', index=df.index)).map(CITY_RISK_SCORE_MAP).fillna(0.15).astype(float)


def calculate_exposure_risk_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized exposure risk based on trip distance."""
    trip_distance = normalize_numeric_series(df, 'Trip_Distance_km', 0.0)
    return np.minimum(trip_distance / 100.0, 1.0)


def calculate_environment_risk_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized environmental risk based on road type and time of day risk scores."""
    return 0.5 * calculate_road_type_risk_series(df) + 0.5 * calculate_time_of_day_risk_series(df)


def calculate_environmental_risk_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized environmental risk for a DataFrame."""
    weather_risk = calculate_weather_risk_series(df)
    road_risk = calculate_road_type_risk_series(df)
    time_risk = calculate_time_of_day_risk_series(df)
    day_risk = calculate_day_of_week_risk_series(df)
    city_risk = calculate_city_risk_series(df)
    return (
        0.20 * weather_risk
        + 0.30 * road_risk
        + 0.20 * time_risk
        + 0.10 * day_risk
        + 0.20 * city_risk
    ).clip(0.0, 1.0)


def calculate_vehicle_condition_risk_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized vehicle condition risk for a DataFrame."""
    vehicle_age = pd.to_numeric(df.get('Vehicle_Age_Years', pd.Series(5.0, index=df.index)), errors='coerce').fillna(5.0)
    efficiency = pd.to_numeric(df.get('Fuel_Efficiency_L_per_100km', pd.Series(8.5, index=df.index)), errors='coerce').fillna(8.5)
    age_risk = np.minimum(vehicle_age / 20.0, 1.0) * 0.5
    efficiency_risk = np.maximum((efficiency - 5.0) / 10.0, 0.0) * 0.3
    return age_risk + efficiency_risk


def calculate_behavioral_risk_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized behavioral risk for a DataFrame."""
    night_driving = df.get('Night_Driving_Flag', pd.Series(0, index=df.index)).fillna(0).astype(bool)
    is_weekend = df.get('Is_Weekend', pd.Series(0, index=df.index)).fillna(0).astype(bool)
    evening_peak = df.get('Evening_Peak_Flag', pd.Series(0, index=df.index)).fillna(0).astype(bool)
    night_risk = np.where(night_driving, 0.6, 0.2)
    weekend_risk = np.where(is_weekend, 0.35, 0.25)
    peak_risk = np.where(evening_peak, 0.5, 0.3)
    return 0.4 * night_risk + 0.3 * weekend_risk + 0.3 * peak_risk


def calculate_risk_score_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized overall risk score for a DataFrame."""
    return (
        0.30 * calculate_speeding_risk_series(df)
        + 0.25 * calculate_harsh_events_risk_series(df)
        + 0.15 * calculate_fatigue_risk_series(df)
        + 0.10 * calculate_weather_risk_series(df)
        + 0.10 * calculate_exposure_risk_series(df)
        + 0.10 * calculate_environment_risk_series(df)
    ).clip(0, 1)


def _apply_risk_score_smoothing(new_score: pd.Series, prior_score: pd.Series, max_delta: float = 0.02) -> pd.Series:
    """Limit per-vehicle risk score movement to a small step each refresh."""
    prior = pd.to_numeric(prior_score, errors='coerce')
    new_score = pd.to_numeric(new_score, errors='coerce').clip(0.0, 1.0)
    if prior.isna().all():
        return new_score

    delta = (new_score - prior).clip(lower=-max_delta, upper=max_delta)
    smoothed = prior + delta
    return pd.Series(np.where(prior.notna(), smoothed, new_score), index=new_score.index).clip(0.0, 1.0)


def assign_risk_band(risk_score: pd.Series) -> pd.Series:
    """Assign a risk band from a numeric score."""
    risk_score = pd.to_numeric(risk_score, errors='coerce').fillna(0.5)
    risk_score = risk_score.clip(0.0, 1.0)
    conditions = [
        risk_score >= RISK_BAND_CRITICAL_MIN,
        risk_score >= RISK_BAND_HIGH_MIN,
        risk_score >= RISK_BAND_MEDIUM_MIN,
    ]
    choices = ['Critical', 'High', 'Medium']
    return pd.Series(np.select(conditions, choices, default='Low'), index=risk_score.index)


def normalize_shared_risk_columns(
    df: pd.DataFrame,
    source_column: str | None = None,
) -> pd.DataFrame:
    """Keep shared risk columns aligned everywhere the same attributes are reused."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    result = df.copy()
    candidate_columns: list[str] = []
    if source_column:
        candidate_columns.append(source_column)
    candidate_columns.extend(['Calculated_Risk_Score', 'Risk_Score', 'LightGBM_Risk_Score', 'Custom_Risk_Score'])

    canonical_score = None
    seen_columns: set[str] = set()
    for column in candidate_columns:
        if column in seen_columns or column not in result.columns:
            continue
        seen_columns.add(column)
        numeric_values = pd.to_numeric(result[column], errors='coerce')
        if numeric_values.notna().any():
            canonical_score = numeric_values
            break

    if canonical_score is None:
        canonical_score = pd.Series(0.5, index=result.index, dtype=float)

    canonical_score = canonical_score.fillna(0.5).clip(0.0, 1.0).astype(float)
    result['Calculated_Risk_Score'] = canonical_score
    result['Risk_Score'] = canonical_score

    model_score = pd.to_numeric(
        result.get('LightGBM_Risk_Score', pd.Series(np.nan, index=result.index, dtype=float)),
        errors='coerce',
    )
    result['LightGBM_Risk_Score'] = model_score.fillna(canonical_score).clip(0.0, 1.0).astype(float)
    result['Risk_Band'] = assign_risk_band(canonical_score)
    result['Calculated_Risk_Band'] = result['Risk_Band']
    return result


def ensure_risk_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure derived risk columns exist and are calculated consistently."""
    df = df.copy()
    speed = normalize_numeric_series(df, 'Speed_kmh', 0.0)
    road_limit = pd.to_numeric(df.get('Base_Road_Max_Speed_kmh', pd.Series(np.nan, index=df.index)), errors='coerce')

    speed_ratio = pd.to_numeric(df.get('Speed_to_Road_Limit_Ratio', pd.Series(np.nan, index=df.index)), errors='coerce')
    speed_ratio = speed_ratio.fillna(speed.div(road_limit.replace({0.0: np.nan})))
    speed_ratio = speed_ratio.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df['Speed_to_Road_Limit_Ratio'] = speed_ratio.clip(lower=0.0)

    speeding_excess = pd.to_numeric(df.get('Speeding_Excess_kmh', pd.Series(np.nan, index=df.index)), errors='coerce')
    speeding_excess = speeding_excess.fillna((speed - road_limit).clip(lower=0.0))
    df['Speeding_Excess_kmh'] = speeding_excess.fillna(0.0)

    speeding_flag = pd.to_numeric(df.get('Speeding_Flag', pd.Series(np.nan, index=df.index)), errors='coerce')
    if speeding_flag.isna().all():
        speeding_flag = (speed > road_limit).astype(int)
    df['Speeding_Flag'] = speeding_flag.fillna(0).astype(int)

    duration = pd.to_numeric(
        df.get('Trip_Duration_Hour', df.get('Trip_Duration_Hours', pd.Series(1.0, index=df.index))),
        errors='coerce'
    ).replace(0.0, 0.1).fillna(1.0)

    harsh_events = normalize_numeric_series(df, 'Recent_Harsh_Events', 0.0)
    harsh_count_columns = [
        'Harsh_Corner_Count',
        'Harsh_Accel_Count',
        'Harsh_Brake_Count',
    ]
    if all(col in df.columns for col in harsh_count_columns):
        df['Harsh_Events_Per_Day'] = df[harsh_count_columns].fillna(0).astype(int).sum(axis=1)
    elif 'Harsh_Events_Per_Day' not in df.columns or df['Harsh_Events_Per_Day'].isna().all():
        df['Harsh_Events_Per_Day'] = (harsh_events / duration * 24.0).fillna(0.0).round(0).astype(int)
    else:
        df['Harsh_Events_Per_Day'] = normalize_numeric_series(df, 'Harsh_Events_Per_Day', 0.0).round(0).astype(int)

    df['Aggressive_Driving_Score'] = pd.to_numeric(df.get('Aggressive_Driving_Score', pd.Series(np.nan, index=df.index)), errors='coerce')
    if df['Aggressive_Driving_Score'].isna().all():
        df['Aggressive_Driving_Score'] = np.clip(
            0.35 * (harsh_events / 5.0)
            + 0.35 * df['Speeding_Flag'].astype(float)
            + 0.30 * np.clip(np.abs(normalize_numeric_series(df, 'Acceleration_mps2', 0.0)) / 6.0, 0.0, 1.0),
            0.0,
            1.0,
        )
    else:
        df['Aggressive_Driving_Score'] = normalize_numeric_series(df, 'Aggressive_Driving_Score', 0.0)

    df['Recent_Avg_Speed'] = pd.to_numeric(df.get('Recent_Avg_Speed', pd.Series(np.nan, index=df.index)), errors='coerce')
    if df['Recent_Avg_Speed'].isna().all():
        df['Recent_Avg_Speed'] = np.clip(speed * 0.95, 0.0, 180.0)
    else:
        df['Recent_Avg_Speed'] = normalize_numeric_series(df, 'Recent_Avg_Speed', 0.0)

    df['Recent_Night_Distance'] = pd.to_numeric(df.get('Recent_Night_Distance', pd.Series(np.nan, index=df.index)), errors='coerce')
    if df['Recent_Night_Distance'].isna().all():
        night_flag = df.get('Night_Driving_Flag', pd.Series(0, index=df.index)).fillna(0).astype(int)
        df['Recent_Night_Distance'] = np.where(
            night_flag == 1,
            speed * 0.45,
            speed * 0.08,
        ).astype(float)
    else:
        df['Recent_Night_Distance'] = normalize_numeric_series(df, 'Recent_Night_Distance', 0.0)

    df['Night_Driving_Flag'] = pd.to_numeric(df.get('Night_Driving_Flag', pd.Series(np.nan, index=df.index)), errors='coerce')
    if df['Night_Driving_Flag'].isna().all():
        df['Night_Driving_Flag'] = df.get('Time_of_Day', pd.Series('Mid-Day', index=df.index)).isin(
            ['Night', 'Late Night', 'Dawn']
        ).astype(int)
    else:
        df['Night_Driving_Flag'] = df['Night_Driving_Flag'].fillna(0).astype(int)

    df['Road_Type_Risk_Score'] = calculate_road_type_risk_series(df)
    df['Time_of_Day_Risk_Score'] = calculate_time_of_day_risk_series(df)
    df['Day_of_Week_Risk_Score'] = calculate_day_of_week_risk_series(df)

    df['Total_Distance_Monthly_km'] = pd.to_numeric(df.get('Total_Distance_Monthly_km', pd.Series(np.nan, index=df.index)), errors='coerce')
    df['Total_Distance_Monthly_km'] = df['Total_Distance_Monthly_km'].fillna(normalize_numeric_series(df, 'Trip_Distance_km', 0.0) * 22.0)

    return df


def compute_live_risk_outputs(scored: pd.DataFrame) -> pd.DataFrame:
    """Compute live risk outputs for all vehicles in the dataset."""
    df = scored.copy()
    if df.empty:
        return df

    df = ensure_risk_feature_columns(df)
    df['City_Risk_Score'] = calculate_city_risk_series(df)
    df['Speeding_Risk'] = calculate_speeding_risk_series(df)
    df['Harsh_Risk'] = calculate_harsh_events_risk_series(df)
    df['Engine_Risk'] = calculate_engine_risk_series(df)
    df['Environmental_Risk'] = calculate_environmental_risk_series(df)
    df['Vehicle_Condition_Risk'] = calculate_vehicle_condition_risk_series(df)
    df['Behavioral_Risk'] = calculate_behavioral_risk_series(df)

    prior_score = pd.to_numeric(
        df.get('Calculated_Risk_Score', df.get('Risk_Score', pd.Series(np.nan, index=df.index))),
        errors='coerce'
    )
    df['Calculated_Risk_Score'] = calculate_risk_score_series(df)
    df['Calculated_Risk_Score'] = _apply_risk_score_smoothing(df['Calculated_Risk_Score'], prior_score, max_delta=0.02)

    usage_series = df.get('Usage', pd.Series('', index=df.index)).astype(str).str.lower()
    type_series = df.get('Type', pd.Series('', index=df.index)).astype(str)
    public_transport_mask = type_series.isin(PUBLIC_TRANSPORT_TYPES)
    commercial_vehicle_mask = (usage_series == 'commercial') | type_series.isin(COMMERCIAL_RISK_TYPES)
    older_public_vehicle_mask = public_transport_mask & (
        pd.to_numeric(df.get('Vehicle_Age_Years', pd.Series(0.0, index=df.index)), errors='coerce').fillna(0.0) >= 12.0
    )

    df.loc[commercial_vehicle_mask, 'Calculated_Risk_Score'] = (
        df.loc[commercial_vehicle_mask, 'Calculated_Risk_Score'] + 0.015
    ).clip(0.0, 1.0)
    df.loc[public_transport_mask, 'Calculated_Risk_Score'] = (
        df.loc[public_transport_mask, 'Calculated_Risk_Score'] + 0.025
    ).clip(0.0, 1.0)
    df.loc[older_public_vehicle_mask, 'Calculated_Risk_Score'] = (
        df.loc[older_public_vehicle_mask, 'Calculated_Risk_Score'] + 0.010
    ).clip(0.0, 1.0)

    df = normalize_shared_risk_columns(df, source_column='Calculated_Risk_Score')

    lightgbm_score = pd.to_numeric(df.get('LightGBM_Risk_Score', pd.Series(np.nan, index=df.index)), errors='coerce')
    df['Custom_Risk_Score'] = np.where(
        lightgbm_score.notna(),
        (df['Calculated_Risk_Score'] + lightgbm_score) / 2.0,
        df['Calculated_Risk_Score']
    )

    if 'Risk_Label' not in df.columns:
        df['Risk_Label'] = df['Risk_Band']

    default_expected_claim = df['Risk_Band'].map(EXPECTED_CLAIM).fillna(2200.0).astype(float)
    existing_expected_claim = pd.to_numeric(
        df.get('Expected_Claim', pd.Series(np.nan, index=df.index, dtype=float)),
        errors='coerce',
    )
    expected_claim_usd = pd.to_numeric(
        df.get('Expected_Claim_USD', pd.Series(np.nan, index=df.index, dtype=float)),
        errors='coerce',
    )
    aligned_expected_claim = expected_claim_usd.fillna(existing_expected_claim).fillna(default_expected_claim).astype(float)
    df['Expected_Claim'] = aligned_expected_claim
    df['Expected_Claim_USD'] = aligned_expected_claim
    df['Risk_Action'] = df['Risk_Band'].map({
        'Low': 'Monitor',
        'Medium': 'Recommend Coaching',
        'High': 'Schedule Inspection',
        'Critical': 'Immediate Review'
    }).fillna('Monitor')

    return df


def calculate_risk_kpis(df: pd.DataFrame) -> dict:
    """Return a set of high-level risk KPIs for dashboard display."""
    if df is None or df.empty:
        return {
            'total_vehicles': 0,
            'avg_risk_score': 0.0,
            'high_risk_count': 0,
            'critical_risk_count': 0,
            'median_risk_score': 0.0,
            'expected_claim_total': 0.0,
            'expected_claim_avg': 0.0,
            'risk_action_counts': {},
        }
    has_live_risk_columns = {'Calculated_Risk_Score', 'Risk_Band'}.issubset(df.columns)
    risk_df = normalize_shared_risk_columns(df) if has_live_risk_columns else compute_live_risk_outputs(df)
    if 'Expected_Claim' not in risk_df.columns:
        if 'Expected_Claim_USD' in risk_df.columns:
            risk_df['Expected_Claim'] = pd.to_numeric(risk_df['Expected_Claim_USD'], errors='coerce').fillna(0.0)
        else:
            risk_df['Expected_Claim'] = risk_df['Risk_Band'].map(EXPECTED_CLAIM).fillna(2200.0).astype(float)
    if 'Risk_Action' not in risk_df.columns:
        risk_df['Risk_Action'] = risk_df['Risk_Band'].map({
            'Low': 'Monitor',
            'Medium': 'Recommend Coaching',
            'High': 'Schedule Inspection',
            'Critical': 'Immediate Review'
        }).fillna('Monitor')
    expected_claim = pd.to_numeric(risk_df['Expected_Claim'], errors='coerce').fillna(0.0)
    band_counts = risk_df['Risk_Band'].value_counts().to_dict()
    total = len(risk_df)
    low_count = int(band_counts.get('Low', 0))
    medium_count = int(band_counts.get('Medium', 0))
    high_count = int(band_counts.get('High', 0))
    critical_count = int(band_counts.get('Critical', 0))
    avg_risk_score = float(risk_df['Calculated_Risk_Score'].dropna().mean()) if total else 0.0
    return {
        'total_vehicles': total,
        'avg_risk_score': avg_risk_score,
        'low_count': low_count,
        'medium_count': medium_count,
        'high_count': high_count,
        'critical_count': critical_count,
        'low_pct': float(low_count / total) if total else 0.0,
        'medium_pct': float(medium_count / total) if total else 0.0,
        'high_pct': float(high_count / total) if total else 0.0,
        'critical_pct': float(critical_count / total) if total else 0.0,
        'high_risk_count': int((risk_df['Risk_Band'].isin(['High', 'Critical'])).sum()),
        'critical_risk_count': critical_count,
        'median_risk_score': float(risk_df['Calculated_Risk_Score'].median()),
        'expected_claim_total': float(expected_claim.sum()),
        'expected_claim_avg': float(expected_claim.mean()),
        'risk_action_counts': risk_df['Risk_Action'].value_counts().to_dict(),
    }


def build_risk_band_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize risk exposure by risk band."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = compute_live_risk_outputs(df)
    return (
        df.groupby('Risk_Band')
          .agg(
              Vehicles=('Plate', 'count'),
              Avg_Risk_Score=('Calculated_Risk_Score', 'mean'),
              Total_Expected_Claim=('Expected_Claim', 'sum'),
              Avg_Expected_Claim=('Expected_Claim', 'mean')
          )
          .reset_index()
          .sort_values(by='Avg_Risk_Score', ascending=False)
    )


def build_city_risk_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize risk exposure by city using the full preprocessed dataset."""
    if df is None or df.empty:
        return pd.DataFrame()
    full_df = df.copy()
    if 'Calculated_Risk_Score' not in full_df.columns or 'Risk_Band' not in full_df.columns:
        full_df = compute_live_risk_outputs(full_df)

    def most_common_band(series: pd.Series) -> str:
        mode = series.mode()
        if not mode.empty:
            return mode.iloc[0]
        return series.iloc[0] if len(series) > 0 else ''

    summary = (
        full_df.groupby('City')
               .agg(
                   Vehicles=('Plate', 'count'),
                   Calculated_Risk_Score_avg=('Calculated_Risk_Score', 'mean'),
                   Risk_Band=('Risk_Band', most_common_band)
               )
               .reset_index()
    )
    summary = summary.rename(columns={'Calculated_Risk_Score_avg': 'Calculated_Risk_Score(avg)'})
    return summary.sort_values(by='Calculated_Risk_Score(avg)', ascending=False)


def build_risk_recommendation_table(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Return the top vehicles requiring review, with risk action recommendations."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = compute_live_risk_outputs(df)
    selected = [
        'Plate', 'City', 'Type', 'Speed_kmh', 'Risk_Band',
        'Calculated_Risk_Score', 'Expected_Claim', 'Risk_Action',
        'Speeding_Risk', 'Harsh_Risk', 'Engine_Risk',
        'Environmental_Risk', 'Vehicle_Condition_Risk', 'Behavioral_Risk'
    ]
    columns = [c for c in selected if c in df.columns]
    return (
        df.sort_values(by='Calculated_Risk_Score', ascending=False)
          .loc[:, columns]
          .head(top_n)
          .reset_index(drop=True)
    )


def build_prediction_scoring_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build prediction and scoring table for display using the preprocessed dataset."""
    if df is None or df.empty:
        return pd.DataFrame()
    if 'Calculated_Risk_Score' not in df.columns or 'Risk_Band' not in df.columns:
        df = compute_live_risk_outputs(df)
    cols = [
        'Plate', 'Type', 'City', 'City_Risk_Score', 'Speed_kmh', 'Recent_Harsh_Events',
        'Road_Type', 'Weather', 'Time_of_Day',
        'Calculated_Risk_Score', 'Risk_Band', 'Expected_Claim'
    ]
    columns = [c for c in cols if c in df.columns]
    return df.loc[:, columns].head(100).copy()


def style_prediction_dataframe(df: pd.DataFrame):
    """Apply styling to prediction dataframe."""
    def color_risk_band(val):
        if val == 'Critical':
            return 'background-color: #8B0000; color: white'
        if val == 'High':
            return 'background-color: #E74C3C; color: white'
        if val == 'Medium':
            return 'background-color: #F39C12; color: white'
        if val == 'Low':
            return 'background-color: #1ABC9C; color: white'
        return ''

    styled = df.style.applymap(
        lambda x: color_risk_band(x) if isinstance(x, str) and x in ['Low', 'Medium', 'High', 'Critical'] else ''
    )
    return styled
