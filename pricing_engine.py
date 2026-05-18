"""
Pricing Engine Module
Handles all premium calculations, profitability analysis, and pricing strategies.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import threading
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import streamlit as st

from risk_scoring_logic import (
    RISK_BAND_CRITICAL_MIN,
    RISK_BAND_HIGH_MIN,
    RISK_BAND_MEDIUM_MIN,
    normalize_shared_risk_columns,
)

# Base premium by vehicle type (Annual USD) - scaled for $30-$400/month range
BASE_PREMIUM = {
    "Sedan": 600, "Hatchback": 480, "Station Wagon": 550, "SUV": 900,
    "Pickup": 1100, "Van": 550, "Kombi": 520, "Minibus": 520, "Bus": 400,
    "Light Truck": 1400, "Heavy Truck": 2400,
    "Fuel Tanker": 2300, "Tipper Truck": 2100,
    "Tractor": 350, "Luxury": 1200,
}

# Expense and yield constants
EXPENSE_RATIO = 0.17
INV_YIELD = 0.04
PRIVATE_EXPENSE_RATIO = 0.12
COMMERCIAL_EXPENSE_RATIO = 0.16

# Action-related cost arrays
ACTION_EXPECTED_LOSS_MULTIPLIERS = np.array([1.00, 0.88, 0.75, 0.50], dtype=float)
ACTION_MONTHLY_CONTROL_COSTS = np.array([0.0, 8.0, 5.5, 7.5], dtype=float)

# Time of day risk multipliers
TOD_RISK = {'Day': 1.0, 'Night': 1.45, 'Early_Morning': 1.35, 'Evening': 1.25, 'Dawn': 1.15}

# Weather risk multipliers
WX_RISK = {
    'Sunny': 1.0, 'Cloudy': 1.08, 'Rainy': 1.35, 'Heavy_Rain': 1.65,
    'Foggy': 1.42, 'Snowy': 1.78, 'Hail': 1.85
}

# Currency exchange
USD_TO_ZIG_RATE_DEFAULT = 26.5  # Use realistic interbank USD→ZIG rates (25.0–28.0 ZIG per USD)
USD_TO_ZIG_RATE_MIN = 25.0
USD_TO_ZIG_RATE_MAX = 28.0

# PPO policy artifact path and action labels
LIGHTGBM_MODEL_PATH = Path("artifacts") / "hybrid_risk" / "lightgbm_risk_model.pkl"
PPO_POLICY_PATH = Path("artifacts") / "hybrid_risk" / "ppo_premium_policy"
PPO_SCALER_PATH = Path("artifacts") / "hybrid_risk" / "ppo_observation_scaler.pkl"
PPO_ACTION_LABELS = ["standard_offer", "monitoring", "surcharge", "decline"]
PPO_ACTION_BY_RISK_BAND = {
    "Low": "standard_offer",
    "Medium": "monitoring",
    "High": "surcharge",
    "Critical": "decline",
}
PPO_ACTION_PREMIUM_MULTIPLIERS = {
    "standard_offer": 0.85,
    "monitoring": 1.00,
    "surcharge": 1.25,
    "decline": 0.00,
}

# When False, the Streamlit app will NOT load or use trained LightGBM/PPO artifacts
# and will instead use lightweight deterministic fallbacks for scoring and actions.
# This ensures the app runs without ML dependencies or pre-trained artifacts.
USE_TRAINED_MODELS = os.getenv("INSURTECH_USE_TRAINED_MODELS", "1").strip().lower() not in {"0", "false", "no"}
# Background scoring worker state and caches
_SCORER_STATE = {
    "thread": None,
    "stop_event": threading.Event(),
    "lock": threading.RLock(),
    "latest_scored_df": None,
    "last_run": 0.0,
    "interval_seconds": 15.0,
    "running": False,
}

# Cache for PPO StandardScaler per feature set (tuple of feature names)
_PPO_SCALER_CACHE: dict = {}


@dataclass(frozen=True)
class ModelArtifactPaths:
    lightgbm_model_path: Path = LIGHTGBM_MODEL_PATH
    ppo_policy_path: Path = PPO_POLICY_PATH
    ppo_scaler_path: Path = PPO_SCALER_PATH


def start_background_scoring(get_preprocessed_callable: Callable[[], pd.DataFrame], interval_seconds: float = 15.0) -> None:
    """Start a background thread that periodically scores the preprocessed dataset.

    The callable should return the current preprocessed dataset (pd.DataFrame).
    Scored results are stored in an internal cache and can be read via
    `get_latest_scored_dataframe()`.
    """
    if not callable(get_preprocessed_callable):
        return
    # Update interval if already running
    if _SCORER_STATE.get("running"):
        _SCORER_STATE["interval_seconds"] = float(interval_seconds)
        return

    _SCORER_STATE["interval_seconds"] = float(interval_seconds)
    _SCORER_STATE["stop_event"].clear()

    def _worker():
        stop_ev = _SCORER_STATE["stop_event"]
        lock = _SCORER_STATE["lock"]
        while not stop_ev.is_set():
            try:
                df = get_preprocessed_callable()
                if df is None or df.empty:
                    stop_ev.wait(max(1.0, _SCORER_STATE["interval_seconds"]))
                    continue

                scored = _ensure_lightgbm_scored(df.copy())
                scored = _apply_ppo_policy(scored)

                with lock:
                    _SCORER_STATE["latest_scored_df"] = scored
                    _SCORER_STATE["last_run"] = time.time()
            except Exception:
                # Keep worker alive on errors
                pass
            stop_ev.wait(_SCORER_STATE.get("interval_seconds", 15.0))
        _SCORER_STATE["running"] = False

    t = threading.Thread(target=_worker, daemon=True, name="pricing_engine_scorer")
    _SCORER_STATE["thread"] = t
    _SCORER_STATE["running"] = True
    t.start()


def stop_background_scoring() -> None:
    try:
        ev = _SCORER_STATE["stop_event"]
        ev.set()
        th = _SCORER_STATE.get("thread")
        if th is not None and th.is_alive():
            th.join(timeout=2.0)
    finally:
        _SCORER_STATE["running"] = False


def get_latest_scored_dataframe(max_age_seconds: float = 60.0) -> Optional[pd.DataFrame]:
    """Return a fresh scored DataFrame from the background worker if available."""
    with _SCORER_STATE["lock"]:
        last = _SCORER_STATE.get("last_run", 0.0)
        df = _SCORER_STATE.get("latest_scored_df")
        if df is None:
            return None
        if time.time() - last > float(max_age_seconds):
            return None
        try:
            return df.copy()
        except Exception:
            return df


def _resolve_policy_artifact_path(path: Path | str) -> Path:
    resolved_path = Path(path)
    if resolved_path.exists():
        return resolved_path
    if resolved_path.suffix != ".zip":
        zip_path = resolved_path.with_suffix(".zip")
        if zip_path.exists():
            return zip_path
    return resolved_path


def _get_or_fit_ppo_scaler(encoded: pd.DataFrame, feature_columns: list[str]):
    """Return a fitted fallback StandardScaler for the given feature columns."""
    key = tuple(feature_columns)
    scaler = _PPO_SCALER_CACHE.get(key)
    if scaler is None:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(encoded[feature_columns].fillna(0.0).astype(float))
        _PPO_SCALER_CACHE[key] = scaler
    return scaler

@st.cache_resource(show_spinner=False)
def _load_lightgbm_model(model_path: Path | str = LIGHTGBM_MODEL_PATH):
    if not USE_TRAINED_MODELS:
        return None
    from train_hybrid_risk_model import load_trained_model

    model_path = Path(model_path)
    if not model_path.exists():
        return None
    try:
        return load_trained_model(model_path)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_ppo_policy(path: Path | str = PPO_POLICY_PATH):
    if not USE_TRAINED_MODELS:
        return None
    from train_hybrid_risk_model import load_ppo_policy

    path = _resolve_policy_artifact_path(path)
    if not path.exists():
        return None

    try:
        return load_ppo_policy(path)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_ppo_scaler(path: Path | str = PPO_SCALER_PATH):
    path = Path(path)
    if not path.exists():
        return None

    try:
        import joblib

        return joblib.load(path)
    except Exception:
        return None


def _build_ppo_observations(encoded: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    scaler = _load_ppo_scaler()
    if scaler is not None:
        scaler_columns = list(getattr(scaler, "feature_names_in_", feature_columns))
        aligned = encoded.reindex(columns=scaler_columns, fill_value=0.0)
        return scaler.transform(aligned.fillna(0.0).astype(float)).astype(np.float32)

    scaler = _get_or_fit_ppo_scaler(encoded, feature_columns)
    return scaler.transform(encoded[feature_columns].fillna(0.0).astype(float)).astype(np.float32)

# Public vehicle types for commercial pricing
PUBLIC_VEHICLE_TYPES = {"Kombi", "Minibus", "Pickup", "Heavy Truck"}

# Claim severity factor as a proportion of insured vehicle value.
# This keeps expected monthly claim costs in a sustainable band while
# preserving realistic USD premiums and risk differentiation.
AVERAGE_CLAIM_SEVERITY_RATIO = 0.10


def _usd_to_zig(amounts: np.ndarray | pd.Series, usd_to_zig_rate: float) -> np.ndarray:
    values = np.asarray(amounts, dtype=float)
    return np.round(values * float(usd_to_zig_rate), 2)


def _risk_band_from_score(score: float) -> str:
    if score >= RISK_BAND_CRITICAL_MIN:
        return "Critical"
    if score >= RISK_BAND_HIGH_MIN:
        return "High"
    if score >= RISK_BAND_MEDIUM_MIN:
        return "Medium"
    return "Low"


def _jitter_risk_score(df: pd.DataFrame, slot_seconds: int = 900, step: float = 0.005) -> pd.DataFrame:
    if df is None or df.empty or 'Risk_Score' not in df.columns:
        return df

    slot = int(time.time() // slot_seconds)
    today = datetime.utcnow().date()
    result = df.copy()

    def _jitter_row(row: pd.Series) -> float:
        base = row['Risk_Score']
        if pd.isna(base):
            return base

        key = str(row.get('Plate', '')).strip()
        if not key:
            return float(np.clip(base, 0.0, 1.0))

        daily_stable = abs(hash((key, today))) % 12 == 0
        if daily_stable:
            return float(np.clip(base, 0.0, 1.0))

        direction = (abs(hash((key, slot))) % 3) - 1
        prev_direction = (abs(hash((key, slot - 1))) % 3) - 1
        if direction != prev_direction:
            direction = 0

        return float(np.clip(base + direction * step, 0.0, 1.0))

    result['Risk_Score'] = result.apply(_jitter_row, axis=1)
    return result


def _fallback_lightgbm_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic fallback risk scoring used when trained models are disabled.

    Produces `LightGBM_Risk_Score`, `Risk_Score`, `Calculated_Risk_Score`, and `Risk_Band`.
    The formula is a simple, explainable weighted sum of normalized signals.
    """
    if df is None or df.empty:
        return df
    result = df.copy()

    def safe_series(name, default=0.0):
        return pd.to_numeric(result.get(name, pd.Series(default, index=result.index)), errors='coerce').fillna(default).astype(float)

    base_speed = safe_series('Base_Road_Max_Speed_kmh', 85.0)
    speeding_excess = safe_series('Speeding_Excess_kmh', 0.0)
    recent_harsh = safe_series('Recent_Harsh_Events', 0.0)
    battery = safe_series('Battery_Health_Score', 55.0)
    aggressive = safe_series('Aggressive_Driving_Score', 0.0)
    coolant_flag = pd.to_numeric(result.get('Coolant_Overheat_Flag', pd.Series(0, index=result.index)), errors='coerce').fillna(0).astype(int)
    night_flag = pd.to_numeric(result.get('Night_Driving_Flag', pd.Series(0, index=result.index)), errors='coerce').fillna(0).astype(int)

    # Normalize inputs
    harsh_norm = np.clip(recent_harsh / 5.0, 0.0, 1.0)
    speeding_norm = np.clip(speeding_excess / (base_speed + 1e-6), 0.0, 1.0)
    battery_risk = np.clip(1.0 - (battery / 100.0), 0.0, 1.0)

    # Weighted, explainable risk aggregation
    risk_score = (
        0.25 * harsh_norm
        + 0.20 * aggressive
        + 0.20 * speeding_norm
        + 0.15 * battery_risk
        + 0.15 * coolant_flag
        + 0.10 * night_flag
    )
    risk_score = np.clip(risk_score, 0.0, 1.0)

    result['LightGBM_Risk_Score'] = risk_score.astype(float)
    if 'Risk_Score' not in result.columns or result['Risk_Score'].isna().all():
        result['Risk_Score'] = result['LightGBM_Risk_Score']
    if 'Calculated_Risk_Score' not in result.columns or result['Calculated_Risk_Score'].isna().all():
        result['Calculated_Risk_Score'] = result['LightGBM_Risk_Score']
    if 'Risk_Band' not in result.columns:
        result['Risk_Band'] = result['Risk_Score'].apply(_risk_band_from_score)
    return result


def _ensure_lightgbm_scored(df: pd.DataFrame, model_path: Path | str = LIGHTGBM_MODEL_PATH) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    # If trained models are disabled, use deterministic fallback scoring.
    if not USE_TRAINED_MODELS:
        try:
            return _fallback_lightgbm_scoring(df)
        except Exception:
            return df
    if "Calculated_Risk_Score" in df.columns and "LightGBM_Risk_Score" in df.columns:
        return df

    # Try to use a recent background-scored dataset to avoid re-running the model
    try:
        cached = get_latest_scored_dataframe(max_age_seconds=max(30.0, _SCORER_STATE.get("interval_seconds", 15.0) * 3))
        if cached is not None and not cached.empty:
            result = df.copy()
            if "LightGBM_Risk_Score" in cached.columns:
                if "Plate" in result.columns and "Plate" in cached.columns:
                    lookup = cached.set_index("Plate")["LightGBM_Risk_Score"]
                    result["LightGBM_Risk_Score"] = result["Plate"].map(lookup).astype(float)
                else:
                    try:
                        result["LightGBM_Risk_Score"] = cached["LightGBM_Risk_Score"].reindex(result.index).astype(float)
                    except Exception:
                        result["LightGBM_Risk_Score"] = np.nan

            if "Risk_Score" not in result.columns or result["Risk_Score"].isna().all():
                result["Risk_Score"] = result.get("LightGBM_Risk_Score")
            if "Calculated_Risk_Score" not in result.columns or result["Calculated_Risk_Score"].isna().all():
                result["Calculated_Risk_Score"] = result.get("LightGBM_Risk_Score")
            if "Risk_Band" not in result.columns:
                if "Risk_Score" in result.columns:
                    result["Risk_Band"] = result["Risk_Score"].apply(_risk_band_from_score)
                else:
                    result["Risk_Band"] = np.nan

            if result["LightGBM_Risk_Score"].notna().any():
                return result
    except Exception:
        # Fall back to on-demand scoring if cache access fails
        pass

    model_path = Path(model_path)
    if not model_path.exists():
        return _fallback_lightgbm_scoring(df)

    model = _load_lightgbm_model(model_path)
    if model is None:
        return _fallback_lightgbm_scoring(df)

    try:
        import train_hybrid_risk_model as trainer
    except Exception:
        return _fallback_lightgbm_scoring(df)

    encoded, _ = trainer.prepare_features(df)
    if hasattr(model, "feature_name_"):
        feature_names = list(model.feature_name_)
    else:
        try:
            feature_names = list(model.booster_.feature_name())
        except Exception:
            feature_names = list(encoded.columns)

    encoded = encoded.reindex(columns=feature_names, fill_value=0.0)
    scored = trainer.score_dataset(model, encoded, feature_names)

    result = df.copy()
    result["LightGBM_Risk_Score"] = scored["Predicted_Risk_Score"].astype(float)
    if "Risk_Score" not in result.columns:
        result["Risk_Score"] = result["LightGBM_Risk_Score"]
    if "Calculated_Risk_Score" not in result.columns:
        result["Calculated_Risk_Score"] = result["LightGBM_Risk_Score"]
    if "Risk_Band" not in result.columns:
        result["Risk_Band"] = result["Risk_Score"].apply(_risk_band_from_score)
    return result


def _apply_ppo_policy(df: pd.DataFrame, policy_path: Path | str = PPO_POLICY_PATH) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    # Simple deterministic action mapping when models are disabled
    if not USE_TRAINED_MODELS:
        result = df.copy()
        # Ensure we have a risk band
        if 'Risk_Band' in result.columns:
            bands = result['Risk_Band'].fillna('Medium')
        else:
            bands = result.get('Calculated_Risk_Score', pd.Series(0.5, index=result.index)).apply(_risk_band_from_score)
        result['PPO_Action'] = bands.map(PPO_ACTION_BY_RISK_BAND).fillna('standard_offer')
        return result
    policy_path = Path(policy_path)
    ppo_model = _load_ppo_policy(policy_path)
    if ppo_model is None:
        result = df.copy()
        bands = result.get('Risk_Band', result.get('Calculated_Risk_Score', pd.Series(0.5, index=result.index)).apply(_risk_band_from_score)).fillna('Medium')
        result['PPO_Action'] = bands.map(PPO_ACTION_BY_RISK_BAND).fillna('standard_offer')
        return result

    try:
        import train_hybrid_risk_model as trainer
        encoded, feature_columns = trainer.prepare_features(df)
        if not feature_columns:
            return df

        observations = _build_ppo_observations(encoded, feature_columns)
        actions, _ = ppo_model.predict(observations, deterministic=True)
        action_series = pd.Series(actions.astype(int), index=df.index).map(
            {i: label for i, label in enumerate(PPO_ACTION_LABELS)}
        )
        result = df.copy()
        result["PPO_Action"] = action_series.fillna("standard_offer")
        return result
    except Exception:
        result = df.copy()
        bands = result.get('Risk_Band', result.get('Calculated_Risk_Score', pd.Series(0.5, index=result.index)).apply(_risk_band_from_score)).fillna('Medium')
        result['PPO_Action'] = bands.map(PPO_ACTION_BY_RISK_BAND).fillna('standard_offer')
        return result


def get_base_premium_for_vehicle(vehicle_type: str) -> float:
    """Get base annual premium USD for a vehicle type."""
    return BASE_PREMIUM.get(vehicle_type, 8000)

def calculate_risk_multiplier(risk_band: str) -> float:
    """Get risk multiplier based on risk band."""
    risk_mult = {"Low": 0.62, "Medium": 1.00, "High": 1.42, "Critical": 1.95}
    return risk_mult.get(risk_band, 1.0)


def calculate_daily_premium_from_risk(risk_score: np.ndarray, vehicle_type: np.ndarray) -> np.ndarray:
    """
    Calculate DAILY premium based on current risk score and vehicle type.
    This updates in real-time as risk scores change.
    Daily premium range: $1-$13 USD per day (translates to $30-$400 monthly).
    
    Args:
        risk_score: Array of current risk scores (0.0 - 1.0)
        vehicle_type: Array of vehicle types
    
    Returns:
        Array of daily premiums in USD
    """
    # Base annual premium for each vehicle
    base_annual = np.array([BASE_PREMIUM.get(vtype, 600.0) for vtype in vehicle_type], dtype=float)
    
    # Risk multiplier based on score (0.62 to 1.95)
    risk_multiplier = 0.62 + (np.asarray(risk_score) * 1.33)
    
    # Calculate daily premium: (base_annual * risk_multiplier) / 12 / 30
    daily_premium = (base_annual * risk_multiplier) / 12.0 / 30.0
    
    return np.clip(daily_premium, 1.0, 13.33)  # $1 to $13.33 per day


def calculate_monthly_premiums(scored: pd.DataFrame, usd_to_zig_rate: float = USD_TO_ZIG_RATE_DEFAULT) -> tuple:
    """
    Calculate daily premiums based on current risk (real-time updates).
    Monthly premiums accumulate daily once per 24 hours.
    Premium range: $1-$13 USD daily, accumulating to $30-$400 monthly.
    
    Returns tuple of (daily_premium_usd, monthly_premium_usd, daily_premium_zig, monthly_premium_zig).
    """
    vehicle_type = scored["Type"].fillna("Sedan")
    risk_score = scored.get("Calculated_Risk_Score", pd.Series(0.5)).fillna(0.5).astype(float)
    
    # Scaled base annual premium for $30-$400 monthly range
    base_annual = vehicle_type.map(BASE_PREMIUM).fillna(600.0).astype(float)
    
    # Risk multiplier based on current risk score (0.62 to 1.95)
    risk_multiplier = 0.62 + (risk_score * 1.33)
    
    # Daily premium: (base_annual * risk_multiplier) / 12 / 30
    daily_usd = (base_annual * risk_multiplier / 12.0 / 30.0).clip(lower=1.0, upper=13.33)
    
    # Monthly premiums from "Monthly_Premium_USD" column (accumulated) or default to 0
    if "Monthly_Premium_USD" in scored.columns:
        monthly_usd = scored["Monthly_Premium_USD"].astype(float)
    else:
        monthly_usd = pd.Series([0.0] * len(scored))

    daily_zig = daily_usd * usd_to_zig_rate if usd_to_zig_rate > 0 else daily_usd
    monthly_zig = monthly_usd * usd_to_zig_rate if usd_to_zig_rate > 0 else monthly_usd

    return daily_usd.values, monthly_usd.values, daily_zig.values, monthly_zig.values

def calculate_monthly_premium(row: pd.Series) -> float:
    """Calculate monthly premium (USD) from risk score and vehicle type.

    This uses the same annual base premium mapping as the daily premium
    calculation, then scales to a monthly amount matching the target
    $30-$400 monthly premium band used across the product.
    """
    risk_score = float(row.get('Calculated_Risk_Score', 0.5) or 0.5)
    vehicle_type = str(row.get('Type', 'Sedan') or 'Sedan')
    base_annual = float(BASE_PREMIUM.get(vehicle_type, 600.0))
    risk_multiplier = 0.62 + (risk_score * 1.33)
    monthly_premium = (base_annual * risk_multiplier) / 12.0
    return float(np.clip(monthly_premium, 30.0, 400.0))

def calculate_claim_probability(row: pd.Series) -> float:
    """Estimate claim probability from risk score (1% – 45% range for Zimbabwe profitability)."""
    risk_score = row.get('Calculated_Risk_Score', 0.5)
    return 0.01 + (risk_score * 0.44)

def calculate_profitability_metrics(scored: pd.DataFrame, usd_to_zig_rate: float = USD_TO_ZIG_RATE_DEFAULT) -> pd.DataFrame:
    """
    Calculate profitability metrics for all vehicles.
    Includes daily/monthly premiums, expected claims, expenses, profit estimates, and ZIG conversions.
    """
    df = _ensure_lightgbm_scored(scored.copy())
    df["Calculated_Risk_Score"] = df.get("Calculated_Risk_Score", df.get("LightGBM_Risk_Score", 0.5)).fillna(0.5).astype(float)
    df["Risk_Band"] = df.get("Risk_Band", df["Calculated_Risk_Score"].apply(_risk_band_from_score)).astype(str)
    df["Base_Price_USD"] = pd.to_numeric(df.get("Base_Price_USD", 0.0), errors="coerce").fillna(0.0)
    vehicle_type = df.get("Type", pd.Series("Sedan", index=df.index)).fillna("Sedan").astype(str)
    risk_score = df["Calculated_Risk_Score"].fillna(0.5).astype(float).clip(0.0, 1.0)
    base_annual = vehicle_type.map(BASE_PREMIUM).fillna(600.0).astype(float)
    computed_daily_premium = pd.Series(
        calculate_daily_premium_from_risk(risk_score.to_numpy(), vehicle_type.to_numpy()),
        index=df.index,
        dtype=float,
    )
    computed_monthly_premium = ((base_annual * (0.62 + (risk_score * 1.33))) / 12.0).clip(lower=30.0, upper=400.0)

    # Ensure daily and monthly premiums exist
    existing_daily = pd.to_numeric(df.get("Daily_Premium_USD", pd.Series(np.nan, index=df.index)), errors="coerce")
    existing_monthly = pd.to_numeric(df.get("Monthly_Premium_USD", pd.Series(np.nan, index=df.index)), errors="coerce")
    df["Daily_Premium_USD"] = existing_daily.fillna(computed_daily_premium).astype(float)
    df["Monthly_Premium_USD"] = existing_monthly.fillna(computed_monthly_premium).astype(float)

    # Calculate expected claims from the current risk score / base price model.
    claim_probability = (0.01 + (risk_score * 0.44)).clip(lower=0.01, upper=0.45)
    df["Expected_Claim"] = (
        df["Base_Price_USD"].astype(float) * claim_probability * AVERAGE_CLAIM_SEVERITY_RATIO / 12.0
    ).astype(float)

    df["Expected_Claim_USD"] = df["Expected_Claim"].astype(float)
    df["USD_to_ZIG_Interbank_Rate"] = float(usd_to_zig_rate)
    df["Daily_Premium_ZIG"] = _usd_to_zig(df["Daily_Premium_USD"].fillna(0.0), usd_to_zig_rate)
    df["Monthly_Premium_ZIG"] = _usd_to_zig(df["Monthly_Premium_USD"].fillna(0.0), usd_to_zig_rate)

    # Recommended actions
    public_or_commercial = (df["Usage"] == "Commercial") | df["Type"].isin(PUBLIC_VEHICLE_TYPES)
    action_codes = np.where(
        df["Risk_Band"] == "Critical", 3,
        np.where(df["Risk_Band"] == "High", 2,
                np.where(df["Risk_Band"] == "Medium", 1, 0))
    )
    df["Recommended_Action"] = action_codes

    control_cost_usd = ACTION_MONTHLY_CONTROL_COSTS[action_codes]
    fixed_admin_cost_usd = np.where(public_or_commercial, 0.5, 0.25)
    variable_expense_ratio = np.where(public_or_commercial, COMMERCIAL_EXPENSE_RATIO, PRIVATE_EXPENSE_RATIO)
    variable_expense_ratio = variable_expense_ratio + np.select(
        [df["Risk_Band"] == "Critical", df["Risk_Band"] == "High", df["Risk_Band"] == "Medium"],
        [0.02, 0.01, 0.005], default=0.0,
    )

    expected_claims_usd = df["Expected_Claim_USD"].astype(float).to_numpy()
    monthly_premium_usd = df["Monthly_Premium_USD"].astype(float).to_numpy()
    variable_expenses_usd = monthly_premium_usd * variable_expense_ratio
    total_expenses_usd = variable_expenses_usd + control_cost_usd + fixed_admin_cost_usd
    expected_profit_usd = monthly_premium_usd - expected_claims_usd - total_expenses_usd
    expected_profit_usd = np.maximum(expected_profit_usd, 0.0)

    df["Variable_Expenses_USD"] = np.round(variable_expenses_usd, 2)
    df["Control_Cost_USD"] = np.round(control_cost_usd, 2)
    df["Underwriting_Expenses_USD"] = np.round(total_expenses_usd, 2)
    df["Underwriting_Profit_USD"] = np.round(expected_profit_usd, 2)

    # Convert all USD columns to ZIG using the selected interbank rate.
    for usd_column in [col for col in df.columns if col.endswith("_USD")]:
        zig_column = usd_column.replace("_USD", "_ZIG")
        df[zig_column] = _usd_to_zig(df[usd_column].fillna(0.0), usd_to_zig_rate)

    return df

def build_insurance_premium_schedule(df: pd.DataFrame, usd_to_zig_rate: float = USD_TO_ZIG_RATE_DEFAULT) -> pd.DataFrame:
    """
    Build a filterable insurance premium schedule from the preprocessed dataset.
    If the incoming columns already match the preprocessed dataset, preserve that dataset structure and enrich it.
    Every USD-denominated field gets a matching ZIG conversion at the interbank rate.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    required_columns = {
        "Risk_Score",
        "Risk_Band",
        "Base_Price_USD",
        "Expected_Claim_USD",
        "Daily_Premium_USD",
        "Monthly_Premium_USD",
        "Underwriting_Expenses_USD",
        "Underwriting_Profit_USD",
    }
    if required_columns.issubset(df.columns):
        df = df.copy()
    else:
        df = calculate_profitability_metrics(df, usd_to_zig_rate=usd_to_zig_rate)

    live_df = build_live_vehicle_risk_table(df)

    # Use live risk table values for overlapping risk columns so the schedule aligns with the live view.
    if 'Plate' in live_df.columns and 'Plate' in df.columns:
        live_lookup = live_df.drop_duplicates(subset=['Plate'], keep='last').set_index('Plate')
        schedule_indexed = df.set_index('Plate', drop=False)
        overlap = schedule_indexed.index.intersection(live_lookup.index)
        for column in ['Make', 'Model', 'Risk_Score', 'Risk_Band']:
            if column not in live_lookup.columns:
                continue
            aligned_live_values = live_lookup[column].reindex(schedule_indexed.index)
            if column not in schedule_indexed.columns:
                schedule_indexed[column] = aligned_live_values
                continue
            if overlap.empty:
                continue
            try:
                schedule_indexed.loc[overlap, column] = aligned_live_values.loc[overlap].to_numpy()
            except (TypeError, ValueError):
                schedule_indexed[column] = schedule_indexed[column].astype(object)
                schedule_indexed.loc[overlap, column] = aligned_live_values.loc[overlap].to_numpy()
        df = schedule_indexed.reset_index(drop=True)

    usd_columns = [column for column in df.columns if column.endswith("_USD")]
    for usd_column in usd_columns:
        zig_column = usd_column.replace("_USD", "_ZIG")
        if zig_column not in df.columns or usd_column in {"Expected_Claim_USD", "Daily_Premium_USD", "Monthly_Premium_USD", "Underwriting_Expenses_USD", "Underwriting_Profit_USD"}:
            df[zig_column] = _usd_to_zig(pd.to_numeric(df[usd_column], errors="coerce").fillna(0.0), usd_to_zig_rate)

    requested_columns = [
        'Plate', 'Type', 'City', 'Make', 'Model', 'Usage',
        'Risk_Score', 'Risk_Band', 'Recommended_Action',
        'Base_Price_USD', 'Expected_Claim_USD',
        'Daily_Premium_USD', 'Monthly_Premium_USD',
        'Underwriting_Expenses_USD', 'Underwriting_Profit_USD',
        'Expected_Claim_ZIG', 'Daily_Premium_ZIG', 'Monthly_Premium_ZIG',
        'Underwriting_Expenses_ZIG', 'Underwriting_Profit_ZIG',
    ]

    available_cols = [col for col in requested_columns if col in df.columns]
    schedule = df[available_cols].copy()

    round_map = {
        'Base_Price_USD': 2,
        'Expected_Claim_USD': 2,
        'Daily_Premium_USD': 2,
        'Monthly_Premium_USD': 2,
        'Underwriting_Expenses_USD': 2,
        'Underwriting_Profit_USD': 2,
        'Expected_Claim_ZIG': 2,
        'Daily_Premium_ZIG': 2,
        'Monthly_Premium_ZIG': 2,
        'Underwriting_Expenses_ZIG': 2,
        'Underwriting_Profit_ZIG': 2,
        'Risk_Score': 2,
    }
    schedule = schedule.round({k: v for k, v in round_map.items() if k in schedule.columns})
    return schedule

def build_portfolio_profitability_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build portfolio profitability analysis table."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    total_monthly_premium_usd = df['Monthly_Premium_USD'].sum() if 'Monthly_Premium_USD' in df.columns else 0.0
    total_expected_claim_usd = df['Expected_Claim_USD'].sum() if 'Expected_Claim_USD' in df.columns else 0.0
    total_expenses_usd = df['Underwriting_Expenses_USD'].sum() if 'Underwriting_Expenses_USD' in df.columns else 0.0
    total_vehicles = len(df)
    total_profit_usd = total_monthly_premium_usd - total_expected_claim_usd - total_expenses_usd
    avg_premium_usd = (total_monthly_premium_usd / total_vehicles) if total_vehicles > 0 else 0.0
    avg_profit_usd = (total_profit_usd / total_vehicles) if total_vehicles > 0 else 0.0
    profit_margin = (total_profit_usd / total_monthly_premium_usd * 100) if total_monthly_premium_usd > 0 else 0.0

    total_monthly_premium_zig = df['Monthly_Premium_ZIG'].sum() if 'Monthly_Premium_ZIG' in df.columns else 0.0
    total_expected_claim_zig = df['Expected_Claim_ZIG'].sum() if 'Expected_Claim_ZIG' in df.columns else 0.0
    total_expenses_zig = df['Underwriting_Expenses_ZIG'].sum() if 'Underwriting_Expenses_ZIG' in df.columns else 0.0
    total_profit_zig = total_monthly_premium_zig - total_expected_claim_zig - total_expenses_zig
    avg_profit_zig = (total_profit_zig / total_vehicles) if total_vehicles > 0 else 0.0

    summary = pd.DataFrame({
        'Metric': [
            'Total Vehicles',
            'Total Monthly Premium ($)',
            'Total Monthly Premium (ZIG)',
            'Total Expected Claims ($)',
            'Total Expected Claims (ZIG)',
            'Total Expenses ($)',
            'Total Expenses (ZIG)',
            'Total Monthly Profit ($)',
            'Total Monthly Profit (ZIG)',
            'Average Premium per Vehicle ($)',
            'Average Profit per Vehicle ($)',
            'Average Profit per Vehicle (ZIG)',
            'Profit Margin (%)',
        ],
        'Value': [
            len(df),
            total_monthly_premium_usd,
            total_monthly_premium_zig,
            total_expected_claim_usd,
            total_expected_claim_zig,
            total_expenses_usd,
            total_expenses_zig,
            total_profit_usd,
            total_profit_zig,
            avg_premium_usd,
            avg_profit_usd,
            avg_profit_zig,
            profit_margin,
        ]
    })
    
    summary = summary.round({
        'Value': 2
    })
    
    return summary

def build_risk_band_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Build risk band analysis table."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    risk_bands = df['Risk_Band'].unique()
    analysis_data = []
    
    for band in ['Low', 'Medium', 'High', 'Critical']:
        if band in risk_bands:
            band_df = df[df['Risk_Band'] == band]
            analysis_data.append({
                'Risk_Band': band,
                'Vehicle_Count': len(band_df),
                'Avg_Premium': band_df['Monthly_Premium_USD'].mean(),
                'Total_Premium': band_df['Monthly_Premium_USD'].sum(),
                'Avg_Profit': band_df['Underwriting_Profit_USD'].mean(),
                'Total_Profit': band_df['Underwriting_Profit_USD'].sum(),
            })
    
    analysis_df = pd.DataFrame(analysis_data)
    analysis_df = analysis_df.round({
        'Avg_Premium': 2,
        'Total_Premium': 2,
        'Avg_Profit': 2,
        'Total_Profit': 2
    })
    
    return analysis_df

def build_city_level_risk_premium(df: pd.DataFrame) -> pd.DataFrame:
    """Build city-level risk and premium analysis."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    city_analysis = df.groupby('City').agg({
        'Plate': 'count',
        'Calculated_Risk_Score': 'mean',
        'Monthly_Premium_USD': 'mean',
        'Risk_Band': lambda x: x.value_counts().index[0] if len(x) > 0 else 'Unknown'
    }).reset_index()
    
    city_analysis.columns = ['City', 'Vehicle_Count', 'Avg_Risk_Score', 'Avg_Premium', 'Risk_Level']
    city_analysis = city_analysis.round({
        'Avg_Risk_Score': 3,
        'Avg_Premium': 2
    })
    
    return city_analysis

def layout_with_text_color(layout: dict, text_color: str = "black") -> dict:
    """Update layout with text color."""
    new_layout = layout.copy()
    new_layout['font']['color'] = text_color
    return new_layout

def _derive_live_policy_fields(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    result = normalize_shared_risk_columns(df)
    result = _jitter_risk_score(result)
    result = normalize_shared_risk_columns(result, source_column='Risk_Score')

    result = _apply_ppo_policy(result)

    if 'PPO_Action' not in result.columns or result['PPO_Action'].isna().all():
        if 'Risk_Band' in result.columns:
            result['PPO_Action'] = result['Risk_Band'].map(PPO_ACTION_BY_RISK_BAND).fillna('standard_offer')
        else:
            result['PPO_Action'] = 'standard_offer'
    else:
        result['PPO_Action'] = result['PPO_Action'].fillna('standard_offer')

    result['Premium_Multiplier'] = result['PPO_Action'].map(PPO_ACTION_PREMIUM_MULTIPLIERS).fillna(1.00)

    return result


class HybridPricingInferenceService:
    def __init__(
        self,
        artifacts: ModelArtifactPaths | None = None,
        max_workers: int | None = None,
    ) -> None:
        self.artifacts = artifacts or ModelArtifactPaths()
        self.max_workers = max_workers or max(2, min((os.cpu_count() or 4) * 2, 16))
        self._risk_model = None
        self._ppo_policy = None
        self._ppo_scaler = None

    def load_artifacts(self) -> None:
        if self._risk_model is None:
            self._risk_model = _load_lightgbm_model(self.artifacts.lightgbm_model_path)

        resolved_policy_path = _resolve_policy_artifact_path(self.artifacts.ppo_policy_path)
        if self._ppo_policy is None:
            self._ppo_policy = _load_ppo_policy(resolved_policy_path)

        if self._ppo_scaler is None:
            self._ppo_scaler = _load_ppo_scaler(self.artifacts.ppo_scaler_path)

    @property
    def trained_models_available(self) -> bool:
        self.load_artifacts()
        return self._risk_model is not None and self._ppo_policy is not None and self._ppo_scaler is not None

    def score_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame() if df is None else df.copy()

        result = _ensure_lightgbm_scored(df.copy(), model_path=self.artifacts.lightgbm_model_path)
        if "Risk_Score" not in result.columns and "LightGBM_Risk_Score" in result.columns:
            result["Risk_Score"] = result["LightGBM_Risk_Score"]
        if "Calculated_Risk_Score" not in result.columns and "Risk_Score" in result.columns:
            result["Calculated_Risk_Score"] = result["Risk_Score"]
        if "Risk_Band" not in result.columns and "Calculated_Risk_Score" in result.columns:
            result["Risk_Band"] = result["Calculated_Risk_Score"].apply(_risk_band_from_score)
        return result

    def apply_policy(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame() if df is None else df.copy()

        result = self.score_risk(df) if "Calculated_Risk_Score" not in df.columns else df.copy()
        result = _apply_ppo_policy(result, policy_path=self.artifacts.ppo_policy_path)
        if "PPO_Action" not in result.columns or result["PPO_Action"].isna().all():
            bands = result.get("Risk_Band", pd.Series("Medium", index=result.index)).fillna("Medium").astype(str)
            result["PPO_Action"] = bands.map(PPO_ACTION_BY_RISK_BAND).fillna("standard_offer")
        else:
            result["PPO_Action"] = result["PPO_Action"].fillna("standard_offer")
        result["Premium_Multiplier"] = result["PPO_Action"].map(PPO_ACTION_PREMIUM_MULTIPLIERS).fillna(1.0).astype(float)
        return result

    def score_pricing(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame() if df is None else df.copy()

        result = self.apply_policy(self.score_risk(df))
        risk_score = pd.to_numeric(
            result.get("Calculated_Risk_Score", result.get("Risk_Score", pd.Series(0.5, index=result.index))),
            errors="coerce",
        ).fillna(0.5).clip(0.0, 1.0)
        vehicle_type = result.get("Type", pd.Series("Sedan", index=result.index)).fillna("Sedan").astype(str)
        base_annual = vehicle_type.map(BASE_PREMIUM).fillna(600.0).astype(float)
        risk_multiplier = 0.62 + (risk_score * 1.33)

        risk_based_monthly = np.clip((base_annual * risk_multiplier) / 12.0, 30.0, 400.0)
        risk_based_daily = np.clip(risk_based_monthly / 30.0, 1.0, 13.33)
        adjusted_monthly = risk_based_monthly * result["Premium_Multiplier"].astype(float)
        adjusted_daily = risk_based_daily * result["Premium_Multiplier"].astype(float)

        base_price = pd.to_numeric(
            result.get("Base_Price_USD", pd.Series(6000.0, index=result.index)),
            errors="coerce",
        ).fillna(6000.0)
        claim_probability = 0.01 + (risk_score * 0.44)
        expected_claim = (base_price * claim_probability * AVERAGE_CLAIM_SEVERITY_RATIO) / 12.0

        result["Risk_Based_Daily_Premium_USD"] = np.round(risk_based_daily, 2)
        result["Risk_Based_Monthly_Premium_USD"] = np.round(risk_based_monthly, 2)
        result["Policy_Adjusted_Daily_Premium_USD"] = np.round(adjusted_daily, 2)
        result["Policy_Adjusted_Monthly_Premium_USD"] = np.round(adjusted_monthly, 2)
        result["Expected_Claim_USD"] = np.round(expected_claim, 2)
        result["Decision_Status"] = np.where(result["PPO_Action"] == "decline", "declined", "quoted")
        result["Decision_Timestamp"] = pd.Timestamp.utcnow().isoformat()
        return result

    def score_frame_concurrently(
        self,
        df: pd.DataFrame,
        partition_size: int = 1000,
        max_workers: int | None = None,
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame() if df is None else df.copy()
        if len(df) <= partition_size:
            return self.score_pricing(df)

        chunks = [df.iloc[start : start + partition_size].copy() for start in range(0, len(df), partition_size)]
        worker_count = max_workers or self.max_workers
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            scored_chunks = list(executor.map(self.score_pricing, chunks))
        return pd.concat(scored_chunks, axis=0, ignore_index=True)

    def score_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        frame = pd.DataFrame(records)
        scored = self.score_pricing(frame)
        return scored.replace({np.nan: None}).to_dict(orient="records")

    def build_spark_output_schema(self, input_schema):
        from pyspark.sql import types as T

        existing_names = {field.name for field in input_schema.fields}
        extra_fields = [
            T.StructField("LightGBM_Risk_Score", T.DoubleType(), True),
            T.StructField("Risk_Score", T.DoubleType(), True),
            T.StructField("Calculated_Risk_Score", T.DoubleType(), True),
            T.StructField("Risk_Band", T.StringType(), True),
            T.StructField("PPO_Action", T.StringType(), True),
            T.StructField("Premium_Multiplier", T.DoubleType(), True),
            T.StructField("Risk_Based_Daily_Premium_USD", T.DoubleType(), True),
            T.StructField("Risk_Based_Monthly_Premium_USD", T.DoubleType(), True),
            T.StructField("Policy_Adjusted_Daily_Premium_USD", T.DoubleType(), True),
            T.StructField("Policy_Adjusted_Monthly_Premium_USD", T.DoubleType(), True),
            T.StructField("Expected_Claim_USD", T.DoubleType(), True),
            T.StructField("Decision_Status", T.StringType(), True),
            T.StructField("Decision_Timestamp", T.StringType(), True),
        ]
        merged_fields = list(input_schema.fields) + [field for field in extra_fields if field.name not in existing_names]
        return T.StructType(merged_fields)

    def build_spark_mapper(self) -> Callable[[Any], Any]:
        artifacts = self.artifacts

        def _mapper(iterator):
            local_service = HybridPricingInferenceService(artifacts=artifacts, max_workers=1)
            for pdf in iterator:
                yield local_service.score_pricing(pdf)

        return _mapper

    def score_spark_dataframe(self, spark_df):
        return spark_df.mapInPandas(
            self.build_spark_mapper(),
            schema=self.build_spark_output_schema(spark_df.schema),
        )


def build_fastapi_app(service: HybridPricingInferenceService | None = None):
    from fastapi import FastAPI
    from fastapi.concurrency import run_in_threadpool
    from pydantic import BaseModel, Field

    inference_service = service or HybridPricingInferenceService()

    class DriverRecord(BaseModel):
        payload: dict[str, Any] = Field(default_factory=dict)

    class BatchRequest(BaseModel):
        records: list[dict[str, Any]] = Field(default_factory=list)

    app = FastAPI(
        title="Insurtech Real-Time Pricing Service",
        version="1.0.0",
        description="Concurrent LightGBM + PPO pricing decisions for motor telematics portfolios.",
    )

    @app.on_event("startup")
    def _startup() -> None:
        inference_service.load_artifacts()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "trained_models_available": inference_service.trained_models_available,
            "max_workers": inference_service.max_workers,
        }

    @app.post("/score")
    async def score_one(request: DriverRecord) -> dict[str, Any]:
        records = await run_in_threadpool(inference_service.score_records, [request.payload])
        return records[0] if records else {}

    @app.post("/score/batch")
    async def score_batch(request: BatchRequest) -> dict[str, Any]:
        records = await run_in_threadpool(inference_service.score_records, request.records)
        return {
            "count": len(records),
            "records": records,
        }

    return app


def create_app():
    return build_fastapi_app(HybridPricingInferenceService())


def build_live_vehicle_risk_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build live vehicle risk table for dashboard."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = _ensure_lightgbm_scored(df)
    df = _derive_live_policy_fields(df)
    columns = [
        'Plate', 'Type', 'City', 'Make', 'Model', 'Risk_Score', 'Risk_Band', 'PPO_Action', 'Premium_Multiplier',
        'Speed_kmh', 'Speeding_Excess_kmh', 'Acceleration_mps2',
        'Harsh_Events_Per_Day', 'Aggressive_Driving_Score', 'Weather_Risk_Score', 'Road_Type_Risk_Score',
        'Time_of_Day_Risk_Score', 'Night_Driving_Flag', 'Fatigue_Risk_Score', 'Recent_Avg_Speed',
        'Recent_Harsh_Events', 'Recent_Night_Distance', 'Engine_CC', 'Year'
    ]
    selected_columns = [col for col in columns if col in df.columns]
    return df[selected_columns].copy()
