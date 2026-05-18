from __future__ import annotations

import argparse
import contextlib
import io
import json
import joblib
import math
import os
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import scipy.stats as stats
import shap
import lightgbm as lgb
from gymnasium import spaces
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

warnings.filterwarnings(
    "ignore",
    message=r".*use_label_encoder.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*LightGBM binary classifier with TreeExplainer shap values output has changed.*",
    category=UserWarning,
)

if hasattr(gym.logger, "set_level"):
    gym.logger.set_level(40)
elif hasattr(gym.logger, "min_level"):
    gym.logger.min_level = 40

DEFAULT_DATA_PATH = Path(r"C:\Users\USER\OneDrive\Desktop\Insurtech\preprocessed_telemetry_dataset.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts") / "hybrid_risk"
MODEL_FILENAME = "lightgbm_risk_model.pkl"
SCALER_FILENAME = "ppo_observation_scaler.pkl"

ACTION_NAMES = {0: "standard_offer", 1: "monitoring", 2: "surcharge", 3: "decline"}
PREMIUM_MULTIPLIERS = np.array([1.00, 1.10, 1.25, 0.00], dtype=np.float32)
LOSS_MULTIPLIERS = np.array([1.00, 0.92, 0.78, 0.00], dtype=np.float32)
INTERVENTION_COSTS = np.array([0.0, 20.0, 40.0, 15.0], dtype=np.float32)

ID_COLUMNS = {"Plate", "Trip_ID", "Last_Update"}
TARGET_COLUMNS = {"Risk_Label", "Risk_Score", "Claim_Probability", "Claim_Label"}


def default_ppo_env_count() -> int:
    return max(1, min(os.cpu_count() or 1, 8))


@dataclass(frozen=True)
class ParallelPPOConfig:
    num_envs: int = 1
    vec_mode: str = "dummy"

    def resolved(self, dataset_size: int) -> "ParallelPPOConfig":
        effective_envs = max(1, min(int(self.num_envs), max(1, dataset_size)))
        if effective_envs == 1:
            return ParallelPPOConfig(num_envs=1, vec_mode="dummy")
        return ParallelPPOConfig(num_envs=effective_envs, vec_mode=self.vec_mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hybrid LightGBM + PPO risk pipeline.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to the preprocessed telemetry CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for model artifacts.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Fraction of rows held out for test.")
    parser.add_argument("--val-size", type=float, default=0.20, help="Fraction of training rows held out for validation.")
    parser.add_argument("--ppo-timesteps", type=int, default=500000, help="Number of PPO timesteps.")
    parser.add_argument(
        "--ppo-envs",
        type=int,
        default=default_ppo_env_count(),
        help="Number of parallel PPO environments for experience collection.",
    )
    parser.add_argument(
        "--ppo-vec-mode",
        type=str,
        choices=["dummy", "subproc"],
        default="subproc" if default_ppo_env_count() > 1 else "dummy",
        help="Vectorized PPO environment backend.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def minmax_scale(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    min_val = float(values.min())
    max_val = float(values.max())
    scale = max(max_val - min_val, 1e-6)
    return ((values - min_val) / scale).clip(0.0, 1.0)


def bounded_scale(series: pd.Series, quantile: float = 0.95) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    upper = float(values.quantile(quantile))
    upper = max(upper, 1e-6)
    return (values / upper).clip(0.0, 1.0)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def synthesize_targets(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    if "Status" in df.columns:
        df = df[df["Status"].astype(str).str.upper() == "DRIVING"].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    harsh_total = df.get("Total_Harsh_Events_per_Day")
    if harsh_total is None:
        harsh_total = (
            df.get("Harsh_Brake_Count", 0).fillna(0.0)
            + df.get("Harsh_Accel_Count", 0).fillna(0.0)
            + df.get("Harsh_Corner_Count", 0).fillna(0.0)
        )

    df["RPM_per_Speed"] = df.get("RPM", 0.0) / (df.get("Speed_kmh", 0.0) + 0.1)
    df["Load_per_RPM"] = df.get("Engine_Load_pct", 0.0) / (df.get("RPM", 0.0) + 0.1)

    speeding = bounded_scale(df.get("Speeding_Excess_kmh", 0))
    recent_harsh = bounded_scale(df.get("Recent_Harsh_Events", 0))
    trip_pressure = bounded_scale(df.get("Trip_Duration_Hour", 0) + df.get("Trip_Distance_km", 0) / 40.0)
    engine_stress = bounded_scale(df.get("Engine_Load_pct", 0))
    efficiency = bounded_scale(df.get("Fuel_Efficiency_L_per_100km", 0))

    raw_risk = (
        0.28 * (harsh_total / (1.0 + harsh_total))
        + 0.22 * speeding
        + 0.18 * recent_harsh
        + 0.16 * trip_pressure
        + 0.10 * engine_stress
        + 0.06 * efficiency
    )

    df["Risk_Score"] = minmax_scale(raw_risk)
    df["Risk_Label"] = (df["Risk_Score"] >= df["Risk_Score"].quantile(0.65)).astype(int)
    claim_probability = np.clip(
        0.02 + 0.40 * df["Risk_Score"] + 0.08 * bounded_scale(df.get("Engine_Load_pct", 0)),
        0.01,
        0.95,
    )
    df["Claim_Probability"] = claim_probability.round(4)
    df["Claim_Label"] = rng.binomial(1, df["Claim_Probability"], size=len(df)).astype(int)

    drop_columns = [
        "Trip_ID",
        "GPS_Latitude",
        "GPS_Longitude",
        "Status",
        "Total_Harsh_Events_per_Day",
        "Harsh_Brake_Count",
        "Harsh_Accel_Count",
        "Harsh_Corner_Count",
        "Trip_Distance_km",
        "Trip_Duration_Hour",
        "Fuel_Efficiency_L_per_100km",
        "Direction",
    ]
    df = df.drop(columns=[c for c in drop_columns if c in df.columns], errors="ignore")
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    categorical_columns = [c for c in ["Road_Type", "Weather"] if c in df.columns]
    if categorical_columns:
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    columns = [
        c
        for c in df.columns
        if c not in ID_COLUMNS and c not in TARGET_COLUMNS and c != "Last_Update"
    ]
    encoded = df.copy()
    for column in encoded.columns:
        if encoded[column].dtype == object:
            encoded[column] = encoded[column].fillna("missing").astype("category").cat.codes
    numeric_columns = [c for c in columns if pd.api.types.is_numeric_dtype(encoded[c])]
    return encoded[numeric_columns].fillna(0.0), numeric_columns


def temporal_train_test_split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.Index, pd.Index]:
    if "Last_Update" in df.columns:
        temp = df.copy()
        temp["Last_Update"] = pd.to_datetime(temp["Last_Update"], errors="coerce")
        if temp["Last_Update"].notna().any():
            split_date = temp["Last_Update"].quantile(1.0 - test_size)
            train_idx = temp[temp["Last_Update"] < split_date].index
            test_idx = temp[temp["Last_Update"] >= split_date].index
            if len(train_idx) > 0 and len(test_idx) > 0:
                return train_idx, test_idx
    return train_test_split(df.index, test_size=test_size, random_state=seed, stratify=df["Risk_Label"])


def compute_classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities)),
    }


def get_continuous_features(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c not in ID_COLUMNS
        and c not in TARGET_COLUMNS
        and c != "Last_Update"
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def get_categorical_features(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c not in ID_COLUMNS
        and c not in TARGET_COLUMNS
        and c != "Last_Update"
        and (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]))
    ]


def run_shapiro_wilk_tests(df: pd.DataFrame, columns: list[str] | None = None, max_sample: int = 5000) -> dict[str, dict[str, Any]]:
    cols = columns if columns is not None else get_continuous_features(df)
    results: dict[str, dict[str, Any]] = {}
    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 3:
            results[col] = {"error": "insufficient_data", "n": int(len(series))}
            continue
        sample = series.sample(n=min(len(series), max_sample), random_state=42)
        stat, pvalue = stats.shapiro(sample)
        results[col] = {"statistic": float(stat), "pvalue": float(pvalue), "n": int(len(series))}
    return results


def run_mannwhitneyu_tests(df: pd.DataFrame, columns: list[str] | None = None, group_col: str = "Risk_Label") -> dict[str, dict[str, Any]]:
    cols = columns if columns is not None else get_continuous_features(df)
    results: dict[str, dict[str, Any]] = {}
    for col in cols:
        a = pd.to_numeric(df.loc[df[group_col] == 1, col], errors="coerce").dropna()
        b = pd.to_numeric(df.loc[df[group_col] == 0, col], errors="coerce").dropna()
        if len(a) < 3 or len(b) < 3:
            results[col] = {"error": "insufficient_group_data", "n_group1": int(len(a)), "n_group0": int(len(b))}
            continue
        stat, pvalue = stats.mannwhitneyu(a, b, alternative="two-sided")
        results[col] = {"statistic": float(stat), "pvalue": float(pvalue), "n_group1": int(len(a)), "n_group0": int(len(b))}
    return results


def run_chi2_tests(df: pd.DataFrame, columns: list[str] | None = None, target_col: str = "Risk_Label") -> dict[str, dict[str, Any]]:
    cols = columns if columns is not None else get_categorical_features(df)
    results: dict[str, dict[str, Any]] = {}
    for col in cols:
        contingency = pd.crosstab(df[col].fillna("Missing"), df[target_col])
        if contingency.size == 0 or contingency.shape[0] < 2 or contingency.shape[1] < 2:
            results[col] = {"error": "insufficient_levels", "shape": contingency.shape}
            continue
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        results[col] = {
            "chi2": float(chi2),
            "p_value": float(p),
            "dof": int(dof),
            "n_levels": int(contingency.shape[0]),
        }
    return results


def compute_spearman_correlations(df: pd.DataFrame, target_col: str = "Risk_Score", columns: list[str] | None = None) -> dict[str, dict[str, Any]]:
    cols = columns if columns is not None else get_continuous_features(df)
    results: dict[str, dict[str, Any]] = {}
    for col in cols:
        if col == target_col or col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        y = pd.to_numeric(df[target_col], errors="coerce").dropna()
        mask = x.index.intersection(y.index)
        x_valid = x.loc[mask]
        y_valid = y.loc[mask]
        if len(x_valid) < 3:
            results[col] = {"error": "insufficient_data", "n": int(len(x_valid))}
            continue
        corr, pvalue = stats.spearmanr(x_valid, y_valid)
        results[col] = {
            "correlation": float(corr),
            "p_value": float(pvalue),
            "high_correlation": bool(abs(corr) > 0.9),
            "n": int(len(x_valid)),
        }
    return results


def compute_ks_statistic(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float] | dict[str, Any]:
    pos = probabilities[y_true == 1]
    neg = probabilities[y_true == 0]
    if len(pos) < 2 or len(neg) < 2:
        return {"error": "insufficient_group_data"}
    stat, pvalue = stats.ks_2samp(pos, neg)
    return {"statistic": float(stat), "p_value": float(pvalue), "n_pos": int(len(pos)), "n_neg": int(len(neg))}


def fit_logistic_odds_ratios(X: pd.DataFrame, y: np.ndarray, top_n: int = 5) -> list[dict[str, Any]]:
    # Use a binary-compatible solver for logistic odds ratio estimation.
    # See scikit-learn LogisticRegression solver documentation:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    model = LogisticRegression(C=1e6, solver="liblinear", max_iter=2000, tol=1e-6)
    model.fit(X.fillna(0.0), y)
    coefs = model.coef_[0]
    coef_df = pd.DataFrame({"feature": X.columns, "coef": coefs})
    coef_df["odds_ratio"] = np.exp(coef_df["coef"])
    coef_df["abs_coef"] = coef_df["coef"].abs()
    top = coef_df.sort_values("abs_coef", ascending=False).head(top_n)
    return top[["feature", "coef", "odds_ratio"]].to_dict(orient="records")


def run_statistical_tests(
    raw_df: pd.DataFrame,
    X_full: pd.DataFrame,
    y_full: np.ndarray,
    y_test: np.ndarray,
    y_test_probs: np.ndarray,
) -> dict[str, Any]:
    tests: dict[str, Any] = {}
    tests["shapiro_wilk"] = run_shapiro_wilk_tests(raw_df)
    tests["mannwhitneyu"] = run_mannwhitneyu_tests(raw_df)
    tests["chi2"] = run_chi2_tests(raw_df)
    tests["spearman"] = compute_spearman_correlations(raw_df)
    tests["ks_test"] = compute_ks_statistic(y_test, y_test_probs)
    tests["logistic_odds_ratios"] = fit_logistic_odds_ratios(X_full, y_full, top_n=5)
    return tests


def train_lightgbm_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    seed: int,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.80,
        colsample_bytree=0.80,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
    )
    return model


def summarize_shap_importance(model: lgb.LGBMClassifier, X_sample: pd.DataFrame) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def build_rl_observations(df: pd.DataFrame, feature_columns: list[str], seed: int) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    observations = scaler.fit_transform(df[feature_columns].fillna(0.0).astype(float))
    return observations.astype(np.float32), scaler


def compute_minimum_action(risk_score: np.ndarray) -> np.ndarray:
    risk_score = np.clip(np.asarray(risk_score, dtype=np.float32), 0.0, 1.0)
    return np.where(
        risk_score >= 0.82,
        3,
        np.where(risk_score >= 0.66, 2, np.where(risk_score >= 0.45, 1, 0)),
    ).astype(np.int32)


class PremiumPolicyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, frame: pd.DataFrame, observations: np.ndarray, seed: int):
        super().__init__()
        self.frame = frame.reset_index(drop=True)
        self.observations = observations.astype(np.float32)
        self.seed_value = seed
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(self.observations.shape[1],), dtype=np.float32)

        self.base_premium = np.maximum(self.frame.get("Base_Price_USD", 1.0).fillna(1.0).to_numpy(dtype=np.float32), 1.0)
        self.risk_score = np.clip(self.frame["Risk_Score"].fillna(0.0).to_numpy(dtype=np.float32), 0.0, 1.0)
        self.claim_probability = np.clip(self.frame["Claim_Probability"].fillna(0.0).to_numpy(dtype=np.float32), 0.0, 1.0)
        self.risk_label = self.frame["Risk_Label"].fillna(0).astype(int).to_numpy(dtype=np.int32)
        self.min_action = compute_minimum_action(self.risk_score)
        # Low-risk drivers have min_action 0, so standard_offer is allowed.
        # Higher-risk drivers are action-masked to monitoring/surcharge/decline.
        self.indices = np.arange(len(self.frame), dtype=np.int32)
        self.current_step = 0
        self.episode_count = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        effective_seed = self.seed_value + self.episode_count if seed is None else seed
        rng = np.random.default_rng(effective_seed)
        self.indices = rng.permutation(len(self.frame)).astype(np.int32)
        self.current_step = 0
        self.episode_count += 1
        return self.observations[self.indices[self.current_step]], {}

    def step(self, action: int):
        idx = int(self.indices[self.current_step])
        reward = self._compute_reward(idx, int(action))
        self.current_step += 1
        terminated = self.current_step >= len(self.indices)
        next_obs = np.zeros_like(self.observations[0]) if terminated else self.observations[self.indices[self.current_step]]
        return next_obs, float(reward), terminated, False, {}

    def _compute_reward(self, idx: int, action: int) -> float:
        premium = self.base_premium[idx] * PREMIUM_MULTIPLIERS[action]
        expected_loss = self.base_premium[idx] * (0.05 + 0.35 * self.risk_score[idx]) * LOSS_MULTIPLIERS[action]
        reward = premium - expected_loss - INTERVENTION_COSTS[action]

        # Encourage competitive pricing for lower-risk drivers
        # and reward retention actions.
        if action == 0:
            reward += 18.0
        elif action == 1:
            reward += 12.0

        # Stronger penalty for violating the minimum action policy.
        if action < self.min_action[idx]:
            reward -= 320.0

        # Extra penalty for declining low-risk drivers.
        if action == 3 and self.risk_label[idx] == 0:
            reward -= 90.0

        # Small baseline adjustment to separate decline from other actions.
        reward += 1.5 if action != 3 else -4.0
        return reward / 100.0


def _make_policy_env(frame: pd.DataFrame, observations: np.ndarray, seed: int) -> PremiumPolicyEnv:
    return PremiumPolicyEnv(frame=frame, observations=observations, seed=seed)


def _split_parallel_env_payloads(
    frame: pd.DataFrame,
    observations: np.ndarray,
    num_envs: int,
) -> list[tuple[pd.DataFrame, np.ndarray]]:
    payloads: list[tuple[pd.DataFrame, np.ndarray]] = []
    for shard_indices in np.array_split(np.arange(len(frame)), max(1, num_envs)):
        if len(shard_indices) == 0:
            continue
        payloads.append(
            (
                frame.iloc[shard_indices].reset_index(drop=True),
                observations[shard_indices].astype(np.float32, copy=False),
            )
        )
    return payloads


def _build_vectorized_env(
    frame: pd.DataFrame,
    observations: np.ndarray,
    seed: int,
    config: ParallelPPOConfig,
) -> tuple[VecMonitor, str]:
    payloads = _split_parallel_env_payloads(frame, observations, config.num_envs)
    env_fns = [
        partial(_make_policy_env, shard_frame, shard_observations, seed + shard_index)
        for shard_index, (shard_frame, shard_observations) in enumerate(payloads)
    ]
    if len(env_fns) == 1 or config.vec_mode == "dummy":
        vec_env = DummyVecEnv(env_fns)
        return VecMonitor(vec_env), "dummy"

    start_method = "spawn" if os.name == "nt" else None
    try:
        vec_env = SubprocVecEnv(env_fns, start_method=start_method)
        return VecMonitor(vec_env), config.vec_mode
    except (PermissionError, OSError):
        warnings.warn(
            "Falling back to DummyVecEnv because subprocess PPO workers are unavailable in this environment.",
            RuntimeWarning,
        )
        vec_env = DummyVecEnv(env_fns)
        return VecMonitor(vec_env), "dummy"


def _resolve_ppo_batch_size(rollout_size: int) -> int:
    for candidate in (512, 256, 128, 64, 32, 16, 8):
        if candidate <= rollout_size:
            return candidate
    return max(1, rollout_size)


def train_ppo_policy(
    frame: pd.DataFrame,
    observations: np.ndarray,
    timesteps: int,
    seed: int,
    parallel_config: ParallelPPOConfig | None = None,
) -> tuple[PPO, dict[str, Any]]:
    config = (parallel_config or ParallelPPOConfig()).resolved(len(frame))
    env, effective_vec_mode = _build_vectorized_env(frame, observations, seed=seed, config=config)
    n_steps = max(64, min(2048, math.ceil(len(frame) / max(1, config.num_envs))))
    rollout_size = n_steps * config.num_envs
    batch_size = _resolve_ppo_batch_size(rollout_size)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=seed,
        device="cpu",
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model.learn(total_timesteps=timesteps)
    env.close()
    return model, {
        "parallel_envs": int(config.num_envs),
        "vec_mode": effective_vec_mode,
        "n_steps_per_env": int(n_steps),
        "rollout_size": int(rollout_size),
        "batch_size": int(batch_size),
    }


def evaluate_ppo_policy(frame: pd.DataFrame, actions: np.ndarray) -> dict[str, Any]:
    base_premium = np.maximum(frame.get("Base_Price_USD", 1.0).fillna(1.0).to_numpy(dtype=np.float32), 1.0)
    risk_score = np.clip(frame["Risk_Score"].fillna(0.0).to_numpy(dtype=np.float32), 0.0, 1.0)
    risk_label = frame["Risk_Label"].fillna(0).astype(int).to_numpy(dtype=np.int32)
    min_action = compute_minimum_action(risk_score)
    effective_actions = np.maximum(actions.astype(int), min_action)
    premium = base_premium * PREMIUM_MULTIPLIERS[effective_actions]
    expected_loss = base_premium * (0.05 + 0.35 * risk_score) * LOSS_MULTIPLIERS[effective_actions]
    reward = premium - expected_loss - INTERVENTION_COSTS[effective_actions]
    reward -= np.where(actions < min_action, 250.0, 0.0)
    reward -= np.where((actions == 3) & (risk_label == 0), 90.0, 0.0)
    reward += np.where(effective_actions != 3, 2.0, -5.0)
    return {
        "avg_reward": float(np.mean(reward / 100.0)),
        "median_reward": float(np.median(reward / 100.0)),
        "violation_rate": float(np.mean(actions < min_action)),
        "action_distribution": {ACTION_NAMES[int(a)]: int((actions == int(a)).sum()) for a in range(len(ACTION_NAMES))},
    }


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.ndarray, pd.Series, pd.Index)):
        return [_json_safe_value(v) for v in value.tolist()]
    if isinstance(value, pd.Categorical):
        return [_json_safe_value(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    safe_payload = _json_safe_value(payload)
    path.write_text(json.dumps(safe_payload, indent=2), encoding="utf-8")


def save_trained_model(model: lgb.LGBMClassifier, path: Path) -> None:
    joblib.dump(model, path)


def save_ppo_scaler(scaler: StandardScaler, path: Path) -> None:
    joblib.dump(scaler, path)


def load_trained_model(path: Path) -> lgb.LGBMClassifier:
    return joblib.load(path)


def load_ppo_policy(path: Path):
    from stable_baselines3 import PPO

    return PPO.load(str(path))


def score_dataset(model: lgb.LGBMClassifier, dataset: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    scored = dataset.copy()
    scored["Predicted_Risk_Score"] = model.predict_proba(scored[feature_columns])[:, 1]
    scored["Predicted_Risk_Label"] = model.predict(scored[feature_columns])
    return scored


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.data)
    dataset = synthesize_targets(dataset, seed=args.seed)

    encoded_dataset, feature_columns = prepare_features(dataset)
    X = encoded_dataset[feature_columns]
    y = dataset["Risk_Label"].astype(int)

    train_idx, test_idx = temporal_train_test_split(dataset, args.test_size, args.seed)
    X_train_full, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train_full, y_test = y.loc[train_idx], y.loc[test_idx]
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y_train_full,
    )

    lgb_model = train_lightgbm_classifier(X_train, y_train, X_val, y_val, seed=args.seed)
    save_trained_model(lgb_model, output_dir / MODEL_FILENAME)

    test_probs = lgb_model.predict_proba(X_test)[:, 1]
    metrics = compute_classification_metrics(y_test.to_numpy(), test_probs)

    shap_sample = X_test.sample(n=min(800, len(X_test)), random_state=args.seed)
    shap_importance = summarize_shap_importance(lgb_model, shap_sample)
    shap_importance.to_csv(output_dir / "lightgbm_shap_importance.csv", index=False)

    scored_dataset = score_dataset(lgb_model, encoded_dataset, feature_columns)
    scored_dataset.to_csv(output_dir / "scored_telemetry_dataset.csv", index=False)

    train_obs, scaler = build_rl_observations(encoded_dataset.loc[X_train.index], feature_columns, seed=args.seed)
    test_obs = scaler.transform(encoded_dataset.loc[X_test.index, feature_columns].fillna(0.0).astype(float)).astype(np.float32)

    parallel_config = ParallelPPOConfig(num_envs=args.ppo_envs, vec_mode=args.ppo_vec_mode)
    ppo_model, ppo_training_config = train_ppo_policy(
        dataset.loc[X_train.index].reset_index(drop=True),
        train_obs,
        timesteps=args.ppo_timesteps,
        seed=args.seed,
        parallel_config=parallel_config,
    )
    test_actions, _ = ppo_model.predict(test_obs, deterministic=True)
    ppo_summary = evaluate_ppo_policy(dataset.loc[X_test.index].reset_index(drop=True), test_actions.astype(int))

    stat_tests = run_statistical_tests(
        raw_df=dataset,
        X_full=X,
        y_full=y.to_numpy(),
        y_test=y_test.to_numpy(),
        y_test_probs=test_probs,
    )

    ppo_model.save(str(output_dir / "ppo_premium_policy"))
    save_ppo_scaler(scaler, output_dir / SCALER_FILENAME)

    save_json(output_dir / "statistical_tests.json", stat_tests)

    summary = {
        "dataset_path": str(args.data.resolve()),
        "rows": int(len(dataset)),
        "feature_count": len(feature_columns),
        "lightgbm_metrics": metrics,
        "ppo_policy": {
            "timesteps": int(args.ppo_timesteps),
            **ppo_training_config,
            **ppo_summary,
        },
        "top_shap_features": shap_importance.head(20).to_dict(orient="records"),
        "statistical_tests_summary": {
            "shapiro_wilk_tested": len(stat_tests.get("shapiro_wilk", {})),
            "mannwhitneyu_tested": len(stat_tests.get("mannwhitneyu", {})),
            "chi2_tested": len(stat_tests.get("chi2", {})),
            "spearman_tested": len(stat_tests.get("spearman", {})),
            "ks_test": stat_tests.get("ks_test", {}),
            "logistic_odds_ratios": stat_tests.get("logistic_odds_ratios", []),
        },
    }
    save_json(output_dir / "training_summary.json", summary)

    performance_summary = {
        "lightgbm_metrics": metrics,
        "ppo_policy": {
            "timesteps": int(args.ppo_timesteps),
            **ppo_training_config,
            **ppo_summary,
        },
    }
    print("Model performance summary:", flush=True)
    print(json.dumps(performance_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
