from __future__ import annotations

import hashlib
import io
import math
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

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

from telematics_data_generator import _get_spark_session, fill_missing_plate_values, generate_placeholder_plates
from pricing_engine import (
    USD_TO_ZIG_RATE_DEFAULT,
    build_insurance_premium_schedule,
    calculate_profitability_metrics,
)
from risk_scoring_logic import (
    build_city_risk_summary_table,
    build_risk_band_summary_table,
    build_risk_recommendation_table,
    compute_live_risk_outputs,
)
from pricing_engine import HybridPricingInferenceService

MAX_ANALYSIS_WORKERS = max(2, min(os.cpu_count() or 4, 8))
UPLOAD_JOB_RETENTION_SECONDS = 60.0 * 60.0
MAX_STORED_UPLOAD_JOBS = 12
# Keep interactive uploads on the low-latency pandas/threaded path.
# Spark remains available for genuinely large portfolio batches.
SPARK_UPLOAD_ANALYSIS_MIN_ROWS = 15_000

_SERVICE_LOCAL = threading.local()
_UPLOAD_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(2, min(MAX_ANALYSIS_WORKERS, 4)),
    thread_name_prefix="insurtech-upload-analysis",
)
_UPLOAD_JOBS_LOCK = threading.RLock()
_UPLOAD_JOBS: dict[str, dict[str, Any]] = {}
_UPLOAD_WARMUP_FUTURE: Future | None = None

_PORTFOLIO_STATE = {
    "thread": None,
    "stop_event": threading.Event(),
    "lock": threading.RLock(),
    "latest_bundle": None,
    "last_run": 0.0,
    "interval_seconds": 15.0,
    "running": False,
    "last_error": None,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _copy_bundle(bundle: dict[str, Any] | None, *, deep_frames: bool = True) -> dict[str, Any] | None:
    if bundle is None:
        return None

    copied: dict[str, Any] = {}
    for key, value in bundle.items():
        if isinstance(value, pd.DataFrame):
            copied[key] = value.copy(deep=deep_frames)
        elif isinstance(value, dict):
            copied[key] = dict(value)
        elif isinstance(value, list):
            copied[key] = list(value)
        else:
            copied[key] = value
    return copied


def _get_thread_local_service() -> HybridPricingInferenceService:
    service = getattr(_SERVICE_LOCAL, "service", None)
    if service is None:
        service = HybridPricingInferenceService(max_workers=MAX_ANALYSIS_WORKERS)
        service.load_artifacts()
        _SERVICE_LOCAL.service = service
    return service


def _warm_upload_worker_service() -> None:
    _get_thread_local_service()


def prewarm_upload_analysis_service() -> None:
    global _UPLOAD_WARMUP_FUTURE

    with _UPLOAD_JOBS_LOCK:
        if _UPLOAD_WARMUP_FUTURE is not None:
            if not _UPLOAD_WARMUP_FUTURE.done():
                return
            try:
                _UPLOAD_WARMUP_FUTURE.result()
                return
            except Exception:
                _UPLOAD_WARMUP_FUTURE = None

        _UPLOAD_WARMUP_FUTURE = _UPLOAD_EXECUTOR.submit(_warm_upload_worker_service)


def _score_frame_adaptively(
    frame: pd.DataFrame,
    service: HybridPricingInferenceService,
) -> tuple[pd.DataFrame, str]:
    """Use the lightest viable scoring engine, falling back safely when needed."""
    if frame is None or frame.empty:
        return (pd.DataFrame() if frame is None else frame.copy(), "empty")

    partition_size = _pick_partition_size(len(frame))
    if len(frame) <= partition_size:
        return service.score_pricing(frame), "pandas"

    if len(frame) >= SPARK_UPLOAD_ANALYSIS_MIN_ROWS:
        try:
            spark = _get_spark_session()
            spark_ready = frame.copy()
            spark_ready["__row_order__"] = np.arange(len(spark_ready), dtype=np.int64)
            spark_df = spark.createDataFrame(spark_ready)
            target_partitions = max(2, min(MAX_ANALYSIS_WORKERS, math.ceil(len(frame) / max(1, partition_size))))
            spark_df = spark_df.repartition(target_partitions)
            scored = service.score_spark_dataframe(spark_df).toPandas()
            if "__row_order__" in scored.columns:
                scored = (
                    scored.sort_values("__row_order__", kind="stable")
                    .drop(columns=["__row_order__"], errors="ignore")
                    .reset_index(drop=True)
                )
            return scored, "spark"
        except Exception:
            # Fall back to threaded pandas scoring if Spark startup or execution fails.
            pass

    return (
        service.score_frame_concurrently(frame, partition_size=partition_size, max_workers=MAX_ANALYSIS_WORKERS),
        "threaded-pandas",
    )


def _pick_partition_size(row_count: int) -> int:
    if row_count <= 500:
        return max(50, row_count)
    if row_count <= 5_000:
        return 500
    return 1_000


def _prepare_analysis_input(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    frame = df.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    unnamed_columns = [column for column in frame.columns if str(column).startswith("Unnamed:")]
    if unnamed_columns:
        frame = frame.drop(columns=unnamed_columns, errors="ignore")

    if "Plate" not in frame.columns:
        frame["Plate"] = pd.Series(generate_placeholder_plates(len(frame)), index=frame.index, dtype=object)
    else:
        frame["Plate"] = fill_missing_plate_values(frame["Plate"])

    if "Type" not in frame.columns:
        frame["Type"] = "Sedan"
    frame["Type"] = frame["Type"].fillna("Sedan").astype(str)

    if "Usage" not in frame.columns:
        frame["Usage"] = "Private"
    frame["Usage"] = frame["Usage"].fillna("Private").astype(str)

    if "Status" not in frame.columns:
        frame["Status"] = "Driving"
    frame["Status"] = frame["Status"].fillna("Driving").astype(str)

    if "City" not in frame.columns:
        frame["City"] = "Harare"
    frame["City"] = frame["City"].fillna("Harare").astype(str)

    if "Base_Price_USD" not in frame.columns:
        if "Vehicle_Value_USD" in frame.columns:
            frame["Base_Price_USD"] = pd.to_numeric(frame["Vehicle_Value_USD"], errors="coerce")
        else:
            frame["Base_Price_USD"] = 6000.0

    frame["Base_Price_USD"] = pd.to_numeric(frame["Base_Price_USD"], errors="coerce").fillna(6000.0).clip(lower=500.0)
    return frame.reset_index(drop=True)


def _build_action_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "PPO_Action" not in df.columns:
        return pd.DataFrame(columns=["PPO_Action", "Vehicle_Count"])
    return (
        df["PPO_Action"]
        .fillna("standard_offer")
        .value_counts(dropna=False)
        .rename_axis("PPO_Action")
        .reset_index(name="Vehicle_Count")
    )


def _build_analysis_summary(df: pd.DataFrame, trained_models_available: bool, source_name: str) -> dict[str, Any]:
    if df.empty:
        return {
            "source_name": source_name,
            "trained_models_available": trained_models_available,
            "record_count": 0,
            "quoted_count": 0,
            "declined_count": 0,
            "high_risk_count": 0,
            "critical_risk_count": 0,
            "avg_model_risk_score": 0.0,
            "avg_calculated_risk_score": 0.0,
            "avg_monthly_premium_usd": 0.0,
            "total_monthly_premium_usd": 0.0,
            "total_expected_claim_usd": 0.0,
            "total_underwriting_profit_usd": 0.0,
        }

    decision_series = df.get("Decision_Status", pd.Series("quoted", index=df.index)).fillna("quoted").astype(str)
    risk_band_series = df.get("Risk_Band", pd.Series("Medium", index=df.index)).fillna("Medium").astype(str)
    model_risk_series = pd.to_numeric(df.get("LightGBM_Risk_Score", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    calculated_risk_series = pd.to_numeric(df.get("Calculated_Risk_Score", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    monthly_premium_series = pd.to_numeric(df.get("Monthly_Premium_USD", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    expected_claim_series = pd.to_numeric(df.get("Expected_Claim_USD", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    underwriting_profit_series = pd.to_numeric(df.get("Underwriting_Profit_USD", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    return {
        "source_name": source_name,
        "trained_models_available": trained_models_available,
        "record_count": int(len(df)),
        "quoted_count": int((decision_series == "quoted").sum()),
        "declined_count": int((decision_series == "declined").sum()),
        "high_risk_count": int(risk_band_series.isin(["High", "Critical"]).sum()),
        "critical_risk_count": int((risk_band_series == "Critical").sum()),
        "avg_model_risk_score": float(model_risk_series.mean()),
        "avg_calculated_risk_score": float(calculated_risk_series.mean()),
        "avg_monthly_premium_usd": float(monthly_premium_series.mean()),
        "total_monthly_premium_usd": float(monthly_premium_series.sum()),
        "total_expected_claim_usd": float(expected_claim_series.sum()),
        "total_underwriting_profit_usd": float(underwriting_profit_series.sum()),
    }


def analyze_portfolio_dataframe(
    df: pd.DataFrame,
    usd_to_zig_rate: float = USD_TO_ZIG_RATE_DEFAULT,
    source_name: str = "portfolio",
) -> dict[str, Any]:
    frame = _prepare_analysis_input(df)
    if frame.empty:
        empty_bundle = {
            "analysis_df": pd.DataFrame(),
            "premium_schedule_df": pd.DataFrame(),
            "risk_summary_df": pd.DataFrame(),
            "city_summary_df": pd.DataFrame(),
            "recommendations_df": pd.DataFrame(),
            "action_summary_df": pd.DataFrame(),
            "summary": _build_analysis_summary(pd.DataFrame(), trained_models_available=False, source_name=source_name),
            "generated_at": _utc_now_iso(),
            "source_name": source_name,
            "trained_models_available": False,
        }
        return empty_bundle

    service = _get_thread_local_service()
    scored, processing_engine = _score_frame_adaptively(frame, service)

    # Recompute the live risk view from the telemetry features rather than
    # smoothing against trained-model pricing scores that were produced for a
    # different purpose. This preserves realistic live risk distributions in
    # Streamlit while still keeping the trained LightGBM output available.
    scored = scored.drop(
        columns=["Risk_Score", "Calculated_Risk_Score", "Risk_Band", "Calculated_Risk_Band", "Custom_Risk_Score", "Risk_Action"],
        errors="ignore",
    )
    scored = compute_live_risk_outputs(scored)
    analyzed = calculate_profitability_metrics(scored, usd_to_zig_rate=usd_to_zig_rate)
    analyzed["Analysis_Source"] = source_name
    analyzed["Analysis_Generated_At"] = _utc_now_iso()

    trained_models_available = bool(service.trained_models_available)
    return {
        "analysis_df": analyzed,
        "premium_schedule_df": build_insurance_premium_schedule(analyzed, usd_to_zig_rate=usd_to_zig_rate),
        "risk_summary_df": build_risk_band_summary_table(analyzed),
        "city_summary_df": build_city_risk_summary_table(analyzed),
        "recommendations_df": build_risk_recommendation_table(analyzed, top_n=min(50, len(analyzed))),
        "action_summary_df": _build_action_summary(analyzed),
        "summary": _build_analysis_summary(analyzed, trained_models_available=trained_models_available, source_name=source_name),
        "generated_at": _utc_now_iso(),
        "source_name": source_name,
        "trained_models_available": trained_models_available,
        "processing_engine": processing_engine,
        "usd_to_zig_rate": float(usd_to_zig_rate),
    }


def start_background_portfolio_analysis(
    get_dataset_callable: Callable[[], pd.DataFrame],
    interval_seconds: float = 15.0,
    usd_to_zig_rate: float = USD_TO_ZIG_RATE_DEFAULT,
) -> None:
    if not callable(get_dataset_callable):
        return

    if _PORTFOLIO_STATE.get("running"):
        _PORTFOLIO_STATE["interval_seconds"] = float(interval_seconds)
        _PORTFOLIO_STATE["get_dataset_callable"] = get_dataset_callable
        _PORTFOLIO_STATE["usd_to_zig_rate"] = float(usd_to_zig_rate)
        return

    _PORTFOLIO_STATE["interval_seconds"] = float(interval_seconds)
    _PORTFOLIO_STATE["get_dataset_callable"] = get_dataset_callable
    _PORTFOLIO_STATE["usd_to_zig_rate"] = float(usd_to_zig_rate)
    _PORTFOLIO_STATE["last_error"] = None
    _PORTFOLIO_STATE["stop_event"].clear()

    def _worker() -> None:
        stop_event = _PORTFOLIO_STATE["stop_event"]
        lock = _PORTFOLIO_STATE["lock"]
        while not stop_event.is_set():
            try:
                dataset = _PORTFOLIO_STATE["get_dataset_callable"]()
                bundle = analyze_portfolio_dataframe(
                    dataset,
                    usd_to_zig_rate=float(_PORTFOLIO_STATE["usd_to_zig_rate"]),
                    source_name="live_background",
                )
                with lock:
                    _PORTFOLIO_STATE["latest_bundle"] = bundle
                    _PORTFOLIO_STATE["last_run"] = time.time()
                    _PORTFOLIO_STATE["last_error"] = None
            except Exception as exc:
                with lock:
                    _PORTFOLIO_STATE["last_error"] = str(exc)
            stop_event.wait(float(_PORTFOLIO_STATE.get("interval_seconds", 15.0)))
        _PORTFOLIO_STATE["running"] = False

    thread = threading.Thread(target=_worker, daemon=True, name="portfolio_background_analysis")
    _PORTFOLIO_STATE["thread"] = thread
    _PORTFOLIO_STATE["running"] = True
    thread.start()


def stop_background_portfolio_analysis() -> None:
    try:
        stop_event = _PORTFOLIO_STATE["stop_event"]
        stop_event.set()
        thread = _PORTFOLIO_STATE.get("thread")
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
    finally:
        _PORTFOLIO_STATE["running"] = False


def get_latest_portfolio_analysis(max_age_seconds: float = 60.0) -> dict[str, Any] | None:
    with _PORTFOLIO_STATE["lock"]:
        last_run = float(_PORTFOLIO_STATE.get("last_run", 0.0))
        bundle = _PORTFOLIO_STATE.get("latest_bundle")
        if bundle is None:
            return None
        if time.time() - last_run > float(max_age_seconds):
            return None
        return _copy_bundle(bundle, deep_frames=False)


def get_latest_background_error() -> str | None:
    with _PORTFOLIO_STATE["lock"]:
        return _PORTFOLIO_STATE.get("last_error")


def read_uploaded_dataframe(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    suffix = os.path.splitext(file_name or "")[1].lower()
    payload = io.BytesIO(file_bytes)
    polars_module = _get_polars_module()

    if suffix == ".csv":
        if polars_module is not None:
            return polars_module.read_csv(payload, try_parse_dates=True).to_pandas()
        try:
            return pd.read_csv(payload, engine="pyarrow")
        except Exception:
            payload.seek(0)
            return pd.read_csv(payload)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(payload)
    if suffix == ".parquet":
        if polars_module is not None:
            return polars_module.read_parquet(payload).to_pandas()
        return pd.read_parquet(payload)

    try:
        payload.seek(0)
        if polars_module is not None:
            return polars_module.read_csv(payload, try_parse_dates=True).to_pandas()
        return pd.read_csv(payload, engine="pyarrow")
    except Exception:
        try:
            payload.seek(0)
            return pd.read_excel(payload)
        except Exception as excel_error:
            raise ValueError("Unsupported upload format. Use CSV, XLSX, XLS, or Parquet.") from excel_error


def _cleanup_upload_jobs() -> None:
    now = time.time()
    removable: list[str] = []

    for job_id, job in _UPLOAD_JOBS.items():
        completed_at = float(job.get("completed_at_ts") or 0.0)
        if completed_at and (now - completed_at) > UPLOAD_JOB_RETENTION_SECONDS:
            removable.append(job_id)

    if len(_UPLOAD_JOBS) - len(removable) > MAX_STORED_UPLOAD_JOBS:
        completed_jobs = sorted(
            (
                (job_id, float(job.get("completed_at_ts") or 0.0))
                for job_id, job in _UPLOAD_JOBS.items()
                if job.get("status") in {"completed", "failed"}
            ),
            key=lambda item: item[1],
        )
        extra = len(_UPLOAD_JOBS) - len(removable) - MAX_STORED_UPLOAD_JOBS
        removable.extend([job_id for job_id, _ in completed_jobs[:max(0, extra)]])

    for job_id in set(removable):
        _UPLOAD_JOBS.pop(job_id, None)


def _set_upload_job_state(job_id: str, **updates: Any) -> None:
    with _UPLOAD_JOBS_LOCK:
        job = _UPLOAD_JOBS.get(job_id)
        if job is None:
            return
        job.update(updates)


def _run_upload_analysis_job(job_id: str, file_name: str, file_bytes: bytes, usd_to_zig_rate: float) -> dict[str, Any]:
    _set_upload_job_state(job_id, status="running", started_at=_utc_now_iso(), started_at_ts=time.time())
    frame = read_uploaded_dataframe(file_name, file_bytes)
    bundle = analyze_portfolio_dataframe(frame, usd_to_zig_rate=usd_to_zig_rate, source_name=file_name)
    summary = dict(bundle.get("summary", {}))
    summary["file_name"] = file_name
    bundle["summary"] = summary
    return bundle


def _finalize_upload_analysis_job(job_id: str, future: Future) -> None:
    try:
        result = future.result()
    except Exception as exc:
        _set_upload_job_state(
            job_id,
            status="failed",
            error=str(exc),
            completed_at=_utc_now_iso(),
            completed_at_ts=time.time(),
        )
        return

    _set_upload_job_state(
        job_id,
        status="completed",
        result=result,
        error=None,
        completed_at=_utc_now_iso(),
        completed_at_ts=time.time(),
    )


def submit_upload_analysis(
    file_name: str,
    file_bytes: bytes,
    usd_to_zig_rate: float = USD_TO_ZIG_RATE_DEFAULT,
) -> str:
    signature = hashlib.sha1(
        file_name.encode("utf-8", errors="ignore")
        + b"|"
        + file_bytes
        + b"|"
        + f"{float(usd_to_zig_rate):.4f}".encode("utf-8")
    ).hexdigest()

    with _UPLOAD_JOBS_LOCK:
        _cleanup_upload_jobs()
        for job_id, job in _UPLOAD_JOBS.items():
            if job.get("signature") == signature and job.get("status") in {"queued", "running", "completed"}:
                return job_id

        job_id = f"upload-{signature[:12]}"
        future = _UPLOAD_EXECUTOR.submit(
            _run_upload_analysis_job,
            job_id,
            file_name,
            bytes(file_bytes),
            float(usd_to_zig_rate),
        )
        future.add_done_callback(lambda completed_future, target_job_id=job_id: _finalize_upload_analysis_job(target_job_id, completed_future))
        _UPLOAD_JOBS[job_id] = {
            "job_id": job_id,
            "signature": signature,
            "file_name": file_name,
            "status": "queued",
            "submitted_at": _utc_now_iso(),
            "submitted_at_ts": time.time(),
            "result": None,
            "error": None,
            "future": future,
        }
        return job_id


def get_upload_analysis_job(job_id: str) -> dict[str, Any] | None:
    with _UPLOAD_JOBS_LOCK:
        _cleanup_upload_jobs()
        job = _UPLOAD_JOBS.get(job_id)
        if job is None:
            return None

        safe_job = {key: value for key, value in job.items() if key != "future"}
        if isinstance(safe_job.get("result"), dict):
            safe_job["result"] = _copy_bundle(safe_job["result"], deep_frames=False)
        return safe_job


def clear_upload_analysis_job(job_id: str) -> None:
    with _UPLOAD_JOBS_LOCK:
        _UPLOAD_JOBS.pop(job_id, None)
