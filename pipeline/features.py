"""
Feature extraction: time-domain and frequency-domain features, plus scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy, skew, kurtosis
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from .cache import get_memory


@dataclass
class FeatureParams:
    sampling_rate: int


def _window_stats(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"mean": np.nan, "std": np.nan, "rms": np.nan, "energy": np.nan, "skew": np.nan, "kurt": np.nan, "entropy": np.nan}
    x = x.astype(float)
    ps = np.maximum(x ** 2, 1e-12)
    p = ps / np.sum(ps)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "rms": float(np.sqrt(np.mean(x * x))),
        "energy": float(np.sum(x * x)),
        "skew": float(skew(x, bias=False)) if x.size > 2 else 0.0,
        "kurt": float(kurtosis(x, fisher=True, bias=False)) if x.size > 3 else 0.0,
        "entropy": float(scipy_entropy(p)),
    }


def _freq_feats(x: np.ndarray, sr: int) -> Dict[str, float]:
    if x.size == 0:
        return {"bandpower": np.nan, "peak_freq": np.nan, "centroid": np.nan}
    f, Pxx = welch(x.astype(float), fs=sr, nperseg=min(256, len(x)))
    bandpower = float(np.trapz(Pxx, f))
    peak_freq = float(f[np.argmax(Pxx)]) if Pxx.size else np.nan
    centroid = float(np.sum(f * Pxx) / np.sum(Pxx)) if np.sum(Pxx) > 0 else np.nan
    return {"bandpower": bandpower, "peak_freq": peak_freq, "centroid": centroid}


def extract_features_from_windows(signals: pd.DataFrame, windows: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    """
    Extract features from signal windows with caching support.
    
    This function is computationally expensive and benefits from caching when
    the same signals, windows, and parameters are used multiple times.
    """
    # Get memory instance for caching
    memory = get_memory()
    
    # Create cached version of the internal feature extraction logic
    @memory.cache
    def _extract_features_cached(signals_data: pd.DataFrame, windows_data: pd.DataFrame, sampling_rate: int) -> pd.DataFrame:
        """Internal cached function for feature extraction."""
        rows: List[Dict] = []
        for sid, g in signals_data.groupby("subject_id", sort=False):
            g = g.reset_index(drop=True)
            wsub = windows_data[windows_data["subject_id"] == sid]
            for _, w in wsub.iterrows():
                s, e = int(w["start_idx"]), int(w["end_idx"])  # [s, e)
                seg = g.iloc[s:e]
                row: Dict[str, float] = {
                    "subject_id": sid,
                    "label": seg["label"].mode().iloc[0],
                    "start_time": float(seg["timestamp"].iloc[0]),
                    "end_time": float(seg["timestamp"].iloc[-1]),
                }
                for wrist in ("dom", "non"):
                    mag = seg[f"mag_{wrist}"].to_numpy()
                    ts = _window_stats(mag)
                    fq = _freq_feats(mag, sampling_rate)
                    for k, v in {**ts, **fq}.items():
                        row[f"mag_{wrist}_{k}"] = v
                rows.append(row)
        return pd.DataFrame(rows)
    
    # Extract features using cached function
    feats = _extract_features_cached(signals, windows, params.sampling_rate)
    
    # Scale features (exclude id/time/label)
    id_cols = ["subject_id", "label", "start_time", "end_time"]
    feat_cols = [c for c in feats.columns if c not in id_cols]
    scaler = StandardScaler()
    feats[feat_cols] = scaler.fit_transform(feats[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0))
    
    return feats


