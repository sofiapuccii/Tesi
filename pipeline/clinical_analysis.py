"""
Calcola ST% (Sviluppo Tipico) sui dati WEEK
usando il modello di classificazione addestrato.

Workflow:
1. Carica modello di classificazione addestrato
2. Carica AHA scores clinici dal file Excel metadata
3. Calcola ST% per ogni soggetto SUI DATI WEEK
4. Calcola correlazioni ST% vs AHA scores clinici
5. Salva dataset per regressione (ST% WEEK → AHA score)

Obiettivo: Predire capacità motorie (AHA) da movimenti di vita reale (WEEK)"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import pearsonr

# Configurazione TensorFlow
tf.get_logger().setLevel("ERROR")

def load_model_and_metadata(models_dir: Path) -> Tuple[tf.keras.Model, Dict]:
    """Carica modello e metadata."""
    model_path = models_dir / "best_model_classification.keras"
    meta_path = models_dir / "best_model_metadata.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modello non trovato: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata non trovato: {meta_path}")
    
    model = tf.keras.models.load_model(model_path) # carica modello 
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f) # carica metadata
    
    if "feature_stats" not in metadata:
        raise ValueError("Metadata privo di feature_stats")
    
    return model, metadata


def normalize_with_metadata(X: np.ndarray, feature_stats: Dict) -> np.ndarray:
    """Normalizza i dati usando le statistiche salvate dal training."""
    mean = np.array(feature_stats["mean"], dtype=np.float32)
    std = np.array(feature_stats["std"], dtype=np.float32)
    
    if mean.ndim == 1:
        mean = mean.reshape(1, 1, -1)
    if std.ndim == 1:
        std = std.reshape(1, 1, -1)
    
    std = np.where(std < 1e-6, 1e-6, std)
    return (X - mean) / std


def load_aha_scores_from_metadata(metadata_path: Path) -> Dict[int, float]:
    """Carica AHA scores clinici dal file Excel originale."""
    metadata = pd.read_excel(metadata_path) 
    aha_scores = {}

    for _, row in metadata.iterrows():
        subject_num = int(row['subject'])
        aha_scores[subject_num] = float(row['AHA'])
    
    print(f"AHA scores caricati da Excel: {len(aha_scores)} soggetti")
    return aha_scores


def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carica signals e windows DataFrame per dati WEEK."""
    base_path = data_dir / "preprocessed" / "week"
    try:
        sig_path = base_path / "signals.parquet"
        win_path = base_path / "windows.parquet"
        
        signals = pd.read_parquet(sig_path) if sig_path.exists() else pd.read_csv(base_path / "signals.csv")
        windows = pd.read_parquet(win_path) if win_path.exists() else pd.read_csv(base_path / "windows.csv")
        
        print(f"WEEK: {len(signals):,} segnali, {len(windows):,} finestre")
        return signals, windows
    except Exception as e:
        raise FileNotFoundError(f"Errore caricamento dati WEEK: {e}")


def build_subject_windows(subject_id: str, signals_df: pd.DataFrame, windows_df: pd.DataFrame, channels: List[str]) -> np.ndarray:
    """Costruisce finestre per un soggetto."""
    subject_signals = signals_df[signals_df["subject_id"] == subject_id].reset_index(drop=True)
    subject_windows = windows_df[windows_df["subject_id"] == subject_id]
    
    if len(subject_signals) == 0 or len(subject_windows) == 0:
        return np.empty((0, 0, len(channels)), dtype=np.float32)
    
    X_list = []
    for _, row in subject_windows.iterrows():
        segment = subject_signals.iloc[int(row["start_idx"]):int(row["end_idx"])]
        X_list.append(segment[channels].to_numpy(dtype=np.float32))
    
    return np.stack(X_list, axis=0) if X_list else np.empty((0, 0, len(channels)), dtype=np.float32)


def calculate_st_percentage(model: tf.keras.Model, subject_data: np.ndarray, feature_stats: Dict) -> float:
    """Calcola percentuale Sviluppo Tipico (ST%) per soggetto."""
    if subject_data.shape[0] == 0:
        return 0.0
    
    # Normalizza i dati
    X_norm = normalize_with_metadata(subject_data, feature_stats)
    
    # Predizioni del modello
    predictions = model.predict(X_norm, verbose=0)
    
    # Converti probabilità in classi (threshold 0.5)
    if predictions.shape[1] == 1:  # output sigmoid
        class_predictions = (predictions.flatten() < 0.5).astype(int)  # 0=ST, 1=PCU
    else:  # output softmax
        class_predictions = np.argmax(predictions, axis=1)
    
    # Calcola percentuale ST (classe 0)
    st_count = np.sum(class_predictions == 0)
    total_windows = len(class_predictions)
    st_percentage = (st_count / total_windows) * 100
    
    return float(st_percentage)


def calculate_week_st_percentages(model: tf.keras.Model, week_signals_df: pd.DataFrame, week_windows_df: pd.DataFrame, 
                                 aha_scores: Dict[int, float], feature_stats: Dict, channels: List[str]) -> Dict[int, float]:
    """Calcola ST% per tutti i soggetti sui dati WEEK."""
    print(f"\n=== CALCOLO ST% SU DATI WEEK (VITA REALE) ===")
    
    st_percentages = {}
    unique_subjects = week_signals_df["subject_id"].unique()
    
    for subject_id in unique_subjects:
        match = re.search(r"subject_(\d+)", str(subject_id))
        if not match:
            continue
        subject_num = int(match.group(1))
        if subject_num not in aha_scores:
            continue
        
        subject_data = build_subject_windows(subject_id, week_signals_df, week_windows_df, channels)
        if subject_data.shape[0] == 0:
            continue
        
        st_percentage = calculate_st_percentage(model, subject_data, feature_stats)
        st_percentages[subject_num] = st_percentage
        
        print(f"Soggetto {subject_num}: ST%={st_percentage:.1f}%, AHA_target={aha_scores[subject_num]}")
    
    if len(st_percentages) >= 2:
        subjects = list(st_percentages.keys())
        st_values = [st_percentages[s] for s in subjects]
        aha_values = [aha_scores[s] for s in subjects]
        correlation, p_value = pearsonr(st_values, aha_values)
        print(f" Correlazione ST% vs AHA: {correlation:.4f} (p={p_value:.4f})")
    
    return st_percentages


def save_regression_dataset_week_only(st_percentages_week: Dict[int, float], 
                                     aha_scores: Dict[int, float], output_path: Path) -> None:
    """Salva dataset per regressione usando solo ST% calcolati sui dati WEEK."""
    common_subjects = set(st_percentages_week.keys()) & set(aha_scores.keys())
    
    data = [{
        "subject_id": s,
        "st_percentage_week": st_percentages_week[s],
        "true_aha_score": aha_scores[s]
    } for s in sorted(common_subjects)]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(output_path, index=False)
    print(f"   Dataset regressione salvato: {len(data)} soggetti")
    print(f"   Features: ST% da dati WEEK")
    print(f"   Target: AHA score clinico")


def run_clinical_analysis(models_dir: Path, data_dir: Path, metadata_path: Path, output_dir: Path) -> None:
    """Esegue l'analisi clinica completa."""
    
    # Carica modello e metadata
    model, metadata = load_model_and_metadata(models_dir)
    feature_stats = metadata["feature_stats"]
    channels = feature_stats["channels"]
    
    # Carica AHA scores dal file Excel originale
    aha_scores = load_aha_scores_from_metadata(metadata_path)
    
    # Carica dati WEEK e calcola ST%
    week_signals, week_windows = load_data(data_dir)
    st_percentages_week = calculate_week_st_percentages(
        model, week_signals, week_windows, aha_scores, feature_stats, channels
    )
    
    # Salva dataset regressione (solo ST% WEEK)
    save_regression_dataset_week_only(st_percentages_week, aha_scores, 
                                    output_dir / "regression_dataset.csv")
    
    print(f"\n Completato. Risultati in: {output_dir}")