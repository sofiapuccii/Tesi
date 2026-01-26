"""
Calcola ST% (Sviluppo Tipico) sui dati WEEK
usando il modello di classificazione addestrato.

MODIFICHE:
- Genera 4 grafici separati: 2 per soggetto SANO + 2 per soggetto PCU
- Ogni soggetto ha: (1) blocchi disgiunti 6h, (2) finestra scorrevole 6h
- Stile identico alla collega (griglia y: 0,50,100; x: ogni 00:00)
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import matplotlib.dates as mdates

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
    
    model = tf.keras.models.load_model(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    return model, metadata


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


def build_subject_windows(subject_id: str, signals_df: pd.DataFrame, windows_df: pd.DataFrame, channels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Costruisce finestre per un soggetto e restituisce anche i TIMESTAMP reali."""
    subject_signals = signals_df[signals_df["subject_id"] == subject_id].reset_index(drop=True)
    subject_windows = windows_df[windows_df["subject_id"] == subject_id]
    
    if len(subject_signals) == 0 or len(subject_windows) == 0:
        return np.empty((0, 0, len(channels)), dtype=np.float32), np.array([])
    
    time_col = None
    for col in ['timestamp', 'time', 'datetime', 'Date']:
        if col in subject_signals.columns:
            time_col = col
            break
            
    X_list = []
    t_list = []
    
    for _, row in subject_windows.iterrows():
        start_idx = int(row["start_idx"])
        end_idx = int(row["end_idx"])
        
        segment = subject_signals.iloc[start_idx:end_idx]
        X_list.append(segment[channels].to_numpy(dtype=np.float32))
        
        if time_col:
            t_list.append(subject_signals.iloc[start_idx][time_col])
        else:
            t_list.append(start_idx) 

    X_arr = np.stack(X_list, axis=0) if X_list else np.empty((0, 0, len(channels)), dtype=np.float32)
    t_arr = np.array(t_list)
    
    return X_arr, t_arr


def calculate_st_percentage(model: tf.keras.Model, subject_data: np.ndarray, skip_normalization: bool = True) -> Tuple[float, np.ndarray]:
    """Calcola percentuale Sviluppo Tipico (ST%) per soggetto e restituisce anche predizioni."""
    if subject_data.shape[0] == 0:
        return 0.0, np.array([])
    
    if skip_normalization:
        X_norm = subject_data
    else:
        mean = np.mean(subject_data, axis=(0, 1), keepdims=True)
        std = np.std(subject_data, axis=(0, 1), keepdims=True)
        std = np.where(std < 1e-6, 1e-6, std)
        X_norm = (subject_data - mean) / std
    
    predictions = model.predict(X_norm, verbose=0)
    #===================================================================================================================================================
    #PARTE NUOVA
    #class_predictions = (predictions.flatten() >= 0.40).astype(int)
    #Estare la probabilità di malattia (PCU)
    if predictions.shape[1]==1:
        prob_pcu = predictions.flatten()
    else:
        prob_pcu = predictions[:,1]
    #calcola la probabilità di sanità (ST)
    prob_st= 1.0 - prob_pcu
    #si fa la media delle probabilità (invece di contare gli 0)
    st_percentage = np.mean(prob_st) * 100
    #st_count = np.sum(class_predictions == 0)
    #total_windows = len(class_predictions)
    #st_percentage = (st_count / total_windows) * 100
    
    return float(st_percentage), predictions


def calculate_week_st_percentages(model: tf.keras.Model, week_signals_df: pd.DataFrame, week_windows_df: pd.DataFrame, 
                                 aha_scores: Dict[int, float], channels: List[str]) -> Tuple[Dict, Dict, Dict]:
    """Calcola ST% e raccoglie dati temporali."""
    print(f"\n=== CALCOLO ST% SU DATI WEEK (VITA REALE) ===")
    
    st_percentages = {}
    all_predictions = {}
    all_timestamps = {}
    
    unique_subjects = week_signals_df["subject_id"].unique()
    
    for subject_id in unique_subjects:
        match = re.search(r"subject_(\d+)", str(subject_id))
        if not match: continue
        subject_num = int(match.group(1))
        if subject_num not in aha_scores: continue
        
        subject_data, t_subject = build_subject_windows(subject_id, week_signals_df, week_windows_df, channels)
        
        if subject_data.shape[0] == 0: continue
        
        st_percentage, predictions = calculate_st_percentage(model, subject_data, skip_normalization=True)
        
        st_percentages[subject_num] = st_percentage
        all_predictions[subject_num] = predictions
        all_timestamps[subject_num] = t_subject
        
        print(f"Soggetto {subject_num}: ST%={st_percentage:.1f}%, AHA={aha_scores[subject_num]}")
    
    if len(st_percentages) >= 2:
        subjects = list(st_percentages.keys())
        st_values = [st_percentages[s] for s in subjects]
        aha_values = [aha_scores[s] for s in subjects]
        correlation, p_value = pearsonr(st_values, aha_values)
        print(f"\n Correlazione ST% vs AHA: {correlation:.4f} (p={p_value:.4f})")
    
    return st_percentages, all_predictions, all_timestamps

def save_regression_dataset_week_only(st_percentages_week: Dict[int, float], 
                                     aha_scores: Dict[int, float], output_path: Path) -> None:
    """Mantieni identica"""
    common_subjects = set(st_percentages_week.keys()) & set(aha_scores.keys())
    
    data = [{
        "subject_id": s,
        "st_percentage_week": st_percentages_week[s],
        "true_aha_score": aha_scores[s]
    } for s in sorted(common_subjects)]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(output_path, index=False)
    print(f"    Dataset regressione salvato: {len(data)} soggetti")


def save_temporal_predictions(all_predictions: Dict, all_timestamps: Dict, aha_scores: Dict, output_path: Path):
    """Mantieni identica"""
    rows = []
    
    for subject_id, preds in all_predictions.items():
        if subject_id not in all_timestamps: continue
        
        timestamps = all_timestamps[subject_id]
        true_aha = aha_scores.get(subject_id, np.nan)
        
        if preds.shape[1] == 1:
            is_st = (preds.flatten() < 0.40).astype(int)
        else:
            is_st = (np.argmax(preds, axis=1) == 0).astype(int)
            
        limit = min(len(timestamps), len(is_st))
        
        for i in range(limit):
            rows.append({
                'subject_id': subject_id,
                'timestamp': timestamps[i],
                'is_st': is_st[i],
                'true_aha_score': true_aha
            })
            
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"    Dataset temporale salvato: {output_path} ({len(df)} righe)")



def run_clinical_analysis(models_dir: Path, data_dir: Path, metadata_path: Path, output_dir: Path) -> None:
    """Esegue l'analisi clinica completa con 4 grafici separati."""
    
    model, metadata = load_model_and_metadata(models_dir)
    channels = metadata["feature_stats"]["channels"]
    
    aha_scores = load_aha_scores_from_metadata(metadata_path)
    
    print("Caricando dati WEEK per analisi clinica...")
    week_signals, week_windows = load_data(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    st_percentages_week, all_predictions, all_timestamps = calculate_week_st_percentages(
        model, week_signals, week_windows, aha_scores, channels
    )
    
    # Salva dataset per regressione (invariati)
    save_regression_dataset_week_only(st_percentages_week, aha_scores, 
                                    output_dir / "regression_dataset.csv")
    save_temporal_predictions(all_predictions, all_timestamps, aha_scores, 
                              output_dir / "temporal_predictions.csv")
    
    print(f"\n Completato. Risultati in: {output_dir}")