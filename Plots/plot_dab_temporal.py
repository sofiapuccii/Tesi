"""
Visualizzazione temporale del Daily AHA Biomarker (DAB) con Smoothing.

Workflow:
1. Carica i dati generati da clinical_analysis.py
2. Applica una media mobile (Rolling Window 6h) per ottenere l'ST% locale
3. Applica la formula di regressione (Slope/Intercept) per stimare il DAB
4. Plotta il risultato confrontandolo con l'AHA Reale.
"""

from __future__ import annotations
from pathlib import Path

from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.dates as mdates
from datetime import datetime, timedelta

# INCOLLA VALORI OTTENUTI DA linear_regression_analysis.py
REGRESSION_SLOPE = 9.93025     
REGRESSION_INTERCEPT = -273.81723  

# Stile dei grafici
plt.style.use('default')
plt.rcParams['figure.figsize'] = (15, 6)
plt.rcParams['font.size'] = 12

def load_temporal_data(data_path: Path) -> pd.DataFrame:
    """
    Carica il dataset completo delle predizioni temporali.
    Format atteso: subject_id, timestamp, is_st, true_aha_score
    """
    if not data_path.exists():
        raise FileNotFoundError(f"File dati non trovato: {data_path}")
    
    # Carica CSV
    df = pd.read_csv(data_path)
    
    # Converti timestamp in datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        raise ValueError("Il file CSV deve contenere una colonna 'timestamp'")
        
    print(f"Dataset caricato: {len(df)} righe totali.")
    return df

def process_subject_dab(subject_df: pd.DataFrame, slope: float, intercept: float, window: str = '6h') -> pd.DataFrame:
    """
    Calcola il DAB 'smooth' usando una media mobile temporale.
    """
    # Ordina per tempo e imposta indice per il rolling
    df = subject_df.sort_values('timestamp').set_index('timestamp')
    
    # 1. Calcolo ST% Locale 
    # Calcola la media nella finestra temporale
    # Moltiplichiamo per 100 per avere la percentuale (0-100)
    # min_periods=10: serve almeno qualche dato per disegnare la linea
    df['rolling_st_percent'] = df['is_st'].rolling(window, min_periods=10).mean() * 100
    
    # 2. Applicazione Formula Regressione
    # DAB = (ST% * slope) + intercept
    df['dab_smooth'] = (df['rolling_st_percent'] * slope) + intercept
    
    # 3. Clip dei valori (non possono uscire dal range 0-100)
    df['dab_smooth'] = df['dab_smooth'].clip(0, 100)
    
    return df


def plot_dab_temporal(processed_df: pd.DataFrame, subject_id: str, 
                      true_aha: float, output_path: Path) -> None:
    """
    Crea grafico temporale DAB vs tempo per una settimana.
    """
    # Calcola la durata totale dei dati
    time_span = processed_df.index.max() - processed_df.index.min()
    days_span = time_span.total_seconds() / (24 * 3600)
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    ax.plot(processed_df.index, processed_df['dab_smooth'], 
            color='forestgreen', linewidth=2.5, alpha=0.9, label='Estimated Daily AHA (DAB)')
    
    # 2. Linea di Riferimento AHA Reale (Blu Tratteggiato)
    ax.axhline(y=true_aha, color='royalblue', linestyle='--', linewidth=2, 
               alpha=0.8, label=f'True Clinical AHA ({true_aha:.0f})')
    
    # Personalizzazione
    ax.set_title(f'Daily AHA Biomarker {subject_id}', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('DAB', fontsize=12, fontweight='bold')
    ax.set_xlabel('Orario', fontsize=12)
    ax.set_ylim(0, 105) # Un po' di margine sopra il 100
    
    if days_span > 3:  
        # Major ticks ogni giorno alle 00:00
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Mostra solo l'orario (00:00)
        # Minor ticks ogni 12 ore per segnare anche il mezzogiorno
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[12]))
        # Aggiungi linee verticali sottili per separare i giorni
        for day in pd.date_range(start=processed_df.index.min().floor('D'), 
                                end=processed_df.index.max().ceil('D'), freq='D'):
            ax.axvline(x=day, color='gray', linestyle='-', alpha=0.2, linewidth=0.8)
    else:  # Per periodi più brevi, usa formato orario standard
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    
    ax.grid(True, which='major', alpha=0.3)
    ax.grid(True, which='minor', alpha=0.15)
    
    # Statistiche nel box
    avg_dab = processed_df['dab_smooth'].mean()
    stats_text = (f'Mean Estimated AHA: {avg_dab:.1f}\n'
                  f'True Clinical AHA: {true_aha:.1f}\n'
                  f'Error: {avg_dab - true_aha:.1f}\n'
                  f'Data span: {days_span:.1f} days')
    
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    ax.legend(loc='upper right')
    
    # Salva
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close() # Chiude per liberare memoria
    print(f"   -> Grafico salvato: {output_path}")

def main():

    base_dir = Path("results") 
    input_file = base_dir / "clinical_analysis_v2" / "temporal_predictions.csv"
    output_dir = base_dir / "dab_plots"
    
    print(f"Caricamento dati da: {input_file}")
    
    try:
        # 1. Carica tutti i dati
        full_df = load_temporal_data(input_file)
        
        # 2. Trova lista soggetti unici
        unique_subjects = full_df['subject_id'].unique()
        print(f"Trovati {len(unique_subjects)} soggetti da analizzare.")
        
        # 3. Ciclo su ogni soggetto
        for subject_id in unique_subjects:
            # Estrai dati solo per questo soggetto
            sub_df = full_df[full_df['subject_id'] == subject_id].copy()
                
            true_aha = sub_df['true_aha_score'].iloc[0]
            
            # Elabora (Rolling Mean + Regressione)
            processed_df = process_subject_dab(sub_df, REGRESSION_SLOPE, REGRESSION_INTERCEPT)
            
            out_path = output_dir / f"dab_profile_{subject_id}.png"
            plot_dab_temporal(processed_df, subject_id, true_aha, out_path)
            
        print("\nTutti i grafici sono stati generati correttamente.")
            
    except Exception as e:
        print(f"\nERRORE: {e}")

if __name__ == "__main__":
    main()