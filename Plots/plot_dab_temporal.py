"""
Visualizzazione temporale del Daily AHA Biomarker (DAB) per singoli soggetti.

Genera grafici temporali che mostrano:
1. Linea tratteggiata orizzontale: AHA Score clinico reale
2. Linea continua: DAB calcolato nel tempo da dati accelerometrici

Input: Dati temporali con ST% e AHA score per singoli soggetti
Output: Grafico temporale DAB vs tempo con riferimento AHA
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Stile dei grafici
plt.style.use('default')
plt.rcParams['figure.figsize'] = (15, 6)
plt.rcParams['font.size'] = 12


def load_temporal_data(data_path: Path, subject_id: str) -> pd.DataFrame:
    """
    Carica i dati temporali per un singolo soggetto.
    
    Expected CSV format:
    - subject_id: ID soggetto
    - timestamp: Timestamp dei dati
    - st_percentage: ST% calcolato per quella finestra temporale
    - true_aha_score: AHA score clinico reale
    """
    
    if not data_path.exists():
        raise FileNotFoundError(f"File dati non trovato: {data_path}")
    
    data = pd.read_csv(data_path)
    
    # Filtra per soggetto specifico
    subject_data = data[data['subject_id'] == subject_id].copy()
    
    if subject_data.empty:
        raise ValueError(f"Nessun dato trovato per soggetto: {subject_id}")
    
    # Converti timestamp
    if 'timestamp' in subject_data.columns:
        subject_data['timestamp'] = pd.to_datetime(subject_data['timestamp'])
        subject_data = subject_data.sort_values('timestamp')
    else:
        # Se non c'è timestamp, crea uno fittizio
        subject_data = subject_data.reset_index(drop=True)
        subject_data['timestamp'] = pd.date_range('2024-01-01', 
                                                 periods=len(subject_data), 
                                                 freq='H')
    
    print(f"Dati caricati per soggetto {subject_id}: {len(subject_data)} punti temporali")
    print(f"Range temporale: {subject_data['timestamp'].min()} - {subject_data['timestamp'].max()}")
    
    return subject_data


def calculate_dab_temporal(subject_data: pd.DataFrame, slope: float, intercept: float) -> pd.DataFrame:
    """
    Calcola DAB per ogni punto temporale usando i coefficienti del modello già addestrato.
    Formula: DAB = slope * ST% + intercept
    """
    
    # Calcola DAB usando ST% temporale
    st_values = subject_data['st_percentage'].values
    dab_values = slope * st_values + intercept
    
    # Aggiungi DAB ai dati
    result_data = subject_data.copy()
    result_data['dab_predicted'] = dab_values
    
    return result_data


def plot_dab_temporal(temporal_data: pd.DataFrame, subject_id: str, 
                      output_path: Path, window_hours: Optional[int] = None) -> None:
    """
    Crea grafico temporale DAB vs tempo con linea di riferimento AHA.
    """
    
    # Seleziona finestra temporale se specificata
    if window_hours:
        start_time = temporal_data['timestamp'].min()
        end_time = start_time + timedelta(hours=window_hours)
        plot_data = temporal_data[
            (temporal_data['timestamp'] >= start_time) & 
            (temporal_data['timestamp'] <= end_time)
        ].copy()
        title_suffix = f" ({window_hours}h window)"
    else:
        plot_data = temporal_data.copy()
        title_suffix = ""
    
    # Ottieni AHA score reale (dovrebbe essere costante per il soggetto)
    true_aha = plot_data['true_aha_score'].iloc[0]
    
    # Crea figura
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Plot DAB temporale (linea continua arancione)
    ax.plot(plot_data['timestamp'], plot_data['dab_predicted'], 
            color='orange', linewidth=2, alpha=0.9, label='DAB (Predicted)')
    
    # Linea tratteggiata AHA reale (blu)
    ax.axhline(y=true_aha, color='blue', linestyle='--', linewidth=2, 
               alpha=0.8, label=f'AHA (True Score: {true_aha:.1f})')
    
    # Personalizzazione assi
    ax.set_xlabel('Orario', fontsize=12, fontweight='bold')
    ax.set_ylabel('Home-AHA', fontsize=12, fontweight='bold')
    ax.set_title(f'Daily AHA Biomarker - Subject {subject_id}{title_suffix}', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Limiti Y
    ax.set_ylim(0, 100)
    
    # Formattazione tempo
    if window_hours and window_hours <= 24:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Griglia e legenda
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    
    # Statistiche nel grafico
    dab_mean = plot_data['dab_predicted'].mean()
    dab_std = plot_data['dab_predicted'].std()
    
    stats_text = (f'DAB Statistics:\n'
                 f'Mean: {dab_mean:.2f}\n'
                 f'Std: {dab_std:.2f}\n'
                 f'True AHA: {true_aha:.1f}')
    
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Salva grafico
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Grafico temporale salvato: {output_path}")
    
    plt.show()


def plot_multiple_subjects_dab(data_path: Path, subject_ids: List[str], 
                               slope: float, intercept: float, output_dir: Path,
                               window_hours: Optional[int] = 24) -> None:
    """
    Genera grafici temporali DAB per multipli soggetti usando coefficienti già calcolati.
    """
    
    print(f"Generazione grafici temporali per {len(subject_ids)} soggetti...")
    print(f"Usando modello: DAB = {slope:.4f} * ST% + {intercept:.4f}")
    
    for subject_id in subject_ids:
        try:
            # Carica dati soggetto
            subject_data = load_temporal_data(data_path, subject_id)
            
            # Calcola DAB temporale
            temporal_dab = calculate_dab_temporal(subject_data, slope, intercept)
            
            # Genera grafico
            output_path = output_dir / f"dab_temporal_{subject_id}.png"
            plot_dab_temporal(temporal_dab, subject_id, output_path, window_hours)
            
        except Exception as e:
            print(f"Errore nel processare soggetto {subject_id}: {e}")
            continue
    
    print("Generazione grafici completata!")


def main():
    """
    Esempio di utilizzo per generare grafici temporali DAB.
    Usa coefficienti già calcolati dalla regressione lineare.
    """
    
    # Percorsi file
    base_dir = Path("c:/Users/sofia/Desktop/TESI2")
    temporal_data_path = base_dir / "temporal_data.csv"  # File con dati temporali
    output_dir = base_dir / "Plots" / "DAB_Temporal"
    
    # Lista soggetti da analizzare
    subject_ids = ["HC001", "HP001", "HC002", "HP002"]  # Esempio
    
    # Coefficienti dal modello già addestrato in linear_regression_analysis.py
    # Esempio: DAB = 0.8456 * ST% + 12.3456
    slope = 0.8456      # Sostituire con il valore reale dal tuo modello
    intercept = 12.3456 # Sostituire con il valore reale dal tuo modello
    
    try:
        # Genera grafici per tutti i soggetti
        plot_multiple_subjects_dab(
            temporal_data_path, 
            subject_ids, 
            slope,
            intercept, 
            output_dir,
            window_hours=24  # Finestra 24 ore
        )
        
    except Exception as e:
        print(f"Errore: {e}")


if __name__ == "__main__":
    main()