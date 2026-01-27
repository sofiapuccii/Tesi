import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import datetime

# === Funzione per plot a blocchi fissi di 6 ore ===
def plot_st_prob_blocks(timestamps, is_st_data, significativity_threshold, output_path, subject):
    """Plot ST% con blocchi fissi di 6 ore allineati all'orologio (00-06, 06-12, 12-18, 18-24)."""
    
    if isinstance(timestamps, pd.Series):
        
        timestamps = timestamps.reset_index(drop=True)
        
        if not pd.api.types.is_datetime64_any_dtype(timestamps):
            try:
               
                if timestamps.dtype in ['float64', 'int64'] and timestamps.iloc[0] > 1e9:
                    print(f"Rilevati timestamp Unix, conversione...")
                    timestamps = pd.to_datetime(timestamps, unit='s')
                else:
                    timestamps = pd.to_datetime(timestamps)
            except Exception as e:
                print(f"Errore conversione timestamp: {e}")
                # Fallback: crea timestamp finti basati su indici
                start_time = pd.Timestamp('2023-01-01 00:00:00')
                timestamps = pd.date_range(start=start_time, periods=len(timestamps), freq='240S')
                print(f"Usato timestamp sintetico: {start_time} + intervalli 240s")
    else:
       
        try:
            if timestamps[0] > 1e9: 
                timestamps = pd.to_datetime(timestamps, unit='s')
            else:
                timestamps = pd.to_datetime(timestamps)
        except:
            timestamps = pd.to_datetime(timestamps)
    
    print(f"Debug: Range timestamp dopo conversione: {timestamps.min()} → {timestamps.max()}")
    
    # Crea DataFrame per facilitare l'aggregazione
    df = pd.DataFrame({
        'timestamp': timestamps,
        'is_st': is_st_data
    })
    
    # Estrai giorni unici presenti nei dati
    df['date'] = df['timestamp'].dt.date
    unique_dates = sorted(df['date'].unique())
    
    block_timestamps = []
    block_percentages = []
    
    # Definisci i 4 blocchi di 6 ore per ogni giorno
    time_blocks = [
        (0, 6),   # 00:00 - 06:00
        (6, 12),  # 06:00 - 12:00  
        (12, 18), # 12:00 - 18:00
        (18, 0)   # 18:00 - 00:00 
    ]
    
    last_timestamp = df['timestamp'].max()
    
    # Itera sui giorni e blocchi
    for date in unique_dates:
        for start_hour, end_hour in time_blocks:
            # Crea timestamp di inizio blocco
            block_start = pd.Timestamp.combine(date, datetime.time(start_hour, 0))
            
            # Gestione speciale per il blocco 18:00-00:00 (end_hour = 0)
            if end_hour == 0:
                # Fine blocco = mezzanotte del giorno successivo
                next_date = date + datetime.timedelta(days=1)
                block_end = pd.Timestamp.combine(next_date, datetime.time(0, 0))
            else:
                # Blocco normale nello stesso giorno
                block_end = pd.Timestamp.combine(date, datetime.time(end_hour, 0))
            
            # se il blocco inizia dopo l'ultimo timestamp, fermati
            if block_start > last_timestamp:
                break
                
            # Filtra dati nel blocco temporale
            block_mask = (df['timestamp'] >= block_start) & (df['timestamp'] < block_end)
            block_data = df[block_mask]['is_st']
            
            # Calcola percentuale ST per il blocco
            if len(block_data) == 0:
                # Blocco vuoto: assegna 0.0 per disegnare trattino piatto
                st_percentage = 0.0
            else:
                n_st = block_data.sum()  # Conta gli 1 (ST)
                n_total = len(block_data)
                
                # Controllo significatività 
                if n_total < (len(df) * significativity_threshold / 100 / (len(unique_dates) * 4)):
                    st_percentage = np.nan  
                else:
                    st_percentage = (n_st / n_total) * 100
            
            # Aggiungi timestamp del centro del blocco
            block_center = block_start + pd.Timedelta(hours=3)
            block_timestamps.append(block_center)
            block_percentages.append(st_percentage)
        
        # Se abbiamo superato l'ultimo timestamp, ferma tutto
        if block_start > last_timestamp:
            break
    
    # Plotting con Step Plot
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # Grafico a gradini rettangolari
    ax.step(block_timestamps, block_percentages, where='post', 
            linewidth=2.5, color='steelblue', markersize=6)
    
    # Linee verticali alle mezzanotti per separare i giorni
    for date in unique_dates[1:]:  # Salta il primo giorno
        midnight = pd.Timestamp.combine(date, datetime.time(0, 0))
        if midnight <= last_timestamp:
            ax.axvline(midnight, color='grey', linestyle='-', alpha=0.7, linewidth=2)
    
    # Configurazione assi
    ax.set_ylim([0, 100])
    ax.set_yticks([0, 50, 100])
    ax.grid(True, alpha=0.3)
    
    # Formattazione asse X
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
    
    # Labels e titolo
    ax.set_xlabel("Orario", fontsize=12, fontweight='bold')
    ax.set_ylabel("% sample ST", fontsize=12, fontweight='bold')
    #ax.set_title(f"Soggetto {subject} - ST% per Blocchi 6h Fissi", fontsize=14, fontweight='bold')
    
    # Salva grafico
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Grafico ST blocchi fissi salvato: {output_path}")
    plt.close()

# === Funzione per plot smooth (finestra scorrevole) === DA SISTEMARE
def plot_st_prob_smooth(timestamps, prob_st, windows_per_block, output_path):
    smooth_st_means = []
    smooth_timestamps = []
    for i in range(len(prob_st) - windows_per_block + 1):
        block_data = prob_st[i:i+windows_per_block]
        st_mean = np.mean(block_data) * 100
        smooth_st_means.append(st_mean)
        smooth_timestamps.append(timestamps[i + windows_per_block//2])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(smooth_timestamps, smooth_st_means, linewidth=2, color='steelblue')
    ax.set_ylim([0, 100])
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel('% sample ST (media probabilità)')
    ax.set_xlabel('Orario')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    ax.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=0.8)
    ax.grid(True, which='minor', alpha=0.3, linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Carica il file CSV esportato da clinical_analysis.py
    df = pd.read_csv("../results/clinical_analysis_v2/temporal_predictions.csv")
    subject_id = 42  # Cambia con il soggetto che vuoi plottare
    subject_df = df[df['subject_id'] == subject_id].copy()
    
    if len(subject_df) == 0:
        print(f"Errore: Nessun dato trovato per soggetto {subject_id}")
        exit(1)
    
    print(f"Dati grezzi trovati: {len(subject_df)} righe")
    print(f"Sample timestamp grezzo: {subject_df['timestamp'].iloc[0]}")
    
    # Converti timestamps 
    try:
        # Controlla se è un timestamp Unix (numero > 1e9)
        sample_timestamp = subject_df['timestamp'].iloc[0]
        if isinstance(sample_timestamp, (int, float)) and sample_timestamp > 1e9:
            print(f"Rilevato timestamp Unix: {sample_timestamp}")
            timestamps = pd.to_datetime(subject_df['timestamp'], unit='s')
            print(f"Conversione Unix timestamp riuscita")
        else:
            timestamps = pd.to_datetime(subject_df['timestamp'])
            print(f"Conversione timestamp standard riuscita")
    except Exception as e:
        print(f"Errore conversione timestamp: {e}")
        print("Creando timestamp sintetici...")
        start_time = pd.Timestamp('2023-01-01 00:00:00')
        timestamps = pd.date_range(start=start_time, periods=len(subject_df), freq='240S')
    
    # Usa la colonna is_st (dati binari 0/1)
    if 'is_st' in subject_df.columns:
        is_st_data = subject_df['is_st'].values.astype(int)
    else:
        raise ValueError("Il file deve contenere la colonna is_st!")
    
    # Parametri
    significativity_threshold = 75  # Soglia di significatività
    
    print(f"Elaborando soggetto {subject_id}: {len(timestamps)} finestre temporali")
    print(f"Periodo: {timestamps.min()} → {timestamps.max()}")
    print(f"ST positivi: {is_st_data.sum()}/{len(is_st_data)} ({is_st_data.mean()*100:.1f}%)")
    
    # Grafico a blocchi fissi di 6 ore
    plot_st_prob_blocks(timestamps, is_st_data, significativity_threshold, 
                       Path(f"st_blocks_fixed_subject{subject_id}.png"), 
                       subject=subject_id)
    
    # Grafico smooth 
    windows_per_block = 90  #  6h se ogni finestra è 4 minuti
    plot_st_prob_smooth(timestamps, is_st_data, windows_per_block, 
                       Path(f"st_smooth_subject{subject_id}.png"))
    
    print(f"Grafici generati per soggetto {subject_id}")