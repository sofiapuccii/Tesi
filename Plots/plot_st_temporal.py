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
    ax.xaxis.set_major_locator(mdates.DayLocator())  # Tick a mezzanotte ogni giorno
    
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

# === Funzione per plot smooth (finestra scorrevole) ===
def plot_st_prob_smooth(timestamps, is_st_data , window_hours, output_path):
  # Crea DataFrame con timestamp come indice
    df = pd.DataFrame({'is_st': is_st_data}, index=pd.to_datetime(timestamps))
    df = df.sort_index()

    # Calcola la percentuale di ST su una finestra mobile di 6 ore
    rolling = df['is_st'].rolling(f'{window_hours}H', min_periods=1).mean() * 100

    # Plot
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(rolling.index, rolling.values, color='steelblue', linewidth=2.5)

    ax.set_ylim([0, 100])
    ax.set_yticks([0, 50, 100])
    ax.grid(True, alpha=0.3) 

    # Formattazione asse X come in plot_st_prob_blocks
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.DayLocator())  # Tick a mezzanotte ogni giorno

    ax.set_xlabel("Orario", fontsize=12, fontweight='bold')
    ax.set_ylabel("% sample ST", fontsize=12, fontweight='bold')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Grafico ST smooth (rolling) salvato: {output_path}")
    plt.close()


def plot_dab_smooth_from_binary(timestamps, is_st_binary, true_aha, regressor, output_path, subject_id, window_hours=6):
    
    # 1. Preparazione DataFrame
    df = pd.DataFrame({
        'is_st': is_st_binary
    }, index=pd.to_datetime(timestamps))
    
    df = df.sort_index()
  
    df['st_rolling_percent'] = df['is_st'].rolling(window=f'{window_hours}H', min_periods=10).mean() * 100
    
    # 3. Applicazione del Regressore
    # DAB = (ST% * slope) + intercept
    
    # Gestione input regressore (Tupla manuale o Oggetto Sklearn)
    if isinstance(regressor, (list, tuple, np.ndarray)):
        slope, intercept = regressor
        df['dab_predicted'] = (df['st_rolling_percent'] * slope) + intercept
    elif hasattr(regressor, 'predict'):
        # Logica per oggetto sklearn
        valid_mask = ~df['st_rolling_percent'].isna()
        predictions = np.full(len(df), np.nan)
        if valid_mask.sum() > 0:
            X = df.loc[valid_mask, 'st_rolling_percent'].values.reshape(-1, 1)
            predictions[valid_mask] = regressor.predict(X).flatten()
        df['dab_predicted'] = predictions
    else:
        # Fallback se passi solo numeri sfusi non in tupla
        raise ValueError("Il regressore deve essere una tupla (slope, intercept) o un oggetto sklearn")

    # 4. Clipping (0-100)
    df['dab_clipped'] = df['dab_predicted'].clip(0, 100)
    
    # 5. Plotting
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Linea DAB (Verde continua)
    ax.plot(df.index, df['dab_clipped'], 
            color='forestgreen', linewidth=2.5)
    
    # Linea Ground Truth
    ax.axhline(y=true_aha, color='royalblue', linestyle='--', 
               linewidth=2, label=f'True AHA Score)')
    
    # Estetica
    ax.set_ylim(-5, 105)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel('DAB', fontweight='bold')
    ax.set_xlabel('Orario', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Formattazione Date
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) # Solo orario
    ax.xaxis.set_major_locator(mdates.DayLocator())  # Tick a mezzanotte ogni giorno
    
    ax.legend(loc='upper right', frameon=True)
    #plt.title(f'Subject {subject_id} - Temporal Home-AHA (From Binary Labels)', pad=15)
    
    # Salvataggio
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Grafico salvato: {output_path}")
    plt.close()

if __name__ == "__main__":
    # Carica il file CSV esportato da clinical_analysis.py
    df = pd.read_csv("../results/clinical_analysis_v2/temporal_predictions.csv")
    subject_id = 11  # Cambia con il soggetto che vuoi plottare
    subject_df = df[df['subject_id'] == subject_id].copy()
    regressor_params = (4.53327, 37.51612)  # Esempio: slope=0.5, intercept=10


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
    plot_st_prob_smooth(timestamps, is_st_data, window_hours=6, output_path= Path(f"st_smooth_subject{subject_id}.png"))
    plot_dab_smooth_from_binary(timestamps= timestamps, is_st_binary= is_st_data, true_aha= subject_df['true_aha_score'].iloc[0], regressor= regressor_params, output_path= Path(f"dab_smooth_from_binary_subject{subject_id}.png"), subject_id= subject_id)
    
    print(f"Grafici generati per soggetto {subject_id}")