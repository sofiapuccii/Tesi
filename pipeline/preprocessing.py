"""
Preprocessing module: loading, synchronization, interpolation, filtering, and windowing.
All parameters and paths come from pipeline/config.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from .cache import get_memory

'''Questo file serve per caricare i segnali grezzi,sincronizzarli e pulirli, filtrarli, normalizzare i timestamp per l'analisi successiva'''

@dataclass 
class PreprocessParams: # classe per i parametri di preprocessing
    sampling_rate: int
    window_size: int
    overlap: float

# funzione per il filtro butterworth
def _butter_lowpass(x: np.ndarray, sr: float, cutoff: float = 20.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * sr # frequenza di Nyquist che rappresenta la metà della frequenza di campionamento
    b, a = butter(order, cutoff / nyq, btype="low") # filtro butterworth lowpass
    return filtfilt(b, a, x) # applica il filtro Butterworth da dx a sx e viceversa per rimuovere le componenti alte frequenze


def _moving_average(x: np.ndarray, k: int) -> np.ndarray: 
    if k <= 1:
        return x # se la finestra k è <=1 restituisce l'array orginale
    kernel = np.ones(k) / k # crea un array di dimensione k riempito con il valore 1.0
    return np.convolve(x, kernel, mode="same")

#converte la colonna timestamp in un dataframe in secondi Unix coerenti anche se i dati sono scritti in formati diversi
def _parse_timestamps_robust(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust timestamp parsing handling various formats:
    - ISO datetime strings (with/without timezone)
    - Unix timestamps (seconds, milliseconds, microseconds)
    - Custom datetime formats
    - Mixed formats per subject
    """
    import re #per il parsing delle stringhe
    from datetime import datetime, timezone #per il parsing delle date e delle ore
    import pytz #per la gestione dei fusi orari
    
    df = df.copy() #primo step si copiano i dati orginali
    original_timestamps = df["timestamp"].copy() #si copia la colonna timestamp in un array
    
    parsed_timestamps = [] #array per salvare i timestamp parseati
    parsing_errors = [] #array per salvare gli errori di parsing
    
    for idx, ts in enumerate(original_timestamps): #per ogni timestamp si prova a parsearlo
        parsed_ts = None 
        error_msg = None 
        
        try:
            # Strategia 1: se timestamp numerico
            if pd.api.types.is_numeric_dtype(type(ts)) or isinstance(ts, (int, float)): 
                ts_num = float(ts) #si converte il timestamp in un numero float
                if ts_num > 1e12:  
                    parsed_ts = ts_num / 1000.0 # millisecondi -> secondi
                elif ts_num > 1e15:  
                    parsed_ts = ts_num / 1e6 # microsecondi -> secondi
                else: 
                    parsed_ts = ts_num # già in secondi
                parsed_timestamps.append(parsed_ts) #si aggiunge il timestamp parseato all'array
                continue
            
            # Strategia 2: parsing string con multipli formati
            ts_str = str(ts).strip() #rimuove gli spazi e ocnverte in stringa 
            
            # Formati datetime comuni da provare
            formats = [
                "%Y-%m-%d %H:%M:%S.%f",  # 2023-01-01 12:30:45.123456
                "%Y-%m-%d %H:%M:%S",     # 2023-01-01 12:30:45
                "%Y-%m-%dT%H:%M:%S.%f", # 2023-01-01T12:30:45.123456
                "%Y-%m-%dT%H:%M:%S",    # 2023-01-01T12:30:45
                "%Y-%m-%dT%H:%M:%S.%fZ", # 2023-01-01T12:30:45.123456Z
                "%Y-%m-%dT%H:%M:%SZ",   # 2023-01-01T12:30:45Z
                "%Y/%m/%d %H:%M:%S",    # 2023/01/01 12:30:45
                "%d/%m/%Y %H:%M:%S",    # 01/01/2023 12:30:45
                "%m/%d/%Y %H:%M:%S",    # 01/01/2023 12:30:45
            ]
            
            # prova a far capire a pandas il formato datetime con diverse opzioni
            for infer_datetime_format in [True, False]: 
                for utc in [True, False]: # prova sia con che senza conversione in UTC
                    try:
                        dt = pd.to_datetime(ts_str, infer_datetime_format=infer_datetime_format, utc=utc) # converte la stringa in datetime
                        if not pd.isna(dt): # se il datetime non è NaN
                            # Converte in timestamp Unix (secondi dall'epoca)
                            parsed_ts = dt.timestamp()
                            break
                    except:
                        continue
                if parsed_ts is not None: # se il timestamp è stato parseato correttamente si esce dal ciclo
                    break
            
            # Strategia 3: parsing manuale del formato datetime
            if parsed_ts is None: # se il timestamp non è stato parseato correttamente si prova a parsearlo manualmente
                for fmt in formats:
                    try:
                        dt = datetime.strptime(ts_str, fmt) # prova ad analizzare la stringa con il formato datetime 
                    
                        if dt.tzinfo is None: # se il datetime non ha timezone si converte in UTC
                            dt = dt.replace(tzinfo=timezone.utc) 
                        parsed_ts = dt.timestamp()
                        break
                    except ValueError:
                        continue
            
            # Strategia 4: gestione delle stringhe con timezone
            if parsed_ts is None:
                try:
                    # prova a parseare la stringa con la timezone info
                    dt = pd.to_datetime(ts_str, utc=True) # converte la stringa in datetime con la timezone info
                    if not pd.isna(dt): # se il datetime non è NaN
                        parsed_ts = dt.timestamp() # si converte il datetime in timestamp Unix (secondi dall'epoca)
                except:
                    pass
            
            if parsed_ts is None:
                error_msg = f"Could not parse timestamp: {ts_str}" # messaggio di errore
                parsing_errors.append((idx, ts_str, error_msg)) # si aggiunge l'errore all'array
                parsed_ts = np.nan # si imposta il timestamp a NaN
            
        except Exception as e:
            error_msg = f"Parsing error: {str(e)}"
            parsing_errors.append((idx, str(ts), error_msg))
            parsed_ts = np.nan
        
        parsed_timestamps.append(parsed_ts) # si aggiunge il timestamp parseato all'array
    
    # si conta il numero di timestamp validi
    valid_count = sum(1 for ts in parsed_timestamps if not pd.isna(ts)) # si conta il numero di timestamp validi
    total_count = len(parsed_timestamps) # si conta il numero totale di timestamp
    
    print(f"Timestamp parsing results:")
    print(f"  - Valid timestamps: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)") # si stampa il numero di timestamp validi e il totale
    
    if parsing_errors:
        print(f"  - Parsing errors: {len(parsing_errors)}")
        # Show first few errors for debugging
        for i, (idx, ts_str, error) in enumerate(parsing_errors[:5]):
            print(f"    Error {i+1}: {error}")
        if len(parsing_errors) > 5: 
            print(f"    ... and {len(parsing_errors)-5} more errors")
    
    # si aggiorna la colonna timestamp con i timestamp parseati
    df["timestamp"] = parsed_timestamps
    
    # si normalizza i timestamp per ogni subject
    df = _normalize_timestamps_per_subject(df) # si normalizza i timestamp per ogni subject
    
    return df # si restituisce il dataframe con i timestamp parseati

def _normalize_timestamps_per_subject(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizza i timestamp per ogni subject per gestire differenti fusi orari e tempi di inizio.
    I timestamp di ogni subject sono regolati per iniziare da 0 e avere un campionamento coerente.
    """
    normalized_frames = [] # array per salvare i dataframe normalizzati
    for subject_id, group in df.groupby("subject_id"): #divide il dataframe in blocchi, uno per ogni valore di subject_id 
        group = group.copy()
        
        if len(group) == 0: #se il gruppo è vuoto si passa al prossimo subject
            continue
        
        # si rimuovono i NaN dalla colonna timestamp
        group = group.dropna(subset=["timestamp"])
        
        if len(group) == 0:
            continue
        
        # ordinamento temporale dei timestamp
        group = group.sort_values("timestamp") #passaggio importante per poter effettuare differenze temporali, garantire coerenza e stimare la frequenza di campionamento
        
        # controllo della consistenza dei timestamp
        timestamps = group["timestamp"].values
        if len(timestamps) > 1:
            
            time_diffs = np.diff(timestamps) #calcola gli intervalli tra un timestamp e il successivo
            median_diff = np.median(time_diffs) #calcolo ogni quanti secondi viene acquisito un nuovo campione
            
            # se la differenza tra i timestamp è <= 0 si stampa un warning e si aggiunge un millisecondo al timestamp successivo
            if median_diff <= 0:
                print(f"Warning: Subject {subject_id} has non-increasing timestamps")
                
                for i in range(1, len(timestamps)): 
                    if timestamps[i] <= timestamps[i-1]: #se il timestamp è <= al timestamp precedente 
                        timestamps[i] = timestamps[i-1] + 0.001  # si aggiunge un millisecondo al timestamp successivo
                group["timestamp"] = timestamps #si aggiorna la colonna timestamp con i timestamp normalizzati
            
            # si normalizza i timestamp per ogni subject
            min_time = timestamps.min() #si calcola il timestamp minimo
            group["timestamp"] = timestamps - min_time # si normalizza i timestamp per ogni subject
            
            # se il numero di timestamp è > 1 si calcola la frequenza di campionamento
            if len(timestamps) > 1:
                actual_sr = 1.0 / np.median(time_diffs) #si calcola la frequenza di campionamento
                print(f"Subject {subject_id}: {len(group)} samples, estimated SR: {actual_sr:.1f} Hz") 
        
        normalized_frames.append(group) #si aggiunge il dataframe normalizzato all'array
    
    if normalized_frames:
        result = pd.concat(normalized_frames, ignore_index=True) #si concatena il dataframe normalizzato con gli altri dataframe normalizzati
        print(f"Normalized timestamps for {result['subject_id'].nunique()} subjects") #si stampa il numero di subject normalizzati
        return result
    else:
        print("Warning: No valid data after timestamp normalization")
        return df #si restituisce il dataframe con i timestamp normalizzati


def load_csv_folder(data_dir: Path) -> pd.DataFrame:
    """
    Load and combine multiple session files (subject_X_AHA.csv, subject_X_WEEK.csv).
    Handles date/time parsing and assigns correct labels based on session type.
    """
    import re
    from datetime import datetime
    
    # Find all subject files with pattern subject_X_AHA.csv or subject_X_WEEK.csv
    # Check both direct directory and subdirectories
    aha_files = sorted(data_dir.glob("subject_*_AHA.csv"))
    week_files = sorted(data_dir.glob("subject_*_WEEK.csv"))
    
    # If no files found in main directory, check subdirectories
    if not aha_files and not week_files:
        aha_files = sorted(data_dir.glob("AHA/subject_*_AHA.csv"))
        week_files = sorted(data_dir.glob("WEEK/subject_*_WEEK.csv"))
    
    if not aha_files and not week_files:
        raise FileNotFoundError(f"No subject files found in {data_dir}. Expected pattern: subject_X_AHA.csv or subject_X_WEEK.csv")
    
    frames: List[pd.DataFrame] = [] #array per salvare i dataframe letti
    
    # Process AHA files (label = 1 for hemiplegia)
    for file_path in aha_files: #per ogni file AHA
        try:
            df = pd.read_csv(file_path)
            # si estrae l'ID del subject dal nome del file
            match = re.search(r'subject_(\d+)_AHA\.csv', file_path.name)
            if match:
                subject_id = f"subject_{match.group(1)}_AHA" #si crea l'ID del subject
            else:
                subject_id = f"subject_unknown_AHA"
            
            # se la colonna subject_id non è presente si aggiunge
            if "subject_id" not in df.columns:
                df["subject_id"] = subject_id #si aggiunge la colonna subject_id
            if "label" not in df.columns:
                df["label"] = 1  # AHA = hemiplegia
            
            # se la colonna session_type non è presente si aggiunge
            df["session_type"] = "AHA" #si aggiunge la colonna session_type
            frames.append(df) #si aggiunge il dataframe all'array
            
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
            continue
    
    # stesso processo per i file WEEK
    for file_path in week_files:
        try:
            df = pd.read_csv(file_path)
            # si estrae l'ID del subject dal nome del file
            match = re.search(r'subject_(\d+)_WEEK\.csv', file_path.name)
            if match:
                subject_id = f"subject_{match.group(1)}_WEEK"
            else:
                subject_id = f"subject_unknown_WEEK"
            
            # se la colonna subject_id non è presente si aggiunge
            if "subject_id" not in df.columns:
                df["subject_id"] = subject_id
            if "label" not in df.columns:
                df["label"] = 0  # WEEK = control
            
            # se la colonna session_type non è presente si aggiunge
            df["session_type"] = "WEEK"
            frames.append(df)
            
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
            continue
    
    if not frames:
        raise ValueError("No valid CSV files could be loaded")
    
    # si concatena il dataframe con gli altri dataframe
    data = pd.concat(frames, ignore_index=True)
    
    # si mappa il nome delle colonne al formato standard se necessario
    column_mapping = {
        'x_D': 'x_dom', 'y_D': 'y_dom', 'z_D': 'z_dom',
        'x_ND': 'x_non', 'y_ND': 'y_non', 'z_ND': 'z_non',
        'datetime': 'timestamp'
    }
    
    # se le colonne esistono si rinominano
    for old_name, new_name in column_mapping.items():
        if old_name in data.columns and new_name not in data.columns:
            data = data.rename(columns={old_name: new_name})
    
    # si controllano le colonne richieste
    required = ["timestamp", "x_dom", "y_dom", "z_dom", "x_non", "y_non", "z_non", "subject_id", "label"]
    missing = [c for c in required if c not in data.columns] 
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # parsing robusto dei timestamp
    data = _parse_timestamps_robust(data)
    
    # si rimuovono le righe con timestamp invalidi
    data = data.dropna(subset=["timestamp"])
    
    # ordinamento temporale dei timestamp per subject
    data = data.sort_values(["subject_id", "timestamp"]).reset_index(drop=True)
    
    print(f"Loaded {len(frames)} files:")
    print(f"  - AHA files: {len(aha_files)} (hemiplegia, label=1)")
    print(f"  - WEEK files: {len(week_files)} (control, label=0)")
    print(f"  - Total subjects: {data['subject_id'].nunique()}")
    print(f"  - Total samples: {len(data)}")
    print(f"  - Label distribution: {data['label'].value_counts().to_dict()}")
    
    return data

# funzione per la sincronizzazione e l'interpolazione dei dati
def synchronize_and_interpolate(df: pd.DataFrame, sampling_rate: int) -> pd.DataFrame:
    """
    Sincronizza e interpola i dati per ogni subject, gestendo multiple sessioni.
    I dati di ogni subject sono resampled a una griglia uniforme alla frequenza di campionamento target.
    Gestione robusta dei problemi di allineamento temporale.
    """
    dt = 1.0 / float(sampling_rate) #si calcola l'intervallo di tempo tra due campioni
    out: List[pd.DataFrame] = [] #array per salvare i dataframe sincronizzati e interpolati
    
    for sid, g in df.groupby("subject_id", sort=False): #divide il dataframe in blocchi, uno per ogni valore di subject_id
        g = g.sort_values("timestamp").copy() #si ordina il dataframe per timestamp
        
        if len(g) == 0:
            print(f"Warning: Subject {sid} has no data")
            continue
        
        # controllo della consistenza temporale
        timestamps = g["timestamp"].values #si estrae il vettore dei timestamp
        if len(timestamps) > 1:
            time_diffs = np.diff(timestamps) #calcola gli intervalli tra un timestamp e il successivo
            median_diff = np.median(time_diffs) #calcolo ogni quanti secondi viene acquisito un nuovo campione
            actual_sr = 1.0 / median_diff if median_diff > 0 else sampling_rate # frequenza=1/periodo
            
            # se la frequenza di campionamento è molto differente dalla frequenza di campionamento target si stampa un warning
            if abs(actual_sr - sampling_rate) > sampling_rate * 0.5:  # 50% tolerance
                print(f"Warning: Subject {sid} has estimated SR {actual_sr:.1f} Hz (expected {sampling_rate} Hz)")
        t0, t1 = g["timestamp"].iloc[0], g["timestamp"].iloc[-1] #si calcola il timestamp minimo e il timestamp massimo
        grid = np.arange(t0, t1 + dt/2, dt)  # creazione griglia esmepio: set t0=0, t1=1.9 e dt=0.5 --> [0,0.5,1.0,1.5,2.0]
        
        # si imposta il timestamp come indice per il resampling
        gi = g.set_index("timestamp") 
        
        
        gi = gi.reindex(gi.index.union(grid)).sort_index() # unisce i timestamp originali con la griglia target, e poi sort() riordina l'indice risultante
        
        # seleziona le colonne numeriche (segnali accelerometrici)
        numeric_cols = [c for c in gi.columns if c not in ("label", "subject_id", "session_type")]
        for col in numeric_cols:
            if col in gi.columns:
                # si interpola linearmente i segnali accelerometrici
                gi[col] = gi[col].interpolate(method="linear", limit_direction="both") 
        
        # si riempie il valore mancante per le colonne categoriche
        gi["label"] = gi["label"].ffill().bfill() #propaga il valore mancante in avanti e indietro
        if "session_type" in gi.columns:
            gi["session_type"] = gi["session_type"].ffill().bfill() 
        
        # si estrae i dati alla nuova griglia
        aligned = gi.loc[grid].reset_index().rename(columns={"index": "timestamp"}) # seleziona solo i punti della griglia temporale, converte l'inidce in colonna e rinomina la colonna indice in "timestamp"
        aligned["subject_id"] = sid 
        
        # si assegna il valore della label più frequente
        if len(aligned) > 0:
            label_mode = aligned["label"].mode() #calcola la moda della label
            if len(label_mode) > 0:
                aligned["label"] = label_mode.iloc[0] #assegna il valore della label più frequente  
            else:
                # se non c'è moda, si assegna il primo valore non nullo
                non_null_labels = aligned["label"].dropna()
                if len(non_null_labels) > 0:
                    aligned["label"] = non_null_labels.iloc[0] #assegna il primo valore non nullo
        
        # controllo della qualità: si controlla se c'è un numero di campioni ragionevole
        if len(aligned) < 10:  # avvisa se un soggetto ha troppi pochi campioni
            print(f"Warning: Subject {sid} has only {len(aligned)} samples after synchronization")
        
        out.append(aligned) #si aggiunge il dataframe sincronizzato e interpolato all'array
    
    if not out:
        raise ValueError("No valid data after synchronization")
    
    result = pd.concat(out, ignore_index=True) #si concatena il dataframe sincronizzato e interpolato con gli altri dataframe sincronizzati e interpolati
    print(f"Synchronized data: {len(result)} samples across {result['subject_id'].nunique()} subjects") #si stampa il numero di campioni sincronizzati e interpolati e il numero di soggetti
    
    # controllo della qualità: si controlla se c'è un numero di campioni ragionevole per ogni subject
    for sid, group in result.groupby("subject_id"):
        if len(group) == 0:
            print(f"Warning: Subject {sid} has no data after synchronization")
        elif len(group) < 50:  
            print(f"Warning: Subject {sid} has only {len(group)} samples")
    
    return result

# funzione per il filtro dei segnali
def filter_signals(df: pd.DataFrame, sampling_rate: int, method: str = "butter", ma_window_ms: int = 200) -> pd.DataFrame:
    df = df.copy() #si copia il dataframe
    #for _, g in df.groupby("subject_id"): #divide il dataframe in blocchi, uno per ogni valore di subject_id
     #   pass
    if method == "butter": #se il metodo è butterworth
        for col in ["x_dom","y_dom","z_dom","x_non","y_non","z_non"]: #per ogni colonna dei segnali accelerometrici
            df[col] = _butter_lowpass(df[col].to_numpy(dtype=float), sampling_rate, cutoff=20.0)
    elif method == "moving_average": #se il metodo è moving average
        k = max(1, int(round((ma_window_ms / 1000.0) * sampling_rate))) #si calcola la finestra di smoothing
        for col in ["x_dom","y_dom","z_dom","x_non","y_non","z_non"]:
            df[col] = _moving_average(df[col].to_numpy(dtype=float), k) #si applica il filtro moving average
    else:
        raise ValueError("Unknown filter method")
    # magnitude
    for wrist in ("dom","non"): #per ogni polso
        x,y,z = df[f"x_{wrist}"].astype(float), df[f"y_{wrist}"].astype(float), df[f"z_{wrist}"].astype(float) #si convertono le colonne in float
        df[f"mag_{wrist}"] = np.sqrt(x*x + y*y + z*z)
    return df

# funzione per la segmentazione dei dati
def window_segments(df: pd.DataFrame, window_size: int, overlap: float) -> pd.DataFrame:
    step = int(round(window_size * (1.0 - float(overlap)))) #si calcola il passo di avanzamento della finestra
    rows: List[Dict] = []
    for sid, g in df.groupby("subject_id", sort=False): #divide il dataframe in blocchi, uno per ogni valore di subject_id
        g = g.reset_index(drop=True) #si resetta l'indice
        n = len(g) #si calcola il numero di campioni
        s = 0
        while s + window_size <= n: #si crea la finestra
            e = s + window_size #si calcola l'indice di fine finestra
            seg = g.iloc[s:e]
            rows.append({
                "subject_id": sid, #si aggiunge l'ID del subject
                "start_idx": s, #si aggiunge l'indice di inizio finestra
                "end_idx": e, #si aggiunge l'indice di fine finestra
                "start_time": float(seg["timestamp"].iloc[0]), #si aggiunge il timestamp di inizio finestra
                "end_time": float(seg["timestamp"].iloc[-1]), #si aggiunge il timestamp di fine finestra
                "label": seg["label"].mode().iloc[0],
            }) #si aggiunge il dataframe alla lista
            s += step if step > 0 else window_size #si calcola l'indice di inizio finestra successivo
    return pd.DataFrame(rows) #si restituisce il dataframe con le finestre

# funzione per salvare i dati preprocessati
def save_preprocessed(df: pd.DataFrame, windows: pd.DataFrame, out_dir: Path) -> None: 
    out_dir.mkdir(parents=True, exist_ok=True) #si crea la directory se non esiste
    df.to_csv(out_dir / "signals.csv", index=False) #si salva il dataframe dei segnali
    windows.to_csv(out_dir / "windows.csv", index=False) #si salva il dataframe delle finestre

# funzione per il preprocessing dei dati
def preprocess_signals(data_dir: Path, output_dir: Path, window_size: int, overlap: float, sampling_rate: int) -> Tuple[pd.DataFrame, pd.DataFrame]: #restituisce il dataframe dei segnali e il dataframe delle finestre
    """
    Carica, sincronizza e segmenta i dati accelerometrici da multiple sessioni.
    Gestisce automaticamente subject_X_AHA.csv (hemiplegia, label=1) e subject_X_WEEK.csv (control, label=0).
    Restituisce (signals_df, windows_df) con finestre pronte per feature extraction e
    salva i CSV in results/preprocessed/.
    Questa funzione usa la caching per evitare di ricalcolare operazioni costose quando i parametri non sono cambiati.
    """
    # si ottiene l'istanza della memoria cache
    memory = get_memory()
    
    # si creano le versioni cachate delle funzioni expensive
    cached_synchronize = memory.cache(synchronize_and_interpolate) #si cacha la funzione synchronize_and_interpolate
    cached_filter = memory.cache(filter_signals) #si cacha la funzione filter_signals
    cached_window = memory.cache(window_segments) #si cacha la funzione window_segments
    
    print("Loading multi-session data...") #si stampa il messaggio di caricamento dei dati multi-sessione
    raw = load_csv_folder(data_dir) #si caricano i dati multi-sessione
    
    print("Synchronizing and interpolating (cached)...") #si stampa il messaggio di sincronizzazione e interpolazione
    synced = cached_synchronize(raw, sampling_rate) #si sincronizza e si interpola i dati
    
    print("Filtering signals (cached)...") #si stampa il messaggio di filtraggio dei segnali
    filtered = cached_filter(synced, sampling_rate, method="butter") #si filtrano i segnali
    
    print("Creating windows (cached)...")
    windows = cached_window(filtered, window_size, overlap) #si creano le finestre
    
    print("Saving preprocessed data...")
    save_preprocessed(filtered, windows, output_dir / "preprocessed") #si salvano i dati preprocessati
    
    print(f"Preprocessing complete:")
    print(f"  - Signals: {len(filtered)} samples")
    print(f"  - Windows: {len(windows)} windows")
    print(f"  - Subjects: {filtered['subject_id'].nunique()}")
    print(f"  - Label distribution: {filtered['label'].value_counts().to_dict()}")
    
    return filtered, windows


