from __future__ import annotations
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.signal import decimate

def load_subject_metadata(metadata_file: Path) -> Tuple[Dict[int, int], Dict[int, float]]: 
   
    try:
        df_meta = pd.read_excel(metadata_file)
        subject_labels = {}
        subject_aha_scores = {}
        
        for _, row in df_meta.iterrows():
            subject_id = int(row['subject'])
            hemi = int(row['hemi'])
            
            #(1->0=controllo, 2->1=emiplegico)
            if hemi == 1:
                subject_labels[subject_id] = 0  # controllo sano
            elif hemi == 2:
                subject_labels[subject_id] = 1  # emiplegico
            else:
                continue
        
            # Carica AHA score 
            subject_aha_scores[subject_id] = float(row['AHA'])
        
        return subject_labels, subject_aha_scores # ritorna le etichette e i punteggi AHA
        
    except Exception as e:
        print(f"Error loading metadata file: {e}")
        return {}, {}


def _load_aha_file(file_path: Path, subject_labels: Dict[int, int], subject_aha_scores: Dict[int, float]) -> Optional[pd.DataFrame]:
    """Load and process di un singolo file AHA."""
    try:
        df = pd.read_csv(file_path, decimal=',')
        
        # estare l'id del paziente
        import re
        match = re.search(r'(\d+)_AHA_RAW\.csv', file_path.name) # estraiamo l'id del paziente dal filename
        subject_id = f"subject_{match.group(1)}_AHA" # id del paziente
        subject_num = int(match.group(1)) 
        label = subject_labels[subject_num] # assegniamo la label dal metadata
        
        # aggiungiamo le colonne richieste
        df["subject_id"] = subject_id
        df["label"] = label
        df["AHA"] = float(subject_aha_scores[subject_num])  # punteggio AHA sempre presente
        df["session_type"] = "AHA" 
        
        return df
        
    except Exception as e:
        print(f"Warning: Could not load {file_path}: {e}")
        return None

    
def _load_week_file(file_path: Path) -> Optional[pd.DataFrame]:
    """Load and process a single WEEK CSV file."""
    try:
        df = pd.read_csv(file_path, decimal=',')
        
        import re
        match = re.search(r'(\d+)_week_RAW\.csv', file_path.name) # estraiamo l'id del paziente dal filename
        subject_id = f"subject_{match.group(1)}_WEEK" # id del paziente
        
        # aggiungiamo le colonne richieste
        df["subject_id"] = subject_id
        df["session_type"] = "WEEK"
        
        return df

    except Exception as e:
        print(f"Warning: Could not load {file_path}: {e}")
        return None


def _apply_common_preprocessing(df: pd.DataFrame, is_week_data: bool = False) -> pd.DataFrame:
    """Apply common preprocessing including timestamp conversion."""

    # Rinomina direttamente datetime -> timestamp se necessario
    if 'datetime' in df.columns:
        df = df.rename(columns={'datetime': 'timestamp'})
    
    # Converte timestamp in formato Unix
    if 'timestamp' not in df.columns:
        raise ValueError("No timestamp or datetime column found")
    
    df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') / 1e9 # converti a secondi 

    return df


def load_csv_folder_multiprocess(data_dir: Path, metadata_file: Path = None, n_workers: int = None, 
                                 load_only: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)  # limitiamo il numero di workers a 8
    
    # carichiamo il metadata (etichette e punteggi AHA)
    subject_labels = {}
    subject_aha_scores = {}
    if metadata_file is not None: 
        subject_labels, subject_aha_scores = load_subject_metadata(metadata_file) # carichiamo il metadata
    
    # troviamo i file separatamente
    aha_files = sorted(data_dir.glob("AHA/*_AHA_RAW.csv")) if load_only != 'WEEK' else []  
    week_files = sorted(data_dir.glob("week/*_week_RAW.csv")) if load_only != 'AHA' else []
    
    if load_only == 'AHA':
        print(f"Found {len(aha_files)} AHA files (loading only AHA)")
    elif load_only == 'WEEK':
        print(f"Found {len(week_files)} WEEK files (loading only WEEK)")
    else:
        print(f"Found {len(aha_files)} AHA files and {len(week_files)} WEEK files")
    print(f"Using {n_workers} workers for loading")
    
    if not aha_files and not week_files:
        raise FileNotFoundError(f"No subject files found in {data_dir}")
    
    # STEP 1: Caricamento AHA file
    aha_frames = [] 
    if aha_files:
        print(f"Loading {len(aha_files)} AHA files...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_file = {executor.submit(_load_aha_file, file_path, subject_labels, subject_aha_scores): file_path 
                            for file_path in aha_files}
            
            for future in as_completed(future_to_file): 
                result = future.result()
                if result is not None:
                    aha_frames.append(result)
    
    # Combina tutti i file AHA in un singolo DataFrame
    aha_data = pd.concat(aha_frames, ignore_index=True) if aha_frames else pd.DataFrame()
    print(f"Successfully loaded {len(aha_frames)}/{len(aha_files)} AHA files")
    
    # Applica preprocessing BASE a tutti i file AHA
    if not aha_data.empty:
        aha_data = _apply_common_preprocessing(aha_data, is_week_data=False)
        
    
    # STEP 2: Load WEEK files IN BATCHES 
    
    week_data = pd.DataFrame()
    if week_files:
        print(f"Loading {len(week_files)} WEEK files...")
        
        # Configurazione per il batch processing
        batch_size = 5  
        week_workers = min(n_workers, batch_size, 4)  
        all_week_batches: List[pd.DataFrame] = []
        
        for i in range(0, len(week_files), batch_size): # processa i file in batch
            batch = week_files[i:i+batch_size] # prendi il batch corrente
            batch_num = i // batch_size + 1 # numero del batch corrente
            total_batches = (len(week_files) - 1) // batch_size + 1 # numero totale di batch
            print(f"\n Processing batch {batch_num}/{total_batches} ({len(batch)} files)...") 
            
            batch_frames = []
            with ProcessPoolExecutor(max_workers=week_workers) as executor:
                future_to_file = {executor.submit(_load_week_file, file_path): file_path 
                                for file_path in batch}
                
                for future in as_completed(future_to_file):
                    result = future.result()
                    if result is not None:
                        batch_frames.append(result)
            
            print(f" Loaded {len(batch_frames)}/{len(batch)} files in batch {batch_num}")
            
            if batch_frames:
                # Combina batch in un singolo DataFrame
                batch_df = pd.concat(batch_frames, ignore_index=True)
                print(f"      Loaded batch {batch_num}: {batch_df.shape[0]:,} rows")
                
                # Applica preprocessing BASE al batch IMMEDIATAMENTE (prima di caricare il batch successivo)
                print(f"      Applying BASE preprocessing to batch {batch_num}...")
                batch_df = _apply_common_preprocessing(batch_df, is_week_data=True)
                print(f"      Preprocessed batch {batch_num}: {batch_df.shape[0]:,} rows") 
                
                all_week_batches.append(batch_df) 
                
                del batch_frames, batch_df # libera memoria
                import gc
                gc.collect()  # Forza la garbage collection
                print(f" Batch {batch_num} complete. Memory freed.")
        
        # Concatena una volta alla fine per efficienza
        week_data = pd.concat(all_week_batches, ignore_index=True) if all_week_batches else pd.DataFrame() # concatena i batch

    return aha_data, week_data


def calculate_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    
    # Calcolo della magnitudine
    xD, yD, zD = df["x_D"].astype(float), df["y_D"].astype(float), df["z_D"].astype(float)
    xND, yND, zND = df["x_ND"].astype(float), df["y_ND"].astype(float), df["z_ND"].astype(float)
    df["mag_dom"] = np.sqrt(xD*xD + yD*yD + zD*zD)
    df["mag_non"] = np.sqrt(xND*xND + yND*yND + zND*zND)
    return df


def decimate_signals(df: pd.DataFrame, decimation_factor: int = 2, data_type: str = "UNKNOWN") -> pd.DataFrame:
    """Decima i segnali di un DataFrame (AHA o WEEK) usando vettorizzazione."""
    result_frames = []
    
    # Processa ogni soggetto separatamente
    for _, group in df.groupby("subject_id", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        
        if len(group) < decimation_factor:
            result_frames.append(group) 
            continue
        
        # Identifica le colonne dei segnali da decimare
        signal_cols = ["x_D", "y_D", "z_D", "x_ND", "y_ND", "z_ND"]
        available_signal_cols = [col for col in signal_cols if col in group.columns]
        
        if not available_signal_cols:
            result_frames.append(group)
            continue
        
        # VETTORIZZAZIONE: Converte tutte le colonne dei segnali in un array NumPy 2D
        # Shape: (n_samples, n_channels)
        signals_matrix = group[available_signal_cols].to_numpy(dtype=float)
        
        # Applica decimate una volta sola su tutti i canali contemporaneamente (axis=0)
        decimated_matrix = decimate(signals_matrix, decimation_factor, ftype='iir', zero_phase=True, axis=0)
        
        # Converti il risultato decimato in un DataFrame
        decimated_data = pd.DataFrame(decimated_matrix, columns=available_signal_cols)
        new_length = decimated_matrix.shape[0]

        # Colonne di metadata da preservare (di base)
        metadata_cols = ["subject_id", "session_type"]
        
        # Per i dati AHA, preserva anche le colonne per la classificazione
        if data_type == "AHA":
            additional_cols = ["label", "AHA"]  # label per classificazione, AHA per punteggio
            metadata_cols.extend(additional_cols)
        
        for col in metadata_cols:
            if col in group.columns:
                decimated_data[col] = group[col].iloc[0] # preserva il valore costante per il soggetto
        
        # Mantieni solo i timestamp decimati corrispondenti ai campioni
        decimated_data["timestamp"] = group["timestamp"].iloc[::decimation_factor][:new_length].values
        
        result_frames.append(decimated_data) # aggiungi il DataFrame decimato alla lista
    
    result = pd.concat(result_frames, ignore_index=True) # concatena tutti i DataFrame decimati
    return result

def window_segments(df: pd.DataFrame, window_size: int, overlap: float) -> pd.DataFrame:
    """Crea finestre scorrevoli dai dati."""
    step = int(round(window_size * (1.0 - float(overlap)))) # calcola il passo
    rows = []
    
    for sid, g in df.groupby("subject_id", sort=False): #lavoro su un soggetto alla volta
        g = g.reset_index(drop=True) # resetto gli indici per il soggetto
        n = len(g) # lunghezza del segnale per il soggetto
        s = 0 # indice di inizio finestra
        
        while s + window_size <= n: # finché la finestra non supera la lunghezza del segnale
            e = s + window_size # calcola l'indice di fine finestra
            seg = g.iloc[s:e] # estrai il segmento della finestra
            window_row = { 
                "subject_id": sid,
                "start_idx": s,
                "end_idx": e,
                "start_time": float(seg["timestamp"].iloc[0]),
                "end_time": float(seg["timestamp"].iloc[-1]),
            }
            
            if "label" in seg.columns:
                label_mode = seg["label"].mode() # trova la label più frequente nella finestra
                window_row["label"] = label_mode.iloc[0] if len(label_mode) > 0 else seg["label"].iloc[0] # assegna la label più frequente
            # AHA score 
            if "AHA" in seg.columns:
                window_row["AHA"] = seg["AHA"].iloc[0]  
            rows.append(window_row) # aggiungi la riga della finestra alla lista
            s += step if step > 0 else window_size # aggiorna l'indice di inizio per la prossima finestra

    result = pd.DataFrame(rows)
    return result


def save_preprocessed(df: pd.DataFrame, windows: pd.DataFrame, out_dir: Path) -> None: 

    out_dir.parent.mkdir(parents=True, exist_ok=True) 
    out_dir.mkdir(parents=True, exist_ok=True) 
    
    # Salva i dati in Parquet (più veloce di CSV)
    try:
        import pyarrow
        # Salva i dati in Parquet 
        df.to_parquet(out_dir / "signals.parquet", index=False, engine='pyarrow', compression='snappy')
        windows.to_parquet(out_dir / "windows.parquet", index=False, engine='pyarrow', compression='snappy')
    except ImportError:
        # Salva i dati in CSV se pyarrow non è disponibile
        df.to_csv(out_dir / "signals.csv", index=False)
        windows.to_csv(out_dir / "windows.csv", index=False)


def preprocess_aha_signals(data_dir: Path, output_dir: Path, window_size: int, overlap: float, 
                          sampling_rate: int, metadata_file: Path = None, n_workers: int = None,
                          decimation_factor: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:

    print("Preprocessing AHA signals (clinical data)...")
    
    aha_data, _ = load_csv_folder_multiprocess(data_dir, metadata_file, n_workers, load_only='AHA')
    
    if aha_data.empty:
        raise ValueError("No AHA data found in the dataset") 
    
    # Decimazione, magnitude, windowing
    effective_sampling_rate = sampling_rate // decimation_factor if decimation_factor else sampling_rate
    decimated = decimate_signals(aha_data, decimation_factor=decimation_factor, data_type="AHA")
    print(f"\n Magnitude calculation (at {effective_sampling_rate} Hz)...")
    processed = calculate_magnitude(decimated)
    print(f"\n Windowing...")
    windows = window_segments(processed, window_size, overlap)

    aha_output_dir = output_dir / "preprocessed" / "aha"
    save_preprocessed(processed, windows, aha_output_dir)
    
    return processed, windows


def preprocess_week_signals(data_dir: Path, output_dir: Path, window_size: int, overlap: float, 
                           sampling_rate: int, metadata_file: Path = None, n_workers: int = None,
                           decimation_factor: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    print("Preprocessing WEEK signals (ecological data)...")
    
    # Load WEEK data (only WEEK files)
    _, week_data = load_csv_folder_multiprocess(data_dir, metadata_file, n_workers, load_only='WEEK')
    
    if week_data.empty:
        raise ValueError("No WEEK data found in the dataset")
    
    # Decimazione, magnitude, windowing
    effective_sampling_rate = sampling_rate // decimation_factor
    decimated = decimate_signals(week_data, decimation_factor=decimation_factor, data_type="WEEK")

    print(f"\n Magnitude calculation (at {effective_sampling_rate} Hz)...")
    processed = calculate_magnitude(decimated)
    
    print(f"\n Windowing...")
    windows = window_segments(processed, window_size, overlap)
    
    week_output_dir = output_dir / "preprocessed" / "week"
    print(f"\n Saving WEEK data to disk...")
    save_preprocessed(processed, windows, week_output_dir)
    
    return processed, windows