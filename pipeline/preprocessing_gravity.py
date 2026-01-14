from __future__ import annotations
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.signal import decimate, butter, filtfilt

# funzione per caricare il metadata
def load_subject_metadata(metadata_file: Path) -> Tuple[Dict[int, int], Dict[int, float]]:
    """Load subject metadata with labels and AHA scores."""
    if not metadata_file.exists():
        print(f"Warning: Metadata file not found: {metadata_file}")
        return {}, {}
    
    try:
        df_meta = pd.read_excel(metadata_file)
        subject_labels = {}
        subject_aha_scores = {}
        
        for _, row in df_meta.iterrows():
            subject_id = int(row['subject'])
            hemi = int(row['hemi'])
            
            # Convert hemi to ML label (1->0=controllo, 2->1=emiplegico)
            if hemi == 1:
                subject_labels[subject_id] = 0  # controllo sano
            elif hemi == 2:
                subject_labels[subject_id] = 1  # emiplegico
            else:
                print(f"Warning: Unknown hemi value {hemi} for subject {subject_id}")
                continue

            #carica AHA score
            subject_aha_scores[subject_id] = float(row['AHA'])
        
        return subject_labels, subject_aha_scores
        
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
        df["AHA"]= float(subject_aha_scores[subject_num])
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
    if metadata_file is not None: # se il metadata è disponibile
        subject_labels, subject_aha_scores = load_subject_metadata(metadata_file) # carichiamo il metadata
    
    # troviamo i file separatamente
    aha_files = sorted(data_dir.glob("AHA/*_AHA_RAW.csv")) if load_only != 'WEEK' else [] 
    week_files = sorted(data_dir.glob("week/*_week_RAW.csv")) if load_only != 'AHA' else []
   
    # STEP 1: Caricamento AHA file
    aha_frames = [] 
    if aha_files:
        print(f"Loading {len(aha_files)} AHA files...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_file = {executor.submit(_load_aha_file, file_path, subject_labels, subject_aha_scores): file_path 
                            for file_path in aha_files}
            
            for future in as_completed(future_to_file): # 
                result = future.result()
                if result is not None:
                    aha_frames.append(result)
    
    # Combina tutti i file AHA in un singolo DataFrame
    aha_data = pd.concat(aha_frames, ignore_index=True) if aha_frames else pd.DataFrame()
    
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
        
        for i in range(0, len(week_files), batch_size):
            batch = week_files[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(week_files) - 1) // batch_size + 1
            print(f"\n Processing batch {batch_num}/{total_batches} ({len(batch)} files)...")
            
            # Load batch of WEEK files
            batch_frames = []
            with ProcessPoolExecutor(max_workers=week_workers) as executor:
                future_to_file = {executor.submit(_load_week_file, file_path): file_path 
                                for file_path in batch}
                
                for future in as_completed(future_to_file):
                    result = future.result()
                    if result is not None:
                        batch_frames.append(result)
            
            if batch_frames:
                # Combina batch in un singolo DataFrame
                batch_df = pd.concat(batch_frames, ignore_index=True)
                
                # Applica preprocessing BASE al batch
                batch_df = _apply_common_preprocessing(batch_df, is_week_data=True)
                all_week_batches.append(batch_df)
                
                del batch_frames, batch_df # libera memoria
                import gc
                gc.collect()  # Forza la garbage collection
        
        # Concatena una volta alla fine per efficienza
        week_data = pd.concat(all_week_batches, ignore_index=True) if all_week_batches else pd.DataFrame() # concatena i batch
        print(f"\n Successfully loaded and processed {len(week_files)} WEEK files")

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
    """Decima i segnali di un DataFrame (AHA o WEEK)."""
    result_frames = []
    
    # Processa ogni soggetto separatamente
    for _, group in df.groupby("subject_id", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        
        if len(group) < decimation_factor:
            result_frames.append(group) 
            continue
        
        # Identifica le colonne dei segnali da decimare
        signal_cols = ["x_D", "y_D", "z_D", "x_ND", "y_ND", "z_ND"]
        if "mag_dom" in group.columns:
            signal_cols.extend(["mag_dom", "mag_non"])
        
        decimated_data = {}
        for col in signal_cols:
            if col in group.columns:
                signal = group[col].to_numpy(dtype=float)
                decimated_data[col] = decimate(signal, decimation_factor, ftype='iir', zero_phase=True)
        
        # Crea un nuovo DataFrame con i segnali decimati
        new_length = len(list(decimated_data.values())[0]) if decimated_data else len(group) // decimation_factor
        decimated_df = pd.DataFrame(decimated_data)

        # Colonne di metadata da preservare (di base)
        metadata_cols = ["subject_id", "session_type"]
        
        # Per i dati AHA, preserva anche le colonne per la classificazione
        if data_type == "AHA":
            additional_cols = ["label", "AHA"]  # label per classificazione, AHA per punteggio
            metadata_cols.extend(additional_cols)
        
        for col in metadata_cols:
            if col in group.columns:
                decimated_df[col] = group[col].iloc[0] # preserva il valore costante per il soggetto
        
        # Mantieni solo i timestamp decimati corrispondenti ai campioni
        decimated_df["timestamp"] = group["timestamp"].iloc[::decimation_factor][:new_length].values 
        
        result_frames.append(decimated_df)
    
    result = pd.concat(result_frames, ignore_index=True)
    return result

def _highpass_filter(signal: np.ndarray, cutoff_hz: float, fs: float, order: int = 2) -> np.ndarray:
    """Applica un filtro passa-alto al segnale."""
    nyquist= 0.5 * fs
    norm_cutoff = cutoff_hz / nyquist
    b, a = butter(order, norm_cutoff, btype='high', analog=False)
    return filtfilt(b, a, signal)

def remove_gravity(df: pd.DataFrame, sampling_rate: float, cutoff_hz: float = 0.3, 
                  filter_order: int = 2, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Rimuove la componente di gravità dai segnali utilizzando un filtro passa-alto."""
    df = df.copy()
    
    if columns is None:
        columns = ["x_D", "y_D", "z_D", "x_ND", "y_ND", "z_ND"]
    
    for col in columns:
        if col in df.columns:
            filtered_signals = []
            for _, group in df.groupby("subject_id", sort=False):
                signal = group[col].to_numpy(dtype=float)
                filtered_signal = _highpass_filter(signal, cutoff_hz, sampling_rate, filter_order)
                filtered_signals.append(pd.Series(filtered_signal, index=group.index))
            df[col] = pd.concat(filtered_signals).sort_index()
    
    print(f"Gravity removal complete.")
    return df

def window_segments(df: pd.DataFrame, window_size: int, overlap: float) -> pd.DataFrame:
    """Crea finestre scorrevoli dai dati."""
    step = int(round(window_size * (1.0 - float(overlap)))) # calcola il passo
    rows = []
    
    for sid, g in df.groupby("subject_id", sort=False): #lavoro su un soggetto alla volta
        g = g.reset_index(drop=True)
        n = len(g)
        s = 0
        
        while s + window_size <= n: # finché la finestra non supera la lunghezza del segnale
            e = s + window_size # calcola l'indice di fine finestra
            seg = g.iloc[s:e]
            window_row = { 
                "subject_id": sid,
                "start_idx": s,
                "end_idx": e,
                "start_time": float(seg["timestamp"].iloc[0]),
                "end_time": float(seg["timestamp"].iloc[-1]),
            }
            
            if "label" in seg.columns:
                label_mode = seg["label"].mode() 
                window_row["label"] = label_mode.iloc[0] if len(label_mode) > 0 else seg["label"].iloc[0] # assegna la label più frequente
            # Add AHA score if present (copied per subject)
            if "AHA" in seg.columns:
                window_row["AHA"] = seg["AHA"].iloc[0]  # AHA score is constant per subject
            rows.append(window_row)
            s += step if step > 0 else window_size

    result = pd.DataFrame(rows)
    return result


def save_preprocessed(df: pd.DataFrame, windows: pd.DataFrame, out_dir: Path) -> None: 

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Salva i dati in Parquet (più veloce di CSV)
    try:
        import pyarrow
        # Salva i dati in Parquet (persistent storage)
        df.to_parquet(out_dir / "signals.parquet", index=False, engine='pyarrow', compression='snappy')
        windows.to_parquet(out_dir / "windows.parquet", index=False, engine='pyarrow', compression='snappy')
    except ImportError:
        # Salva i dati in CSV se pyarrow non è disponibile
        df.to_csv(out_dir / "signals.csv", index=False)
        windows.to_csv(out_dir / "windows.csv", index=False)


def preprocess_aha_signals(data_dir: Path, output_dir: Path, window_size: int, overlap: float, 
                          sampling_rate: int, metadata_file: Path = None, n_workers: int = None,
                          decimation_factor: int = None, gravity_cutoff_hz: float = 0.3, 
                          gravity_filter_order: int = 2, gravity_columns: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:

    print("Preprocessing AHA signals (clinical data)...")
    
    aha_data, _ = load_csv_folder_multiprocess(data_dir, metadata_file, n_workers, load_only='AHA')
    
    if aha_data.empty:
        raise ValueError("No AHA data found in the dataset")
    
    # Pipeline: Decimation → Gravity Removal → Magnitude → Windowing
    effective_sampling_rate = sampling_rate // decimation_factor if decimation_factor else sampling_rate
    decimated = decimate_signals(aha_data, decimation_factor=decimation_factor, data_type="AHA")
    print(f"\n Gravity removal (high-pass, {effective_sampling_rate} Hz)...")
    decimated = remove_gravity(decimated, sampling_rate=effective_sampling_rate, 
                             cutoff_hz=gravity_cutoff_hz, filter_order=gravity_filter_order, columns=gravity_columns)
    print(f"\n Magnitude calculation (at {effective_sampling_rate} Hz)...")
    processed = calculate_magnitude(decimated)
    print(f"\n Windowing...")
    windows = window_segments(processed, window_size, overlap)

    aha_output_dir = output_dir / "preprocessed" / "aha"
    print(f"\n Saving AHA data to disk...")
    save_preprocessed(processed, windows, aha_output_dir)
    
    return processed, windows


def preprocess_week_signals(data_dir: Path, output_dir: Path, window_size: int, overlap: float, 
                           sampling_rate: int, metadata_file: Path = None, n_workers: int = None,
                           decimation_factor: int = 2, gravity_cutoff_hz: float = 0.3,
                           gravity_filter_order: int = 2, gravity_columns: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    print("Preprocessing WEEK signals (ecological data)...")
    
    # Load WEEK data separately (only WEEK files)
    _, week_data = load_csv_folder_multiprocess(data_dir, metadata_file, n_workers, load_only='WEEK')
    
    if week_data.empty:
        raise ValueError("No WEEK data found in the dataset")
    
    print(f"Found {len(week_data)} WEEK samples from {week_data['subject_id'].nunique()} subjects")
    
    print(f"\n PREPROCESSING AVANZATO WEEK:")
    print(f"{'-'*50}")
    
    print(f"\n Decimation with anti-aliasing filter (factor: {decimation_factor})...")
    effective_sampling_rate = sampling_rate // decimation_factor
    decimated = decimate_signals(week_data, decimation_factor=decimation_factor, data_type="WEEK")
    print(f"\n Gravity removal (high-pass, {effective_sampling_rate} Hz)...")
    decimated = remove_gravity(decimated, sampling_rate=effective_sampling_rate,
                             cutoff_hz=gravity_cutoff_hz, filter_order=gravity_filter_order, columns=gravity_columns)

    print(f"\n Magnitude calculation (at {effective_sampling_rate} Hz)...")
    processed = calculate_magnitude(decimated)
    
    print(f"\n Windowing...")
    windows = window_segments(processed, window_size, overlap)
    
    week_output_dir = output_dir / "preprocessed" / "week"
    print(f"\n Saving WEEK data to disk...")
    save_preprocessed(processed, windows, week_output_dir)
    
    return processed, windows