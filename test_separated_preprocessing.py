import sys
import yaml
import argparse
from pathlib import Path
import time

# directory del progetto
project_dir = Path(__file__).parent.absolute() 
pipeline_dir = project_dir / "pipeline" 

# Aggiungi la directory pipeline al path
sys.path.insert(0, str(pipeline_dir))

# Verifica e importa il modulo preprocessing appropriato
def import_preprocessing_module(use_gravity_removal=False):
    """Importa il modulo di preprocessing appropriato."""
    if use_gravity_removal:
        preprocessing_file = pipeline_dir / "preprocessing_gravity.py"
        module_name = "preprocessing_gravity.py"
        if not preprocessing_file.exists():
            print(f" Errore: file 'preprocessing_gravity.py' non trovato in {pipeline_dir}")
            sys.exit(1)

        from preprocessing_gravity import preprocess_aha_signals, preprocess_week_signals
        return preprocess_aha_signals, preprocess_week_signals, "gravity"
    else:
        preprocessing_file = pipeline_dir / "preprocessing.py"
        module_name = "preprocessing.py"
        if not preprocessing_file.exists():
            print(f" Errore: file 'preprocessing.py' non trovato in {pipeline_dir}")
            sys.exit(1)

        from preprocessing import preprocess_aha_signals, preprocess_week_signals
        return preprocess_aha_signals, preprocess_week_signals, "standard"


def load_config(config_path: str) -> dict:
    """Carica configurazione da file YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_separated_preprocessing(config_path: str, use_gravity_removal: bool = False):
    
    # Importa il modulo appropriato
    preprocess_aha_signals, preprocess_week_signals, module_type = import_preprocessing_module(use_gravity_removal)
    
    # Carica configurazione PRIMA di fare i print
    config = load_config(config_path)
    
    # Parametri per gravity removal (caricali subito)
    gravity_params = config.get('gravity_removal', {})
    cutoff_hz = gravity_params.get('cutoff_hz', 0.3)
    filter_order = gravity_params.get('filter_order', 2)
    filter_columns = gravity_params.get('columns', None)
    
    print(f"\n TEST PREPROCESSING SEPARATO - {module_type.upper()}")
    print(f" Config: {config_path}")
    if use_gravity_removal:
        print(f"  Pipeline: Load → Decimate → Unix → Gravity Removal (f={cutoff_hz}Hz) → Magnitude → Window")
    else:
        print(f"  Pipeline: Load → Decimate → Unix → Magnitude → Window")
    
    try:
        # La configurazione è già stata caricata sopra
        
        # Verifica che siamo nella directory corretta per i percorsi relativi
        current_dir = Path.cwd()
        print(f" Directory di lavoro: {current_dir}")
        
        # Estrai parametri
        data_dir = Path(config['data_dir'])
        base_output_dir = Path(config['output_dir'])
        
        # Crea cartelle separate per standard vs gravity preprocessing
        if use_gravity_removal:
            output_dir = base_output_dir / "gravity_preprocessing"
        else:
            output_dir = base_output_dir / "standard_preprocessing"
        print(f" Output: {output_dir}")
        
        window_size = config['window_size']
        overlap = config['overlap']
        sampling_rate = config['sampling_rate']
        n_workers = config.get('n_workers', None)
        metadata_file = Path(config.get('metadata_file', None)) if config.get('metadata_file') else None
        enable_cross_dataset = config.get('enable_cross_dataset_analysis', False)
        decimation_factor = config.get('decimation_factor', 2)  # Default 2 for WEEK (50 Hz effective, Nyquist 25 Hz)
        
        # Verifica che i dati esistano
        if not data_dir.exists():
            print(f" Directory dati non trovata: {data_dir}")
            return False
        
        # Verifica sottocartelle AHA e WEEK
        aha_dir = data_dir / "AHA"
        week_dir = data_dir / "week"
        
        if not aha_dir.exists():
            print(f" Directory AHA non trovata: {aha_dir}")
            return False
            
        if not week_dir.exists():
            print(f" Directory WEEK non trovata: {week_dir}")
            return False
        
        print(f" Directory AHA trovata: {len(list(aha_dir.glob('*.csv')))} file CSV")
        print(f" Directory WEEK trovata: {len(list(week_dir.glob('*.csv')))} file CSV")
        
        # Crea directory output
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n Avvio preprocessing separato...")
        start_time = time.time()
        
        # Preprocessing AHA
        print(f"\n Processing AHA (dati clinici)...")
        
        if use_gravity_removal:
            aha_signals, aha_windows = preprocess_aha_signals(
                data_dir=data_dir,
                output_dir=output_dir,
                window_size=window_size,
                overlap=overlap,
                sampling_rate=sampling_rate,
                metadata_file=metadata_file,
                n_workers=n_workers,
                decimation_factor=decimation_factor,
                gravity_cutoff_hz=cutoff_hz,
                gravity_filter_order=filter_order,
                gravity_columns=filter_columns
            )
        else:
            aha_signals, aha_windows = preprocess_aha_signals(
                data_dir=data_dir,
                output_dir=output_dir,
                window_size=window_size,
                overlap=overlap,
                sampling_rate=sampling_rate,
                metadata_file=metadata_file,
                n_workers=n_workers,
                decimation_factor=decimation_factor
            )
        
        # Preprocessing WEEK
        print(f"\n Processing WEEK...")
        
        if use_gravity_removal:
            week_signals, week_windows = preprocess_week_signals(
                data_dir=data_dir,
                output_dir=output_dir,
                window_size=window_size,
                overlap=overlap,
                sampling_rate=sampling_rate,
                metadata_file=metadata_file,
                n_workers=n_workers,
                decimation_factor=decimation_factor,
                gravity_cutoff_hz=cutoff_hz,
                gravity_filter_order=filter_order,
                gravity_columns=filter_columns
            )
        else:
            week_signals, week_windows = preprocess_week_signals(
                data_dir=data_dir,
                output_dir=output_dir,
                window_size=window_size,
                overlap=overlap,
                sampling_rate=sampling_rate,
                metadata_file=metadata_file,
                n_workers=n_workers,
                decimation_factor=decimation_factor
            )
        
        end_time = time.time()
        
        print(f"\n PREPROCESSING COMPLETATO in {end_time - start_time:.1f}s")
        print(f" AHA: {len(aha_signals):,} segnali, {len(aha_windows):,} finestre, {aha_signals['subject_id'].nunique()} soggetti")
        print(f" WEEK: {len(week_signals):,} segnali, {len(week_windows):,} finestre, {week_signals['subject_id'].nunique()} soggetti")
        
        # Verifica veloce struttura dati
        aha_has_labels = 'label' in aha_signals.columns
        week_has_labels = 'label' in week_signals.columns
        if not aha_has_labels:
            print(f"  AHA data missing 'label' column!")
            return False
        if week_has_labels:
            print(f"  WEEK data should not have 'label' column!")
            return False
        
        # Verifica file generati
        outputs = [
            (output_dir / "preprocessed" / "aha" / "signals.parquet", output_dir / "preprocessed" / "aha" / "signals.csv"),
            (output_dir / "preprocessed" / "aha" / "windows.parquet", output_dir / "preprocessed" / "aha" / "windows.csv"),
            (output_dir / "preprocessed" / "week" / "signals.parquet", output_dir / "preprocessed" / "week" / "signals.csv"),
            (output_dir / "preprocessed" / "week" / "windows.parquet", output_dir / "preprocessed" / "week" / "windows.csv"),
        ]

        all_files_exist = True
        for parquet_path, csv_path in outputs:
            if not (parquet_path.exists() or csv_path.exists()):
                all_files_exist = False
                break
        
        if not all_files_exist:
            print(f" Alcuni file output mancanti")
            return False
            
        # Verifica soggetti processati
        aha_subjects = set(aha_signals['subject_id'])
        week_subjects = set(week_signals['subject_id'])
        overlap = aha_subjects.intersection(week_subjects)
        
        print(f"  AHA subjects: {len(aha_subjects)}, WEEK subjects: {len(week_subjects)}")
        print(f"  Subject overlap: {len(overlap)} soggetti (normale se stessi soggetti in entrambi i dataset)")
        
    except Exception as e:
        print(f" ERRORE: {type(e).__name__}: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test preprocessing separato AHA e WEEK')
    parser.add_argument('--config', default='Config/config_separated_test.yaml',
                       help='File di configurazione YAML (default: Config/config_separated_test.yaml)')
    parser.add_argument('--gravity', action='store_true',
                       help='Usa preprocessing_gravity.py invece di preprocessing.py')
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f" File di configurazione non trovato: {config_path}")
        print(f"   Crea il file o specifica un path diverso con --config")
        return False
    
    success = test_separated_preprocessing(str(config_path), use_gravity_removal=args.gravity)
    
    if success:
        preprocessing_type = "gravity removal" if args.gravity else "standard"
        print(f" Test completato con successo!")
        return True
    else:
        print(f"\n TEST FALLITO")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
