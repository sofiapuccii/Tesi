#!/usr/bin/env python3
import sys
import yaml
import argparse
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd

# Ottieni la directory del progetto
project_dir = Path(__file__).parent.absolute()
pipeline_dir = project_dir / "pipeline"

# Aggiungi la directory pipeline al path
sys.path.insert(0, str(pipeline_dir))

from deep_learning import (
    prepare_classification_dataset,
    random_search_training,
    export_best_models,
    save_training_metadata,
)

def load_config(config_path: str) -> dict:
    """Carica configurazione da file YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_preprocessed_data(data_dir: Path, data_type: str = "aha"):
   
    import pandas as pd
    preprocessed_dir = data_dir / "preprocessed" / data_type
    
    # Prova a caricare signals
    signals_parquet = preprocessed_dir / "signals.parquet"
    signals_csv = preprocessed_dir / "signals.csv"
    
    signals = None
    if signals_parquet.exists():
        signals = pd.read_parquet(signals_parquet)
    elif signals_csv.exists():
        signals = pd.read_csv(signals_csv)
    
    if signals is None:
        return None, None
    
    # Prova a caricare windows
    windows_parquet = preprocessed_dir / "windows.parquet"
    windows_csv = preprocessed_dir / "windows.csv"
    
    windows = None
    if windows_parquet.exists():
        windows = pd.read_parquet(windows_parquet)
    elif windows_csv.exists():
        windows = pd.read_csv(windows_csv)
    
    if windows is None:
        return None, None
    
    return signals, windows


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test deep learning pipeline con dati preprocessed')
    parser.add_argument('--config', default='Config/config_improved_deep_learning.yaml',
                       help='File di configurazione YAML (default: Config/config_improved_deep_learning.yaml)')
    
    args = parser.parse_args()
    config_path = Path(args.config)
    
    if not config_path.exists():
        print(f" File di configurazione non trovato: {config_path}")
        print(f"   Configurazioni disponibili:")
        for conf in Path("Config").glob("*.yaml"):
            print(f"     - {conf}")
        sys.exit(1)
    
    print(f" TEST DEEP LEARNING")
    print(f"{'='*70}")
    print(f" Config file: {config_path}")
    
    try:
        # Carica configurazione
        config = load_config(str(config_path))
        
        # Estrai parametri per deep learning
        output_dir = Path(config['output_dir'])
        random_seed = config.get('random_seed', 42)

        # Gestione tipo di preprocessing (standard o gravity)
        preprocessing_type = config.get('preprocessing_type', 'standard').lower()
        if preprocessing_type not in ('standard', 'gravity'):
            print(f" preprocessing_type non valido: {preprocessing_type}. Usa 'standard' o 'gravity'.")
            sys.exit(1)

        # Determina la directory dei dati in base al tipo di preprocessing
        if preprocessing_type == 'standard':
            data_dir = output_dir / 'standard_preprocessing'
            dl_output_dir = output_dir / 'deep_learning_results_standard'
        else:
            data_dir = output_dir / 'gravity_preprocessing'
            dl_output_dir = output_dir / 'deep_learning_results_gravity'

        # Configurazione deep learning
        classification_cfg = config.get('classification', {})
        validation_split = float(config.get('validation_split', 0.2))
        channels = classification_cfg.get('channels', ["x_D", "y_D", "z_D", "x_ND", "y_ND", "z_ND"])
        
        # Parametri training
        random_search_global = config.get('random_search', {})
        n_trials = random_search_global.get('n_trials', 20)
        n_jobs = random_search_global.get('n_jobs', None)
        epochs = config.get('epochs', 50)
        

        # Verifica che la directory dati esista
        if not data_dir.exists():
            print(f" Directory dati non trovata: {data_dir}")
            print(f"   Esegui prima il preprocessing per creare i dati preprocessed")
            return False
        
        # Carica dati preprocessed AHA
        signals, windows = load_preprocessed_data(data_dir, "aha")

        if signals is None or windows is None:
            print(f" Impossibile caricare dati AHA preprocessed da {data_dir}")
            print(f"   Esegui prima il preprocessing AHA")
            sys.exit(1)
        
        print(f" Dati caricati: {len(signals):,} segnali, {len(windows):,} finestre, {signals['subject_id'].nunique()} soggetti")
        
        # Verifica che ci siano le colonne richieste
        missing_channels = [ch for ch in channels if ch not in signals.columns]
        if missing_channels:
            print(f" Colonne mancanti in signals: {missing_channels}")
            sys.exit(1)
        
        if 'label' not in windows.columns:
            print(f" Colonna 'label' mancante in windows")
            sys.exit(1)
        

        # Crea directory per risultati deep learning (già distinta sopra)
        models_dir = dl_output_dir / "models"
        metrics_dir = dl_output_dir / "metrics"
        
        dl_output_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        # FASE 1: Preparazione dataset
        print(f"\n=== PREPARAZIONE DATASET ===")
        
        dataset = prepare_classification_dataset(
            signals=signals,
            windows=windows,
            channels=channels,
            validation_split=validation_split,
            random_state=random_seed,
        )
        
        print(f" Dataset preparato: {dataset.X_train.shape[0]} campioni training, {len(np.unique(dataset.groups_train))} soggetti")
        
        # FASE 2: Random search training
        print(f"\n=== TRAINING MODELLI ===")
        
        model_results = []
        model_types = ["cnn_1d", "lstm"]
        # Training dei modelli
        for model_type in model_types:
            print(f"\n Training {model_type.upper()}...")
            
            # Configurazione specifica per il modello
            model_cfg = classification_cfg.get(model_type, {})
            model_cfg.setdefault('random_state', random_seed)
            model_cfg.setdefault('epochs', epochs)
            model_cfg.setdefault('validation_split', validation_split)
            model_cfg.setdefault('channels', channels)
            model_cfg.setdefault('n_splits', config.get('n_splits', 5))
            model_cfg.setdefault('param_space', config.get('param_space', {}))
            random_search_cfg = model_cfg.setdefault('random_search', {})
            random_search_cfg.setdefault('n_trials', n_trials)
            if n_jobs is not None:
                random_search_cfg.setdefault('n_jobs', n_jobs)

            model_start = time.time()
            result = random_search_training(
                dataset=dataset,
                model_type=model_type,
                base_cfg=model_cfg
            )
            model_end = time.time()
            
            print(f" {model_type.upper()} completato in {model_end - model_start:.1f}s - F1: {result.get('mean_f1_score', 0):.4f}")
            model_results.append(result)
        
        # FASE 3: Export miglior modello  
        print(f"\n=== EXPORT MODELLI ===")
        
        best_result = max(model_results, key=lambda r: r.get("mean_f1_score", 0))
        print(f" Miglior modello: {best_result['model_type'].upper()} (F1: {best_result.get('mean_f1_score', 0):.4f})")
        
        # Salva metadati di training
        save_training_metadata(model_results, metrics_dir)
        
        # Rimuovi modelli non-best per risparmiare memoria
        for res in model_results:
            if res is not best_result:
                res.pop("model", None)
        
        # Export best model
        metadata = export_best_models(best_result, dataset, models_dir)
        
        end_time = time.time()
        
        # RIEPILOGO FINALE
        print(f"\n=== COMPLETATO ===")
        print(f" Tempo totale: {end_time - start_time:.1f}s ({(end_time - start_time)/60:.1f} min)")
        print(f" Risultati salvati in: {dl_output_dir}")
        
        # Verifica file salvati
        expected_files = [
            models_dir / "best_model_classification.keras", 
            models_dir / "best_model_metadata.json",
            metrics_dir / "classification_random_search.json",
        ]
        
        all_files_exist = all(f.exists() for f in expected_files)
        
        if all_files_exist:
            print(f"  Tutti i file generati correttamente!")
            sys.exit(0)
        else:
            print(f"\n  ALCUNI FILE MANCANO")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n ERRORE DURANTE IL TEST:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()