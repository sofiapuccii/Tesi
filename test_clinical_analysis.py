#!/usr/bin/env python3
"""
Test script per clinical analysis con configurazione YAML.
Calcola ST% da dati WEEK usando modello di classificazione addestrato.
"""

from pathlib import Path
import sys
import os

# Aggiungi la directory pipeline al path
sys.path.insert(0, str(Path(__file__).parent / "pipeline"))

from clinical_analysis import run_clinical_analysis
import yaml


def main():
    """Esegue clinical analysis con configurazione da YAML."""
    
    # Percorsi
    base_dir = Path(__file__).parent
    config_path = base_dir / "Config" / "config_clinical_analysis.yaml"
    
    print("=== TEST CLINICAL ANALYSIS ===")
    print(f"Config: {config_path}")
    
    # Verifica che esista il config
    if not config_path.exists():
        print(f"\n File configurazione non trovato: {config_path}")
        return False
    
    # Carica configurazione
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Estrai parametri
    models_dir = Path(config.get('models_dir', 'results/deep_learning_results_gravity/models'))
    data_dir = Path(config.get('data_dir', 'results/gravity_preprocessing'))
    metadata_path = Path(config.get('metadata_path', '../dati_uniti/metadata2023_08.xlsx'))
    output_dir = Path(config.get('output_dir', 'results/clinical_analysis'))
    
    print(f"Models dir: {models_dir}")
    print(f"Data dir: {data_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Output dir: {output_dir}")
    
    # Verifica prerequisiti
    if not (models_dir / "best_model_classification.keras").exists():
        print(f"\n Modello non trovato: {models_dir / 'best_model_classification.keras'}")
        print(" Esegui prima deep learning training per generare il modello")
        return False
    
    if not metadata_path.exists():
        print(f"\n File metadata non trovato: {metadata_path}")
        return False
    
    week_data_dir = data_dir / "preprocessed" / "week"
    if not week_data_dir.exists():
        print(f"\n Dati WEEK non trovati: {week_data_dir}")
        print(" Esegui prima il preprocessing dei dati WEEK")
        return False
    
    try:
        # Esegui analisi clinica
        run_clinical_analysis(
            models_dir=models_dir,
            data_dir=data_dir,
            metadata_path=metadata_path,
            output_dir=output_dir
        )
        
        print(f"\n Clinical analysis completata")
        return True
        
    except Exception as e:
        print(f"\n ERRORE durante clinical analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
