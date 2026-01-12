#!/usr/bin/env python3
"""
Test script per l'analisi di regressione lineare.
Esegue l'analisi completa DAB (Daily AHA Biomarker) prediction.
"""

from pathlib import Path
import sys
import os

# Aggiungi la directory pipeline al path
sys.path.insert(0, str(Path(__file__).parent / "pipeline"))

from linear_regression_analysis import run_regression_analysis


def main():
    
    base_dir = Path(__file__).parent
    dataset_path = base_dir / "results" / "clinical_analysis" / "regression_dataset.csv"
    output_dir = base_dir / "results" / "regression_analysis"
    
    print("=== TEST LINEAR REGRESSION ANALYSIS ===")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    
    # Verifica che esista il dataset
    if not dataset_path.exists():
        print(f"\n Dataset non trovato: {dataset_path}")
        print(" Esegui prima clinical_analysis.py per generare il dataset")
        return False
    
    try:
        # Esegui analisi con 5-fold CV
        success = run_regression_analysis(
            dataset_path=dataset_path,
            output_dir=output_dir,
            k_folds=5
        )
        
        if success:
            print(f"\n Test completato con successo!")
            print(f" Controlla i risultati in: {output_dir}")
            return True
        else:
            print(f"\n Test fallito")
            return False
            
    except Exception as e:
        print(f"\n ERRORE durante il test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)