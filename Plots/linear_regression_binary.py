"""
Analisi di Regressione Lineare per DAB (Daily AHA Biomarker).
VERSIONE AGGIORNATA: Calcola ST% da voti binari (Hard Voting) per coerenza col grafico.

Input: temporal_predictions.csv (o il file con le predizioni grezze per frame)
Output: Modello allenato (regressor_dab.pkl) e parametri Slope/Intercept.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
import joblib

# Configurazione
plt.style.use('default')
sns.set_palette("husl")

def create_binary_dataset_from_raw(raw_predictions_path: Path) -> pd.DataFrame:
    """
    Carica le predizioni grezze e calcola la ST% basata sui binari (0/1).
    Questo assicura che il regressore sia allenato sulla stessa logica del grafico.
    """
    print(f"Generazione dataset da: {raw_predictions_path}")
    df_raw = pd.read_csv(raw_predictions_path)
    
    # Assicuriamoci di avere colonne binarie (es. predictions o is_st)
    # Se i tuoi dati hanno 'predictions' con -1/1, convertiamo in 0/1
    if 'predictions' in df_raw.columns:
        # Se è 1 (sano) e -1 (patologico), convertiamo in 1 e 0
        # Adatta questa riga se i tuoi dati sono diversi!
        df_raw['is_st'] = (df_raw['predictions'] == 1).astype(int)
    elif 'is_st' not in df_raw.columns:
        raise ValueError("Colonna 'predictions' o 'is_st' non trovata nel CSV grezzo.")

    # Raggruppa per soggetto e calcola la % di 1
    # HARD VOTING: (Numero di 1 / Totale finestre) * 100
    dataset = df_raw.groupby('subject_id').agg({
        'is_st': lambda x: (x.sum() / len(x)) * 100,  # Calcolo Percentuale
        'true_aha_score': 'first' # Prende il valore AHA (è uguale per tutte le righe del soggetto)
    }).reset_index()
    
    # Rinomina per chiarezza
    dataset.rename(columns={'is_st': 'st_percentage_binary'}, inplace=True)
    
    print(f"Dataset Hard-Voting creato: {len(dataset)} soggetti.")
    print(dataset.head())
    return dataset

def run_regression_analysis(dataset_path: Path, output_dir: Path):
    """Esegue il training e salva il modello."""
    
    # 1. Carica e Prepara i Dati (Hard Voting)
    try:
        # Tenta di creare il dataset dai dati grezzi per massima coerenza
        data = create_binary_dataset_from_raw(dataset_path)
    except Exception as e:
        print(f"Impossibile creare dataset dai grezzi ({e}). Provo a caricare il CSV standard...")
        data = pd.read_csv(dataset_path)
        if 'st_percentage_binary' not in data.columns:
             # Fallback su st_percentage_week se proprio non c'è altro, ma avvisa
            print("ATTENZIONE: Uso 'st_percentage_week' (Soft Voting?). Il grafico potrebbe essere disallineato.")
            data['st_percentage_binary'] = data['st_percentage_week']

    # Pulizia
    data = data.dropna(subset=['st_percentage_binary', 'true_aha_score'])
    X = data[['st_percentage_binary']].values # Feature (Input)
    y = data['true_aha_score'].values         # Target (Output)

    # 2. Addestra il Modello (Su tutto il dataset per il grafico finale)
    regressor = LinearRegression()
    regressor.fit(X, y)
    
    # Parametri estratti
    slope = regressor.coef_[0]
    intercept = regressor.intercept_
    
    # 3. Predizioni e Metriche
    y_pred = regressor.predict(X)
    r2 = r2_score(y, y_pred)
    pearson_corr, _ = pearsonr(data['st_percentage_binary'], y)
    mae = mean_absolute_error(y, y_pred)
    
    print("\n" + "="*50)
    print(" RISULTATI REGRESSIONE (HARD VOTING) ")
    print("="*50)
    print(f"Dataset Size : {len(data)}")
    print(f"Pearson R    : {pearson_corr:.4f}")
    print(f"R2 Score     : {r2:.4f}")
    print(f"MAE          : {mae:.2f}")
    print("-" * 30)
    print(f"SLOPE (m)    : {slope:.5f}")
    print(f"INTERCEPT (q): {intercept:.5f}")
    print("="*50)

    # 4. Salva il Modello (per plot_dab_temporal.py)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "regressor_dab.pkl"
    joblib.dump(regressor, model_path)
    print(f"\nModello salvato in: {model_path}")
    
    # 5. Salva Grafico di Regressione (Scatter)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data['st_percentage_binary'], y=y, color='blue', alpha=0.6, s=100, label='Soggetti')
    
    # Linea di regressione
    x_range = np.linspace(0, 100, 100).reshape(-1, 1)
    y_range = regressor.predict(x_range)
    plt.plot(x_range, y_range, color='red', linewidth=2, label='Regression Line')
    
    plt.xlabel('ST% (Binary Hard-Voting)', fontweight='bold')
    plt.ylabel('True AHA Score', fontweight='bold')
    plt.title(f'Regression Analysis (R={pearson_corr:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "regression_scatter_hard_voting.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    # PERCORSI (Modifica se necessario)
    base_dir = Path("results")
    
    # Qui devi puntare al file che contiene le predizioni di TUTTI i soggetti
    # Es. "temporal_predictions.csv"
    input_predictions_csv = base_dir / "clinical_analysis_v2" / "temporal_predictions.csv"
    
    output_folder = base_dir / "regression_analysis"
    
    if input_predictions_csv.exists():
        run_regression_analysis(input_predictions_csv, output_folder)
    else:
        print(f"Errore: File non trovato: {input_predictions_csv}")