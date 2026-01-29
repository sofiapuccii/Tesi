from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
import joblib

# Stile dei grafici
plt.style.use('default')
sns.set_palette("husl")


def load_regression_dataset(dataset_path: Path) -> pd.DataFrame:
    """Carica il dataset di regressione."""
    if not dataset_path.exists(): 
        raise FileNotFoundError(f"Dataset non trovato: {dataset_path}")
    
    data = pd.read_csv(dataset_path) 
    required_columns = ["subject_id", "st_percentage_week", "true_aha_score"]
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Colonne mancanti nel dataset: {missing_columns}")
    
    # Rimuovi eventuali valori NaN
    data_clean = data.dropna(subset=["st_percentage_week", "true_aha_score"])
    
    print(f"Dataset caricato: {len(data_clean)} campioni")
    print(f"Range ST%: {data_clean['st_percentage_week'].min():.1f} - {data_clean['st_percentage_week'].max():.1f}")
    print(f"Range AHA Score: {data_clean['true_aha_score'].min():.1f} - {data_clean['true_aha_score'].max():.1f}")
    
    return data_clean


def perform_groupkfold_regression(X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
                                  subject_ids: np.ndarray, k_folds: int = 5) -> Tuple:
    
    from sklearn.model_selection import GroupKFold
    group_kfold = GroupKFold(n_splits=k_folds)

    correlations = []
    r2_scores = []
    mae_scores = []
    slopes = []
    intercepts = []
    all_predictions = []
    all_actuals = []
    all_subject_ids = [] 
    print(f"\n=== GROUP K-FOLD CROSS-VALIDATION (k={k_folds}) ===")

    for fold_idx, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Estrai subject_id del test set
        test_subject_ids = subject_ids[test_idx]
        
        # Addestra modello
        model = LinearRegression()
        model.fit(X_train.reshape(-1, 1), y_train)

        slopes.append(model.coef_[0])
        intercepts.append(model.intercept_)

        # Predizioni
        y_pred = model.predict(X_test.reshape(-1, 1))

        # Metriche
        correlation, p_value = pearsonr(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        correlations.append(correlation)
        r2_scores.append(r2)
        mae_scores.append(mae)
        
        # Salva predizioni + metadata
        all_predictions.extend(y_pred)
        all_actuals.extend(y_test)
        all_subject_ids.extend(test_subject_ids)  
        
        print(f"Fold {fold_idx:2d}: r={correlation:+.4f} (p={p_value:.4f}), "
              f"R²={r2:+.4f}, MAE={mae:.4f}, "
              f"Slope={model.coef_[0]:+.4f}, Intercept={model.intercept_:+.4f}")
    
    return (correlations, r2_scores, mae_scores, slopes, intercepts, 
            all_predictions, all_actuals, all_subject_ids)
    
  
def print_statistical_summary(correlations: List[float], r2_scores: List[float], 
                            mae_scores: List[float]) -> None:
    """Stampa il riepilogo statistico completo."""
    
    print(f"\n=== CORRELATION LIST (Pearson r) ===")
    for i, r in enumerate(correlations, 1):
        print(f"Fold {i:2d}: r = {r:+.6f}")
    
    print(f"\n=== R² SCORE LIST ===")
    for i, r2 in enumerate(r2_scores, 1):
        print(f"Fold {i:2d}: R² = {r2:+.6f}")
    
    print(f"\n=== MAE SCORE LIST ===")
    for i, mae in enumerate(mae_scores, 1):
        print(f"Fold {i:2d}: MAE = {mae:.6f}")
    
    # Statistiche aggregate
    print(f"\n=== STATISTICAL SUMMARY ===")
    print(f"Pearson Correlation:")
    print(f"  Mean ± Std: {np.mean(correlations):+.4f} ± {np.std(correlations):.4f}")
    print(f"  Range: [{min(correlations):+.4f}, {max(correlations):+.4f}]")
    
    print(f"R² Score:")
    print(f"  Mean ± Std: {np.mean(r2_scores):+.4f} ± {np.std(r2_scores):.4f}")
    print(f"  Range: [{min(r2_scores):+.4f}, {max(r2_scores):+.4f}]")
    
    print(f"Mean Absolute Error:")
    print(f"  Mean ± Std: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"  Range: [{min(mae_scores):.4f}, {max(mae_scores):.4f}]")


def create_scatter_plot(predictions: List[float], actuals: List[float], 
                       subject_ids: List[int], metadata_df: pd.DataFrame,
                       output_path: Path) -> None:
    """
    Crea scatter plot Predicted vs Actual con colorazione MACS.
    
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))

    # Palette MACS
    palette = {0: 'forestgreen', 1: 'gold', 2: 'orange', 3: 'red'}
    labels = {0: 'MACS 0', 1: 'MACS 1', 2: 'MACS 2', 3: 'MACS 3'}

    
    macs_values = []
    for sid in subject_ids:
        # Cerca MACS per questo subject_id
        match = metadata_df[metadata_df['subject'] == sid]
        if not match.empty and 'MACS' in match.columns:
            macs_values.append(match.iloc[0]['MACS'])
        else:
            macs_values.append(np.nan)
    
    macs_array = np.array(macs_values)

    # Plot con colorazione MACS
    if not all(pd.isnull(macs_array)):
        for macs_level in sorted(np.unique(macs_array[~pd.isnull(macs_array)])):
            idx = macs_array == macs_level
            ax.scatter(np.array(predictions)[idx], np.array(actuals)[idx],
                       alpha=0.8, s=60, color=palette.get(macs_level, 'gray'),
                       edgecolors='k', linewidth=0.5, 
                       label=labels.get(macs_level, f'MACS {macs_level}'))
    else:
        # Fallback: scatter unico se MACS non disponibile
        ax.scatter(predictions, actuals, alpha=0.7, s=60, color='steelblue', 
                   edgecolors='darkblue', linewidth=0.5, label='Data Points')

    # Etichette
    ax.set_xlabel('DAB', fontsize=12, fontweight='bold')
    ax.set_ylabel('AHA', fontsize=12, fontweight='bold')
    # Limita gli assi come nel grafico allegato
    ax.set_xlim(35, 102)  # Aggiungi margine superiore anche all'asse X
    ax.set_ylim(0, 102)  # Aggiungi margine superiore per non tagliare i punti a 100


    # Griglia
    ax.grid(True, alpha=0.3)

    # Legenda in alto a sinistra
    handles, labels_ = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', 
             fontsize=11, title='MACS Level')

    # Salva grafico
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n Grafico salvato: {output_path}")
    plt.close()


def run_regression_analysis(dataset_path: Path, output_dir: Path, k_folds: int = 5) -> None:
    """Esegue l'analisi completa di regressione lineare."""
    
    print("=== DAILY AHA BIOMARKER (DAB) REGRESSION ANALYSIS ===")
    
    # 1. Caricamento dataset
    data = load_regression_dataset(dataset_path)
    
    # 2. Carica metadata MACS (se disponibile)
    metadata_path = Path("../dati_uniti/metadata2023_08.xlsx")
    metadata_df = None
    
    if metadata_path.exists():
        try:
            metadata_df = pd.read_excel(metadata_path, engine="openpyxl")
            if 'subject' not in metadata_df.columns or 'MACS' not in metadata_df.columns:
                print(f" Metadata incompleto (manca 'subject' o 'MACS')")
                metadata_df = None
            else:
                print(f" File metadata trovato: {metadata_path}")
        except Exception as e:
            print(f" Errore caricamento metadata: {e}")
            metadata_df = None
    else:
        print(f" File metadata non trovato: {metadata_path}")
        print(f"   Il plot non avrà colorazione MACS.")
    
    # 3. Preparazione dati
    X = data['st_percentage_week'].values  # ST% (Features)
    y = data['true_aha_score'].values      # AHA Score (Target)
    groups = data['subject_id'].values     # Gruppi per GroupKFold
    subject_ids = data['subject_id'].values  
    
    print(f"\nFeatures: ST% from WEEK data (Daily Activity Biomarker)")
    print(f"Target: True AHA Clinical Score")
    print(f"Samples: {len(X)}")
    
    # 4. K-fold Cross-Validation
    (correlations, r2_scores, mae_scores, slopes, intercepts, 
     all_predictions, all_actuals, all_subject_ids) = perform_groupkfold_regression(
        X, y, groups, subject_ids, k_folds=k_folds
    )
    
    # 5. Statistiche dettagliate
    print_statistical_summary(correlations, r2_scores, mae_scores)
    
    # 6. Visualizzazione
    if metadata_df is not None:
        create_scatter_plot(all_predictions, all_actuals, all_subject_ids, 
                          metadata_df, output_dir / "regression_scatter_plot.png")
    else:
        # Fallback senza metadata
        create_scatter_plot(all_predictions, all_actuals, all_subject_ids, 
                          pd.DataFrame(), output_dir / "regression_scatter_plot.png")
    
    # 7. Salva risultati numerici
    results = pd.DataFrame({
        'fold': range(1, k_folds + 1),
        'pearson_r': correlations,
        'r2_score': r2_scores,  
        'mae': mae_scores,
        'slope': slopes,
        'intercept': intercepts
    })
    
    results_path = output_dir / "regression_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    print(f" Risultati numerici salvati: {results_path}")
    
    avg_slope = np.mean(slopes)
    avg_intercept = np.mean(intercepts)
    
    print("\n" + "="*60)
    print(" >>> PARAMETRI DA COPIARE PER IL GRAFICO DAB <<< ")
    print(" (Media dei coefficienti della Cross-Validation) ")
    print("="*60)
    print(f"SLOPE (m)     : {avg_slope:.5f}")
    print(f"INTERCEPT (q) : {avg_intercept:.5f}")
    print("="*60 + "\n")
    
    # 8. Salva predizioni individuali con subject_id
    predictions_df = pd.DataFrame({
        'subject_id': all_subject_ids,  
        'true_aha_score': all_actuals,
        'predicted_aha_score': all_predictions
    })
    
    predictions_path = output_dir / "individual_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f" Predizioni individuali salvate: {predictions_path}")
    
    # Salva il modello addestrato sull'intero dataset per uso futuro
    final_regressor = LinearRegression()
    final_regressor.fit(X.reshape(-1, 1), y)
    regressor_path = output_dir / "regressor_dab.pkl"
    joblib.dump(final_regressor, regressor_path)
    print(f"\n Regressore salvato in: {regressor_path}\n")
    
    print(f"\n Analisi completata. Output in: {output_dir}")