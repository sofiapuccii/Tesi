"""
Analisi di Regressione Lineare per predire AHA Score da ST% (Daily AHA Biomarker).

Input: regression_dataset.csv con colonne [subject_id, st_percentage_week, true_aha_score]
Output: Analisi completa con K-fold Cross-Validation, metriche dettagliate e visualizzazioni.

Workflow:
1. Carica dataset CSV
2. K-fold Cross-Validation
3. Per ogni fold: calcola Pearson r, R², MAE
4. Report statistiche complete
5. Scatter plot Predicted vs Actual
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr

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


def perform_kfold_regression(X: np.ndarray, y: np.ndarray, k_folds: int = 5, 
                           random_state: int = 42) -> Tuple[List[float], List[float], List[float], 
                                                           List[np.ndarray], List[np.ndarray]]:
    """
    Esegue K-fold Cross-Validation per regressione lineare.
    
    Returns:
        correlations: Lista coefficienti di correlazione di Pearson per ogni fold
        r2_scores: Lista R² per ogni fold  
        mae_scores: Lista MAE per ogni fold
        all_predictions: Lista predizioni per ogni fold
        all_actuals: Lista valori reali per ogni fold
    """
    
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    
    correlations = []
    r2_scores = []
    mae_scores = []
    all_predictions = []
    all_actuals = []
    
    print(f"\n=== K-FOLD CROSS-VALIDATION (k={k_folds}) ===")
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
        # Divisione train/test per questo fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Training del modello di regressione lineare
        model = LinearRegression()
        model.fit(X_train.reshape(-1, 1), y_train)
        
        # Predizioni sul test set
        y_pred = model.predict(X_test.reshape(-1, 1))
        
        # Calcolo metriche
        correlation, p_value = pearsonr(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Salva risultati
        correlations.append(correlation)
        r2_scores.append(r2)
        mae_scores.append(mae)
        all_predictions.extend(y_pred)
        all_actuals.extend(y_test)
        
        print(f"Fold {fold_idx:2d}: r={correlation:+.4f} (p={p_value:.4f}), "
              f"R²={r2:+.4f}, MAE={mae:.4f}")
    
    return correlations, r2_scores, mae_scores, all_predictions, all_actuals


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
                       output_path: Path) -> None:
    """
    Crea scatter plot Predicted vs Actual con linea identità e regressione.
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(actuals, predictions, alpha=0.7, s=60, color='steelblue', 
               edgecolors='darkblue', linewidth=0.5, label='Data Points')
    
    # Calcola range per le linee
    min_val = min(min(actuals), min(predictions))
    max_val = max(max(actuals), max(predictions))
    range_vals = np.linspace(min_val, max_val, 100)
    
    # Linea identità y=x (perfetta predizione)
    ax.plot(range_vals, range_vals, '--', color='red', linewidth=2, 
            alpha=0.8, label='Perfect Prediction (y=x)')
    
    # Linea di regressione
    z = np.polyfit(actuals, predictions, 1)
    p = np.poly1d(z)
    ax.plot(range_vals, p(range_vals), '-', color='orange', linewidth=2, 
            alpha=0.8, label=f'Regression Line (y={z[0]:.3f}x+{z[1]:+.3f})')
    
    # Calcola metriche globali
    global_r, global_p = pearsonr(actuals, predictions)
    global_r2 = r2_score(actuals, predictions)
    global_mae = mean_absolute_error(actuals, predictions)
    
    # Etichette e titoli
    ax.set_xlabel('True AHA Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted AHA Score (DAB)', fontsize=12, fontweight='bold')
    ax.set_title('Daily AHA Biomarker (DAB) Prediction\nST% → AHA Score Linear Regression', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Statistiche nel grafico
    stats_text = f'Global Metrics:\nr = {global_r:+.4f} (p = {global_p:.4f})\nR² = {global_r2:+.4f}\nMAE = {global_mae:.4f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Personalizzazione
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    
    # Salva grafico
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nGrafico salvato: {output_path}")
    
    plt.show()


def run_regression_analysis(dataset_path: Path, output_dir: Path, k_folds: int = 5) -> None:
    """Esegue l'analisi completa di regressione lineare."""
    
    print("=== DAILY AHA BIOMARKER (DAB) REGRESSION ANALYSIS ===")
    
    # 1. Caricamento dataset
    data = load_regression_dataset(dataset_path)
    
    # 2. Preparazione dati
    X = data['st_percentage_week'].values  # ST% (Features)
    y = data['true_aha_score'].values      # AHA Score (Target)
    
    print(f"\nFeatures: ST% from WEEK data (Daily Activity Biomarker)")
    print(f"Target: True AHA Clinical Score")
    print(f"Samples: {len(X)}")
    
    # 3. K-fold Cross-Validation
    correlations, r2_scores, mae_scores, all_predictions, all_actuals = perform_kfold_regression(
        X, y, k_folds=k_folds
    )
    
    # 4. Statistiche dettagliate
    print_statistical_summary(correlations, r2_scores, mae_scores)
    
    # 5. Visualizzazione
    create_scatter_plot(all_predictions, all_actuals, output_dir / "regression_scatter_plot.png")
    
    # 6. Salva risultati numerici
    results = pd.DataFrame({
        'fold': range(1, k_folds + 1),
        'pearson_r': correlations,
        'r2_score': r2_scores,  
        'mae': mae_scores
    })
    
    results_path = output_dir / "regression_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    print(f"Risultati numerici salvati: {results_path}")
    
    print(f"\n✅ Analisi completata. Output in: {output_dir}")