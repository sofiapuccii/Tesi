#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Aggiungi pipeline al path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

def plot_gravity_comparison(results_dir: Path, subject_id: str, 
                          max_samples: int = 3000) -> None:
    """Plot magnitudine dominante con/senza rimozione gravità per soggetto specifico."""
    
    # Percorsi dei file preprocessati
    no_grav_path = results_dir / "preprocessed" / "standard" / "aha" / "signals.parquet"
    with_grav_path = results_dir / "preprocessed" / "gravity_removed" / "aha" / "signals.parquet"
    
    # Fallback a CSV se parquet non disponibile
    if not no_grav_path.exists():
        no_grav_path = no_grav_path.with_suffix('.csv')
    if not with_grav_path.exists():
        with_grav_path = with_grav_path.with_suffix('.csv')
    
    # Carica i dati
    print(f"Loading data for {subject_id}...")
    if no_grav_path.suffix == '.parquet':
        data_no_grav = pd.read_parquet(no_grav_path)
        data_with_grav = pd.read_parquet(with_grav_path)
    else:
        data_no_grav = pd.read_csv(no_grav_path)
        data_with_grav = pd.read_csv(with_grav_path)
    
    # Filtra per soggetto specifico e limita campioni
    subject_no_grav = data_no_grav[data_no_grav['subject_id'] == subject_id].head(max_samples)
    subject_with_grav = data_with_grav[data_with_grav['subject_id'] == subject_id].head(max_samples)
    
    if subject_no_grav.empty or subject_with_grav.empty:
        print(f"Errore: Soggetto {subject_id} non trovato nei dati!")
        return
    
    # Indici temporali semplici (campioni)
    samples_no_grav = range(len(subject_no_grav))
    samples_with_grav = range(len(subject_with_grav))
    
    # Creazione del plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot 1: Confronto diretto sovrapposto
    ax1.plot(samples_no_grav, subject_no_grav['mag_dom'], 
             color='#FF6B6B', alpha=0.8, linewidth=1, label='Con Gravità')
    ax1.plot(samples_with_grav, subject_with_grav['mag_dom'], 
             color='#4ECDC4', alpha=0.8, linewidth=1, label='Senza Gravità')
    ax1.set_ylabel('Magnitudine', fontsize=11)
    ax1.set_title(f'Confronto Magnitudine Dominante - {subject_id}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Solo segnale originale (con gravità)
    ax2.plot(samples_no_grav, subject_no_grav['mag_dom'], 
             color='#FF6B6B', linewidth=1.2)
    ax2.set_ylabel('Magnitudine (g)', fontsize=11)
    ax2.set_title('Segnale Originale (Con Componente di Gravità)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Solo segnale filtrato (senza gravità)
    ax3.plot(samples_with_grav, subject_with_grav['mag_dom'], 
             color='#4ECDC4', linewidth=1.2)
    ax3.set_ylabel('Magnitudine (g)', fontsize=11)
    ax3.set_xlabel('Campioni', fontsize=11)
    ax3.set_title('Segnale Filtrato (Rimossa Componente di Gravità)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salva il plot
    output_path = results_dir.parent / "Plots" / "output" / f"gravity_comparison_{subject_id}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot salvato: {output_path}")
    
    plt.show()


def main():
    
    # PARAMETRI DA PERSONALIZZARE
    subject_id = "subject_1_AHA"    # Soggetto da visualizzare
    max_samples = 3000              # Numero massimo di campioni da mostrare
    
    # Percorsi
    base_dir = Path(__file__).parent.parent  # Torna alla directory TESI2
    results_dir = base_dir / "results" / "preprocessing"
    
    print(f"=== GRAVITY COMPARISON PLOT ===")
    print(f"Subject: {subject_id}")
    print(f"Max samples: {max_samples}")
    print(f"Results dir: {results_dir}")
    
    # Controlla che esistano i dati preprocessati
    if not results_dir.exists():
        print(f"Errore: Directory {results_dir} non trovata!")
        print("Esegui prima i preprocessing con e senza gravity removal.")
        return
    
    try:
        plot_gravity_comparison(results_dir, subject_id, max_samples)
        
    except Exception as e:
        print(f"Errore durante la creazione del plot: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()