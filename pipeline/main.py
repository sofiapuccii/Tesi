from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import argparse
import numpy as np
import pandas as pd

from .utils import load_config, make_paths, setup_logging, set_seed
from .preprocessing import preprocess_signals
from .features import extract_features_from_windows, FeatureParams
from .train import stratified_split, train_models, save_best_model
from .evaluate import evaluate_model, save_metrics
from .plotting import plot_confusion_matrix, plot_roc_curves, plot_feature_importance, generate_all_plots
from .cache import get_memory, clear_cache, get_cache_info, reset_memory
from .cache_manager import CacheManager, create_preprocessing_params, create_feature_params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path("config.yaml"))) # permette di passare il file di configurazione
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache before running")
    parser.add_argument("--cache-info", action="store_true", help="Show cache information and exit")
    args = parser.parse_args()

    # gestione comandi cache
    if args.cache_info: # mostra informazioni cache
        memory = get_memory() # ottiene memoria cache
        info = get_cache_info(memory)
        print("Cache Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        return
    
    if args.clear_cache: # pulisce cache
        memory = get_memory() # ottiene memoria cache
        clear_cache(memory) # pulisce cache
        print("Cache cleared successfully")
        return

    cfg: Dict[str, Any] = load_config(Path(args.config)) # carica configurazione
    paths = make_paths(cfg) # costruisce le directory data_dir, output_dir, preprocessed_dir, models_dir
    log = setup_logging(paths.output_dir) 
    random_seed = cfg.get("random_seed", 42)
    set_seed(random_seed) #imposta la riproducibilità dei risultati
    
    # Inizializzazione sistema cache
    memory = get_memory()
    cache_manager = CacheManager()
    
    # controlla se i parametri di preprocessing sono cambiati
    preprocessing_params = create_preprocessing_params(
        int(cfg["window_size"]), 
        float(cfg["overlap"]), 
        int(cfg["sampling_rate"]), 
        str(paths.data_dir)
    )
    
    cache_cleared = cache_manager.check_and_invalidate(preprocessing_params, force_clear=args.clear_cache) 
    if cache_cleared: # se cache è stata invalidata, logga il cambiamento
        log.info("Cache invalidated due to parameter changes")
    
    cache_info = get_cache_info(memory) # ottiene informazioni cache
    log.info("Cache system initialized: %s", cache_info) # logga informazioni cache

    # preprocessing
    log.info("Preprocessing with config: %s", cfg) # logga configurazione
    signals, windows = preprocess_signals(paths.data_dir, paths.output_dir, int(cfg["window_size"]), float(cfg["overlap"]), int(cfg["sampling_rate"])) # preprocessing

    # feature extraction
    feats = extract_features_from_windows(signals, windows, FeatureParams(sampling_rate=int(cfg["sampling_rate"])) ) # estrazione features
    feats.to_csv(paths.preprocessed_dir / "features.csv", index=False) # salva features

    # ottiene modalità cross-validation
    cv_mode = cfg.get("hyperparameter_search", {}).get("cv_mode", "groupkfold") # controlla modalità cross-validation impostata nel file di configurazione
    
    train_df, test_df = stratified_split(feats, test_size=0.2, random_seed=random_seed, cv_mode=cv_mode) # restituisce due dataframe: train e test
    X_train = train_df.drop(columns=["label", "subject_id", "start_time", "end_time"])\
        .replace([np.inf, -np.inf], np.nan).fillna(0.0)  #rimuove le colonne non necessarie e sostituisce tutti i valori infiniti con NaN
    y_train = train_df["label"].astype(int) # converte la colonna label in intero
    groups_train = train_df["subject_id"] # crea un gruppo per ogni soggetto
    X_test = test_df.drop(columns=["label", "subject_id", "start_time", "end_time"])\
        .replace([np.inf, -np.inf], np.nan).fillna(0.0) # fa la stessa cosa per il dataframe test
    y_test = test_df["label"].astype(int)

    best_model, info, exp_dir = train_models(X_train, y_train, groups_train, cfg, paths.output_dir) # training e validazione del modello, riceve l'oggetto del modello addestrato, un dizionario con le metriche e i dettagli dell'esperimento e la cartella dove tutto è salvato
    log.info("Best model: %s | CV=%.4f | experiment saved to: %s", info.get("model"), info.get("cv_score"), exp_dir) # scrive nei log un messaggio riassuntivo (nome modello, punteggio CV, percorso del salvataggio)

    # gestione valutazione basandosi sulla modalità cross-validation
    if cv_mode.lower() == "loso":
        # per LOSO,usa i risultati della CV come valutazione finale
        print("LOSO mode: Using cross-validation results as final evaluation")
        results = {
            "cv_score": info.get("cv_score"), 
            "cv_mode": "loso",
            "n_subjects": len(groups_train.unique()), #numero di soggetti
            "note": "Final evaluation based on Leave-One-Subject-Out cross-validation"
        }
        # Per LOSO, possiamo ancora valutare sul dataset completo per confronto
        # ma il punteggio CV è la metrica principale
        try:
            full_results = evaluate_model(best_model, X_test, y_test, cfg.get("metrics", ["accuracy", "f1"])) # valutazione sul dataset completo
            results.update(full_results)
            results["note"] += " (with full dataset evaluation for comparison)"
        except Exception as e:
            log.warning("Could not evaluate on full dataset: %s", e)
    else:
        # Standard evaluation for GroupKFold
        results = evaluate_model(best_model, X_test, y_test, cfg.get("metrics", ["accuracy", "f1"]))
    
    # salva risultati valutazione nella stessa directory esperimento
    import json
    with open(exp_dir / "evaluation_results.json", "w", encoding="utf-8") as f: # salva risultati valutazione nel file evaluation_results.json
        json.dump(results, f, indent=2)
    log.info("Evaluation results saved to %s", exp_dir / "evaluation_results.json")

    # Genera plot se abilitato
    save_plots = cfg.get("save_plots", True)
    if save_plots:
        try:
            generate_all_plots(
                model=best_model,
                results=results,
                X_test=X_test,
                y_test=y_test,
                feature_names=list(X_train.columns),
                exp_dir=exp_dir,
                metrics=cfg.get("metrics", ["accuracy", "f1"]),
                cv_mode=cv_mode
            )
        except Exception as e:
            log.warning("Could not generate plots: %s", e)
    else:
        log.info("Plot generation disabled in configuration")

    # mostra informazioni cache finali
    final_cache_info = get_cache_info(memory)
    cache_stats = cache_manager.get_cache_stats() # 
    log.info("Final cache status: %s", final_cache_info)
    
    print("Done. Complete experiment saved to:", exp_dir)
    print(f"Cache status: {final_cache_info['size_mb']:.1f} MB, {final_cache_info['n_files']} files")
    print(f"Cache directory: {final_cache_info['cache_dir']}")


if __name__ == "__main__": # esecuzione del programma
    main()


