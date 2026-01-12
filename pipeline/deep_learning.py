from __future__ import annotations
import json
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disabilita i log di TensorFlow
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf 
from joblib import Parallel, delayed
# Configurazione TensorFlow ottimizzata per velocità
tf.config.threading.set_inter_op_parallelism_threads(4)  # threads per le operazioni inter processo
tf.config.threading.set_intra_op_parallelism_threads(8)  
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold # shuffle dei dati per gruppo, per evitare che i dati di training e validazione siano dello stesso gruppo
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasClassifier
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class ClassificationDataset:
    X_train: np.ndarray # dati di training
    y_train: np.ndarray # etichette di training
    X_val: np.ndarray # dati di validazione
    y_val: np.ndarray # etichette di validazione
    groups_train: np.ndarray #identificatori dei soggetti per evitare che lo stesso soggetto compaia sia in training che validation
    groups_val: np.ndarray
    feature_stats: Dict[str, Any] #media e deviazione standard
    input_shape: Tuple[int, int] #forma dei dati in input al modello 


#fase di preparazione del dataset per la classificazione
def prepare_classification_dataset(
    signals: pd.DataFrame, 
    windows: pd.DataFrame,
    channels: Iterable[str] | None = None, 
    validation_split: float = 0.2,
    random_state: int = 42,
) -> ClassificationDataset:
    """Crea il dataset di classificazione con normalizzazione e split."""
    if channels is None:
        channels = ["x_D", "y_D", "z_D", "x_ND", "y_ND", "z_ND"]

    channels = list(channels) # assicura che channels sia una lista
    
    # raggruppa i dati per soggetto
    subject_frames = {
        sid: g.reset_index(drop=True) 
        for sid, g in signals.groupby("subject_id", sort=False) 
    }

    X_list: List[np.ndarray] = [] 
    y_list: List[int] = [] 
    groups: List[str] = []

    window_size = None

    for _, row in windows.iterrows(): # itera su ogni finestra
        sid = row["subject_id"] #estrae l'ID del soggetto dalla finestra corrente
        if sid not in subject_frames:
            continue
        subject_df = subject_frames[sid] # uso l'ID per recuperare tutti i dati del soggetto dal dizionario subject_frames
        start_idx = int(row["start_idx"])#estrae gli indici di inizio e fine della finestra
        end_idx = int(row["end_idx"])
        segment = subject_df.iloc[start_idx:end_idx] # legge i dati della finestra, segment è una matrice di dimensioni (window_size, num_channels)
        if segment.shape[0] == 0:
            continue

        data = segment[channels].to_numpy(dtype=np.float32) # converte i dati in un array numpy di dimensioni (window_size, num_channels)
        
        if window_size is None:
            window_size = data.shape[0] # legge la dimensione della finestra

        if data.shape[0] != window_size:
            # salta le finestre che non corrispondono alla dimensione attesa
            continue

        X_list.append(data) # aggiunge i dati alla lista
        label = int(row["label"]) # legge l'etichetta della finestra
        y_list.append(label) # aggiunge l'etichetta alla lista
        groups.append(str(sid)) # aggiunge l'ID del soggetto alla lista

    X = np.stack(X_list, axis=0) #impiliamo tutte le finestre una sopra l'altra per ottenere un array tridimensionale di dimensioni (num_windows, window_size, num_channels)
    y = np.asarray(y_list, dtype=np.int32) #converte la lista di label in un array di interi
    groups_arr = np.asarray(groups)#converte la lista di ID dei soggetti in un array 

    feature_stats = {
        "channels": channels,
        "window_size": int(window_size),
        "n_features": int(len(channels)),
    }

    input_shape = (int(window_size), len(channels)) # dimensione dell'input del modello, (window_size, num_channels)

    return ClassificationDataset(
        X_train=X,
        y_train=y,
        X_val=np.empty((0, *X.shape[1:]), dtype=np.float32),
        y_val=np.empty((0,), dtype=np.int32),
        groups_train=groups_arr,
        groups_val=np.empty((0,), dtype=groups_arr.dtype),
        feature_stats=feature_stats,
        input_shape=input_shape,
    )

#decide come vengono scelti gli iperparametri casuali per ogni modello
def _sample_hyperparameter(entry: Dict[str, Any], rng: np.random.Generator) -> Any:
    entry_type = entry.get("type", "choice")
    if entry_type == "choice":
        values = entry["values"]
        if not values:
            raise ValueError("Choice hyperparameter requires non-empty values.")
        idx = int(rng.integers(0, len(values)))
        return values[idx] # sceglie un valore casuale tra i valori possibili
    if entry_type == "log_uniform":
        low = math.log(entry["low"]) # sceglie in modo continuo su scala logaritmica
        high = math.log(entry["high"])
        return float(np.exp(rng.uniform(low, high))) 
    if entry_type == "uniform":
        return float(rng.uniform(entry["low"], entry["high"])) #sceglie in modo continuo su scala lineare
    raise ValueError(f"Unsupported hyperparameter type: {entry_type}")


def _build_cnn_1d( # prende in input una finestra temporale di accelerazione applica varie convoluzioni per estrarre pattern nel tempo, e produce in uscita una predizione
    input_shape: Tuple[int, int],
    params: Dict[str, Any],
    output_activation: str,
    learning_rate: float,
    loss: str,
    metrics: List[str],
) -> keras.Model:
    kernel_size = int(params["kernel_size"]) #quanti capioni per volta osserva la CNN 
    n_filters = int(params["n_filters"])
    dropout = float(params["dropout"])

    inputs = keras.Input(shape=input_shape, name="signals") # si definisce l'ingresso del modello 
    x = inputs
    #primo blocco convoluzionale che cattura pattern locali(movimenti brevi)
    x = layers.Conv1D(filters=n_filters, kernel_size=kernel_size, padding="same")(x) #applica n filtri convoluzionali di lunghezza kernel_size, ogni filtro scorre nel tempo e cattura pattern di movimento locali
    x = layers.BatchNormalization()(x) #stabilizza e velocizza l'apprendiemnto 
    x = layers.Activation("relu")(x) # funzione di attivazione non lineare
    x = layers.MaxPooling1D(pool_size=2)(x) # dimezza la lunghezza temporale, riducendo la dimensionalità e concentrandosi sui pattern più forti
    #secondo blocco che inizia a riconoscere pattern più lunghi o combinazioni di movimenti
    x = layers.Conv1D(filters=n_filters * 2, kernel_size=kernel_size, padding="same")(x) # raddoppia i filtri cioè aumenta la profondità della rappresentazione 
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x) #altro max pooling per ridurre la lunghezza temporale
    #terzo blocco convoluzionale
    x = layers.Conv1D(filters=n_filters * 2, kernel_size=kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling1D()(x) # condensa tutta la seuqneza in un unico vettore medio per filtro

    if dropout > 0:
        x = layers.Dropout(dropout)(x)  

    x = layers.Dense(n_filters, activation="relu")(x) # si impara a combinare pattern diversi
    if dropout > 0:
        x = layers.Dropout(dropout)(x)# secondo dropout per evitare che il modello memorizzi troppo i dati del training

    outputs = layers.Dense(1, activation=output_activation, name="output")(x) #Dense(1) perchè deve produrre una probabilità se il problema è di classificazione, altrimenti un valore continuo se è di regressione
  
    model = keras.Model(inputs=inputs, outputs=outputs, name="cnn_1d_classifier") # creazione del modello
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate) #algoritmo che aggiorna i pesi del modello
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics) #si definiscono gli aspetti del training (che cosa ottimizzare(loss),come ottimizzare(optimizer) cosa monitorare durante l'allenamento (metrics))
    return model


def _build_lstm(
    input_shape: Tuple[int, int],
    params: Dict[str, Any],
    output_activation: str,# sigmoid per classificazione e linear per regressione
    learning_rate: float,
    loss: str,
    metrics: List[str],
) -> keras.Model:
    units = int(params["lstm_units"])# quanti neuroni ha ciascun layer LSTM
    dropout = float(params["dropout"]) #quanti neuroni vengono disattivati casualmente durante l'allenamento

    inputs = keras.Input(shape=input_shape, name="signals")
    x = inputs
    x = layers.Masking()(x) # aggiunge layer che ignora i valori nulli o padding
    x = layers.LSTM(units=units, return_sequences=True)(x) #legge la serie temporale campione per campione, memorizzando le info precedenti
    x = layers.Dropout(dropout)(x) if dropout > 0 else x # disattiva casualmente alcuni neuroni per evitare overfitting
    x = layers.LSTM(units=units)(x) # legge tutta la sequenza generata dalla prima ,restituisce una rappresentazione globale dell'intera finestra temporale
    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    x = layers.Dense(units, activation="relu")(x) #combina le info estratte dalle LSTM,aumenta la capacità espressiva del modello
    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(1, activation=output_activation, name="output")(x)#Dense(1) perchè deve produrre una probabilità se il problema è di classificazione, altrimenti un valore continuo se è di regressione

    model = keras.Model(inputs=inputs, outputs=outputs, name="lstm_classifier")
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


class _PerChannelStandardizer(BaseEstimator, TransformerMixin):
    """Normalizza ogni canale (samples e window) utilizzando media e deviazione standard del fold di training."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps #evita divisioni per zero
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        if X.ndim != 3: # controlla che l'input sia tridimensionale (samples, window, channels)
            raise ValueError("Expected X with shape (samples, window, channels)")
        mean = np.mean(X, axis=(0, 1), keepdims=True) # calcola la media per canale
        std = np.std(X, axis=(0, 1), keepdims=True) # calcola la deviazione standard per canale
        std = np.where(std < self.eps, self.eps, std) 
        self.mean_ = mean.astype(np.float32)#salvataggio dei parametri di normalizzazione
        self.std_ = std.astype(np.float32) 
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer must be fitted before calling transform.")
        return (X - self.mean_) / self.std_ #applicazione della normalizzazione

#costruisce un oggetto KerasClassifier, viene costruito un modello CNN1D per un numero totale di volte pari a k folds
def _make_keras_classifier(
    input_shape: Tuple[int, int],
    params: Dict[str, Any],
    model_type: str,
    epochs: int,
    random_state: int,
) -> KerasClassifier:
    """Crea un KerasClassifier per mantenere compatibilità sklearn."""

    # Mappa dei builder per tipo di modello
    builder_map = {
        "cnn_1d": _build_cnn_1d,
        "lstm": _build_lstm,
    }
    
    if model_type not in builder_map:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model_builder_fn = builder_map[model_type] #dizionario che associa una stringa al costruttore del modello  
    
    def _model_builder():
        return model_builder_fn(
            input_shape=input_shape,
            params=params,
            output_activation="sigmoid",
            learning_rate=float(params.get("learning_rate", 1e-3)),
            loss="binary_crossentropy",
            metrics=_keras_metrics(),
        )

    batch_size = int(params.get("batch_size", 32))

    return KerasClassifier(
        model=_model_builder,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        validation_split=0.1,
        shuffle=True,
        random_state=random_state,
    )

def _build_cv_pipeline(
    input_shape: Tuple[int, int],
    params: Dict[str, Any],
    model_type: str,
    epochs: int,
    random_state: int,
) -> Pipeline:
    """"Costruisce una pipeline sklearn composta da:
    1. Standardizzazione per canale (senza data leakage)
    2. Classificatore Keras (CNN o LSTM)
    Permette compatibilità tra modelli di Deep Learning e sklearn per cross-validation."""
    classifier = _make_keras_classifier(
        input_shape=input_shape,
        params=params,
        model_type=model_type,#scelta del modello 
        epochs=epochs,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("standardizer", _PerChannelStandardizer()),
            ("classifier", classifier),
        ]
    )


#divide i dati in 5 fold per il cross validation
def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    params: Dict[str, Any],
    model_type: str,
    n_splits: int = 5,
    epochs: int = 50,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Esegue k-fold cross-validation con GroupKFold per evitare data leakage tra soggetti
    """
    gkf = GroupKFold(n_splits=n_splits) # divide i dati in k fold, assicurando che i dati di uno stesso soggetto non compaiano in più fold

    cv_results = {
        "accuracy": [],
        "auc": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "loss": [],
    }

    fold_predictions: List[float] = []
    fold_true_labels: List[int] = []

    input_shape = (X.shape[1], X.shape[2])

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)): # itera su ogni fold
        print(f"Processing fold {fold + 1}/{n_splits}")

        X_train_fold, X_val_fold = X[train_idx], X[val_idx] # divide i dati in training e validation per il fold corrente
        y_train_fold, y_val_fold = y[train_idx], y[val_idx] # divide le etichette in training e validation per il fold corrente

        tf.keras.utils.set_random_seed(random_state + fold)
        pipeline = _build_cv_pipeline(
            input_shape=input_shape,
            params=params,
            model_type=model_type,
            epochs=epochs,
            random_state=random_state + fold,
        )

        pipeline.fit(X_train_fold, y_train_fold) # allena il modello sul fold corrente

        y_pred = pipeline.predict(X_val_fold) # predizioni discrete (0 o 1)
        y_proba = pipeline.predict_proba(X_val_fold) # predizioni probabilistiche
        if y_proba.ndim == 2:
            if y_proba.shape[1] == 1: 
                y_proba = y_proba[:, 0]
            else:
                y_proba = y_proba[:, 1]
        y_proba = y_proba.reshape(-1)

        accuracy = accuracy_score(y_val_fold, y_pred) 
        precision = precision_score(y_val_fold, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val_fold, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val_fold, y_pred, average='weighted', zero_division=0)

        try:
            auc = roc_auc_score(y_val_fold, y_proba)
        except ValueError:
            auc = float("nan")

        try:
            loss_value = log_loss(y_val_fold, np.clip(y_proba, 1e-7, 1 - 1e-7))
        except ValueError:
            loss_value = float("nan")

        cv_results["accuracy"].append(float(accuracy))
        cv_results["precision"].append(float(precision))
        cv_results["recall"].append(float(recall))
        cv_results["f1_score"].append(float(f1))
        cv_results["auc"].append(float(auc))
        cv_results["loss"].append(float(loss_value))

        fold_predictions.extend(y_proba.tolist())
        fold_true_labels.extend(y_val_fold.tolist())

        del pipeline
        tf.keras.backend.clear_session()

    # Calcola statistiche finali
    mean_results = {}
    std_results = {}

    for k, v in cv_results.items(): # calcola media e deviazione standard per ogni metrica di ciascun fold
        if len(v) > 0:
            values = np.asarray(v, dtype=np.float32)
            mean_val = float(np.nanmean(values))
            std_val = float(np.nanstd(values))
            if np.isnan(mean_val):
                mean_val = 0.0
            if np.isnan(std_val):
                std_val = 0.0
            mean_results[f"mean_{k}"] = mean_val
            std_results[f"std_{k}"] = std_val
        else:
            mean_results[f"mean_{k}"] = 0.0
            std_results[f"std_{k}"] = 0.0

    return {
        **mean_results,
        **std_results,
        "cv_results": cv_results,
        "all_predictions": np.array(fold_predictions),
        "all_true_labels": np.array(fold_true_labels),
    }


def _keras_metrics() -> List[keras.metrics.Metric | str]:#funzione che definisce le metriche da monitorare durante l'allenamento
    """
    Metriche compatibili per classificazione binaria.
    Usiamo metriche standard che funzionano con output (batch_size, 1).
    """
    return [
        "accuracy", 
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall")
        # F1-score sarà calcolato manualmente per evitare problemi di compatibilità
    ]

#si esegue una random search su n_trials configurazioni casuali di iperparametri per trovare il miglior modello
def random_search_training(
    dataset: ClassificationDataset, #X_train, X_val, y_train, y_val, input_shape e feature_stats
    model_type: str, #cnn 1d o lstm
    base_cfg: Dict[str, Any], #config yaml
) -> Dict[str, Any]:
    merged_cfg = base_cfg.copy()
    validation_split = merged_cfg["validation_split"]
    random_search_cfg = merged_cfg["random_search"]
    n_trials = int(random_search_cfg["n_trials"])
    epochs = int(merged_cfg["epochs"])
    n_splits = int(merged_cfg["n_splits"])  # Per cross-validation
    n_jobs = int(random_search_cfg["n_jobs"])

    #definisce lo spazio di ricerca degli iperparametri
    param_space = merged_cfg["param_space"].copy()


    builder_map: Dict[str, Callable[..., keras.Model]] = { #mappa che associa ogni tipo di modello a una funzione di costruzione
        "cnn_1d": _build_cnn_1d,
        "lstm": _build_lstm,
    }

    if model_type not in builder_map:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_builder = builder_map[model_type] #funzione che verrà chiamata a ogni trial per creare il modello compilato con i parametri correnti

    tf.keras.utils.set_random_seed(merged_cfg["random_state"])

    best_result: Dict[str, Any] | None = None #tiene traccia del miglior risultato trovato finora
    all_trial_results = []  # Salva tutti i risultati per analisi
    
    X_full = dataset.X_train
    y_full = dataset.y_train
    groups_full = dataset.groups_train
    print(f"Using {n_splits}-fold cross-validation with {len(np.unique(groups_full))} unique subjects")

    base_seed = merged_cfg["random_state"]
    seed_sequence = np.random.SeedSequence(base_seed)
    trial_seed_sequences = seed_sequence.spawn(n_trials)#ogni trial avrà il suo generatore di seed separato per garantire indipendenza tra i trial

    def _build_trial_params(trial_idx: int, trial_rng: np.random.Generator) -> Dict[str, Any]:
        params = {}
        for key, entry in param_space.items():
            params[key] = _sample_hyperparameter(entry, trial_rng)
        params["learning_rate"] = float(params["learning_rate"])
        params["_seed"] = base_seed + trial_idx
        return params
    #crea la lista dei trial con i relativi parametri
    trial_specs: List[Tuple[int, Dict[str, Any]]] = []
    for trial_idx, trial_seq in enumerate(trial_seed_sequences):
        trial_rng = np.random.default_rng(trial_seq)
        trial_params = _build_trial_params(trial_idx, trial_rng)
        trial_specs.append((trial_idx, trial_params)) #tabella dei tentativi che andrà eseguita

    def _execute_trial(trial_idx: int, params: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n=== Trial {trial_idx + 1}/{n_trials} ===")
        cv_results = cross_validate_model( #chiama cross_validate_model per eseguire la cross validation con i parametri correnti
            X=X_full,
            y=y_full,
            groups=groups_full,
            params=params,
            model_type=model_type,
            n_splits=n_splits, # numero di fold per la cross validation
            epochs=epochs,
            random_state=params["_seed"],
        )
        trial_result = {
            "trial": trial_idx,
            "model_type": model_type,
            "params": params,
            "cv_results": cv_results,
            "mean_f1_score": cv_results["mean_f1_score"],
            "std_f1_score": cv_results["std_f1_score"],
            "mean_accuracy": cv_results["mean_accuracy"],
            "std_accuracy": cv_results["std_accuracy"],
            "validation_method": "cross_validation",
        }
        selection_metric = cv_results["mean_f1_score"] #metrica usata per scegliere il miglior modello
        print(f"Trial {trial_idx + 1} - Selection metric: {selection_metric:.4f}")
        return trial_result

    if n_jobs == 1:
        trial_results = [_execute_trial(trial_idx, params) for trial_idx, params in trial_specs]
    else:
        print(f"Eseguo Random Search con parallelizzazione n_jobs={n_jobs}")
        trial_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_execute_trial)(trial_idx, params) for trial_idx, params in trial_specs # esegue i trial in parallelo
        )

    for trial_result in trial_results: #scorre tutti i trial 
        selection_metric = trial_result["mean_f1_score"]
        all_trial_results.append(trial_result)
        if best_result is None or selection_metric > best_result.get("mean_f1_score", 0):
            best_result = trial_result
            print(f"New best result found!")

    if best_result is None:
        raise RuntimeError(f"Random search failed for model {model_type}")
    
    # Addestra un modello finale sui dati completi con i migliori parametri
    print("\nTraining final model on full dataset with best hyperparameters...")
    final_model = model_builder(
        input_shape=dataset.input_shape,
        params=best_result["params"],
        output_activation="sigmoid",
        learning_rate=float(best_result["params"]["learning_rate"]),
        loss="binary_crossentropy",
        metrics=_keras_metrics(),
    )
    
    batch_size = int(best_result["params"].get("batch_size", 32))
    #training finale
    history = final_model.fit(
        dataset.X_train, dataset.y_train,
        validation_data=(dataset.X_val, dataset.y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )
    
    best_result["final_model"] = final_model #modello finale addestrato con i migliori iperparametri
    best_result["final_history"] = history.history #storico dell'addestramento finale

    return best_result # contiene migliori iperparametri, metriche di CV, tutti i trial, modello finale per la classificazione


#salva i risultati della random search in un file json
def save_training_metadata(
    results: List[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    def _to_serialisable(obj: Any) -> Any:
        """Converte oggetti numpy/tensorflow in tipi compatibili JSON."""
        if isinstance(obj, (np.generic,)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serialisable(v) for v in obj]
        return obj

    serialisable = []
    for res in results:
        serialisable.append(
            {
                "model_type": res["model_type"],
                "best_trial": res.get("trial"),
                "hyperparameters": {
                    k: _to_serialisable(v)
                    for k, v in res["params"].items()
                    if k not in {"_seed"}
                },
                # Solo cross-validation - usa sempre mean values
                "mean_accuracy": _to_serialisable(res.get("mean_accuracy", 0.0)),
                "std_accuracy": _to_serialisable(res.get("std_accuracy", 0.0)),
                "mean_f1_score": _to_serialisable(res.get("mean_f1_score", 0.0)),
                "std_f1_score": _to_serialisable(res.get("std_f1_score", 0.0)),
                "validation_method": "cross_validation",
                "cv_results": _to_serialisable(res.get("cv_results", {})),
            }
        )
    out_path = output_dir / "classification_random_search.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)
    return out_path

#chiusura della fase di classificazione, esporta il modello migliore
def export_best_models(
    best_result: Dict[str, Any],
    dataset: ClassificationDataset,
    models_dir: Path,
) -> Dict[str, Any]:
    models_dir.mkdir(parents=True, exist_ok=True)
    classification_path = models_dir / "best_model_classification.keras" #salva il modello migliore per la classificazione
    best_result["final_model"].save(classification_path, include_optimizer=True) #salva il modello con i pesi ottimizzati

    metadata = { #dizionario che contiene le informazioni sul modello migliore
        "model_type": best_result["model_type"],
        "classification_model_path": str(classification_path),
        "input_shape": dataset.input_shape,
        "feature_stats": dataset.feature_stats,
        "best_hyperparameters": {
            k: v for k, v in best_result["params"].items() if k not in {"_seed"}
        },
    }

    metadata_path = models_dir / "best_model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


__all__ = [ 
    "ClassificationDataset",
    "prepare_classification_dataset",
    "cross_validate_model",
    "random_search_training",
    "save_training_metadata",
    "export_best_models",
]
