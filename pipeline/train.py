"""
Model training: stratified split, model definitions, and simple hyperparameter search.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
from datetime import datetime
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, GroupKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
try:
    from sklearn.utils.fixes import loguniform
except ImportError:
    from scipy.stats import loguniform
from scipy.stats import uniform, randint
import joblib

try:
    from xgboost import XGBClassifier  # optional
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import optuna
    from optuna.integration import OptunaSearchCV
    from optuna.pruners import MedianPruner
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False


def _create_cv_object(cv_mode: str, cv_folds: int, groups: pd.Series) -> Any:
    """
    Create appropriate cross-validation object based on cv_mode.
    
    Args:
        cv_mode: "groupkfold" or "loso"
        cv_folds: Number of folds for GroupKFold (ignored for LOSO)
        groups: Group labels for cross-validation
    
    Returns:
        Cross-validation object
    """
    if cv_mode.lower() == "loso":
        # Leave-One-Subject-Out: each subject is left out once
        n_subjects = groups.nunique()
        print(f"Using Leave-One-Subject-Out CV with {n_subjects} subjects")
        return LeaveOneGroupOut()
    elif cv_mode.lower() == "groupkfold":
        # Group K-Fold: subjects are split into k folds
        print(f"Using GroupKFold CV with {cv_folds} folds")
        return GroupKFold(n_splits=cv_folds)
    else:
        raise ValueError(f"Unknown cv_mode: {cv_mode}. Must be 'groupkfold' or 'loso'")


def _build_model(name: str, random_seed: int = 42):
    name = name.lower()
    if name == "randomforest":
        return RandomForestClassifier(n_estimators=300, random_state=random_seed)
    if name == "svm":
        return SVC(kernel="rbf", probability=True, random_state=random_seed)
    if name == "logisticregression":
        return LogisticRegression(max_iter=1000, random_state=random_seed)
    if name == "xgboost" and HAS_XGB:
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=random_seed)
    raise ValueError("Unknown or unavailable model: " + name)


def _param_grid(name: str) -> Dict[str, Any]:
    """Grid search parameters (exhaustive but expensive)"""
    name = name.lower()
    if name == "randomforest":
        return {"n_estimators": [200, 300, 500], "max_depth": [None, 10, 20]}
    if name == "svm":
        return {"C": [0.1, 1, 10], "gamma": ["scale", 0.01, 0.1, 1.0]}
    if name == "logisticregression":
        return {"C": [0.1, 1, 10]}
    if name == "xgboost" and HAS_XGB:
        return {"max_depth": [3, 4, 6], "n_estimators": [200, 300], "learning_rate": [0.05, 0.1]}
    return {}


def _param_distributions(name: str) -> Dict[str, Any]:
    """Randomized search parameter distributions (efficient sampling)"""
    name = name.lower()
    if name == "randomforest":
        return {
            "n_estimators": randint(100, 1000),
            "max_depth": [None] + list(range(5, 50, 5)),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False]
        }
    if name == "svm":
        return {
            "C": loguniform(1e-3, 1e3),
            "gamma": loguniform(1e-4, 1e1),
            "kernel": ["rbf", "poly", "sigmoid"]
        }
    if name == "logisticregression":
        return {
            "C": loguniform(1e-3, 1e3),
            "penalty": ["l1", "l2", "elasticnet"],
            "solver": ["liblinear", "saga"]
        }
    if name == "xgboost" and HAS_XGB:
        return {
            "n_estimators": randint(100, 1000),
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "reg_alpha": uniform(0, 1),
            "reg_lambda": uniform(0, 1)
        }
    return {}


def _optuna_objective(trial, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, groups: pd.Series, cv: Any):
    """Optuna objective function for Bayesian optimization"""
    if model_name.lower() == "randomforest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 5, 50) if trial.suggest_categorical("use_max_depth", [True, False]) else None,
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": 42
        }
        model = RandomForestClassifier(**params)
    
    elif model_name.lower() == "svm":
        params = {
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1e1, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
            "random_state": 42
        }
        model = SVC(**params)
    
    elif model_name.lower() == "logisticregression":
        params = {
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"]),
            "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
            "random_state": 42,
            "max_iter": 1000
        }
        model = LogisticRegression(**params)
    
    elif model_name.lower() == "xgboost" and HAS_XGB:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
            "random_state": 42,
            "eval_metric": "logloss"
        }
        model = XGBClassifier(**params)
    
    else:
        raise ValueError(f"Unknown model for Optuna: {model_name}")
    
    # Cross-validation with pruning support
    scores = []
    for step, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        score = f1_score(y_val, preds, average="weighted")
        scores.append(score)
        
        # Report intermediate score for pruning
        trial.report(score, step)
        
        # Check if trial should be pruned
        if trial.should_prune():
            print(f"Trial pruned at step {step} with score {score:.4f}")
            if HAS_OPTUNA:
                raise optuna.TrialPruned()
            else:
                raise Exception("TrialPruned")
    
    final_score = np.mean(scores)
    return final_score


def stratified_split(feats: pd.DataFrame, test_size: float = 0.2, random_seed: int = 42, cv_mode: str = "groupkfold") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data ensuring no subject appears in both train and test sets."""
    if cv_mode.lower() == "loso":
        # For LOSO, we don't need a separate train/test split as each subject is left out once
        # Return the same data for both train and test - the CV will handle the splitting
        print("LOSO mode: Using all data for cross-validation (no separate train/test split)")
        return feats.copy(), feats.copy()
    
    # Original GroupKFold logic
    subjects = feats["subject_id"].unique()
    n_subjects = len(subjects)
    n_test_subjects = max(1, int(n_subjects * test_size))  # At least 1 subject for test
    
    # Randomly select test subjects
    np.random.seed(random_seed)
    test_subjects = np.random.choice(subjects, size=n_test_subjects, replace=False)
    
    # Split based on subjects
    train_mask = ~feats["subject_id"].isin(test_subjects)
    test_mask = feats["subject_id"].isin(test_subjects)
    
    train = feats[train_mask].copy()
    test = feats[test_mask].copy()
    
    return train, test


def train_models(X_train: pd.DataFrame, y_train: pd.Series, groups: pd.Series, config: Dict[str, Any], output_dir: Path) -> Tuple[Any, Dict[str, Any], Path]:
    """
    Train models with efficient hyperparameter search (RandomizedSearchCV or Optuna).
    Returns: (best_model, info_dict, experiment_dir)
    """
    models = config.get("models", ["RandomForest"])  # list of model names
    random_seed = config.get("random_seed", 42)
    search_config = config.get("hyperparameter_search", {})
    search_method = search_config.get("method", "randomized")
    n_trials = search_config.get("n_trials", 50)
    cv_folds = search_config.get("cv_folds", 5)
    cv_mode = search_config.get("cv_mode", "groupkfold")
    n_jobs = search_config.get("n_jobs", -1)
    
    # Create appropriate CV object based on cv_mode
    cv = _create_cv_object(cv_mode, cv_folds, groups)
    best_model = None
    best_score = -np.inf
    best_name = None
    best_params: Dict[str, Any] = {}
    
    # Create timestamped experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    exp_dir = output_dir / f"exp_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    all_cv_results = []
    
    print(f"Using {search_method} hyperparameter search with {n_trials} trials")
    print(f"Cross-validation mode: {cv_mode}")
    
    for name in models:
        print(f"Training {name}...")
        model = _build_model(name, random_seed)
        
        if search_method == "grid":
            # Traditional GridSearchCV (expensive)
            params = _param_grid(name)
            if params:
                search = GridSearchCV(model, params, scoring="f1", cv=cv, n_jobs=n_jobs, return_train_score=True)
                search.fit(X_train, y_train, groups=groups)
                score = search.best_score_
                candidate = search.best_estimator_
                used_params = search.best_params_
                
                # Save detailed CV results
                cv_results_df = pd.DataFrame(search.cv_results_)
                cv_results_df['model_name'] = name
                all_cv_results.append(cv_results_df)
            else:
                # No parameters to tune
                model.fit(X_train, y_train)
                candidate = model
                scores = []
                for tr, va in cv.split(X_train, y_train, groups):
                    candidate.fit(X_train.iloc[tr], y_train.iloc[tr])
                    preds = candidate.predict(X_train.iloc[va])
                    scores.append(f1_score(y_train.iloc[va], preds, average="weighted"))
                score = float(np.mean(scores))
                used_params = {}
        
        elif search_method == "randomized":
            # RandomizedSearchCV (efficient)
            param_dist = _param_distributions(name)
            if param_dist:
                search = RandomizedSearchCV(
                    model, param_dist, n_iter=n_trials, scoring="f1", 
                    cv=cv, n_jobs=n_jobs, random_state=random_seed, return_train_score=True
                )
                search.fit(X_train, y_train, groups=groups)
                score = search.best_score_
                candidate = search.best_estimator_
                used_params = search.best_params_
                
                # Save detailed CV results
                cv_results_df = pd.DataFrame(search.cv_results_)
                cv_results_df['model_name'] = name
                all_cv_results.append(cv_results_df)
            else:
                # No parameters to tune
                model.fit(X_train, y_train)
                candidate = model
                scores = []
                for tr, va in cv.split(X_train, y_train, groups):
                    candidate.fit(X_train.iloc[tr], y_train.iloc[tr])
                    preds = candidate.predict(X_train.iloc[va])
                    scores.append(f1_score(y_train.iloc[va], preds, average="weighted"))
                score = float(np.mean(scores))
                used_params = {}
        
        elif search_method == "optuna" and HAS_OPTUNA:
            # Optuna Bayesian optimization with pruning (most efficient)
            print(f"Running Optuna optimization for {name} with MedianPruner...")
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2, interval_steps=1)
            study = optuna.create_study(
                direction="maximize", 
                sampler=optuna.samplers.TPESampler(seed=random_seed),
                pruner=pruner
            )
            
            def objective(trial):
                return _optuna_objective(trial, name, X_train, y_train, groups, cv)
            
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            # Log pruning statistics
            pruned_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
            print(f"Pruning statistics: {pruned_count}/{len(study.trials)} trials were pruned")
            
            # Get best parameters and create model
            best_trial = study.best_trial
            score = best_trial.value
            used_params = best_trial.params
            
            # Recreate model with best parameters
            if name.lower() == "randomforest":
                candidate = RandomForestClassifier(**used_params, random_state=random_seed)
            elif name.lower() == "svm":
                candidate = SVC(**used_params, random_state=random_seed)
            elif name.lower() == "logisticregression":
                candidate = LogisticRegression(**used_params, random_state=random_seed)
            elif name.lower() == "xgboost" and HAS_XGB:
                candidate = XGBClassifier(**used_params, random_state=random_seed)
            else:
                candidate = model
            
            # Fit final model
            candidate.fit(X_train, y_train)
            
            # Save Optuna study results with pruning information
            pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            
            optuna_results = {
                'model_name': name,
                'best_value': study.best_value,
                'best_params': study.best_params,
                'n_trials': len(study.trials),
                'n_completed': len(completed_trials),
                'n_pruned': len(pruned_trials),
                'pruning_rate': len(pruned_trials) / len(study.trials) if study.trials else 0,
                'pruner_type': 'MedianPruner',
                'study_name': study.study_name
            }
            with open(exp_dir / f"optuna_{name.lower()}_results.json", "w") as f:
                json.dump(optuna_results, f, indent=2)
            
            print(f"Optuna completed: {len(completed_trials)} completed, {len(pruned_trials)} pruned trials")
        
        else:
            # Fallback: no hyperparameter tuning
            if search_method == "optuna" and not HAS_OPTUNA:
                print(f"Warning: Optuna not available, falling back to no tuning for {name}")
            model.fit(X_train, y_train)
            candidate = model
            scores = []
            for tr, va in cv.split(X_train, y_train):
                candidate.fit(X_train.iloc[tr], y_train.iloc[tr])
                preds = candidate.predict(X_train.iloc[va])
                scores.append(f1_score(y_train.iloc[va], preds, average="weighted"))
            score = float(np.mean(scores))
            used_params = {}

        print(f"{name} best score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_model = candidate
            best_name = name
            best_params = used_params

    # Save complete CV results
    if all_cv_results:
        combined_cv_results = pd.concat(all_cv_results, ignore_index=True)
        combined_cv_results.to_csv(exp_dir / "cv_results.csv", index=False)
    
    # Save best model
    model_path = exp_dir / "best_model.pkl"
    joblib.dump(best_model, model_path)
    
    # Prepare final metrics
    final_metrics = {
        "model": best_name,
        "cv_score": best_score,
        "params": best_params,
        "experiment_dir": str(exp_dir),
        "timestamp": timestamp,
        "random_seed": random_seed,
        "search_method": search_method,
        "n_trials": n_trials,
        "cv_mode": cv_mode,
        "cv_folds": cv_folds if cv_mode.lower() == "groupkfold" else groups.nunique()
    }
    
    # Save metrics JSON
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    return best_model, final_metrics, exp_dir


def save_best_model(model: Any, models_dir: Path) -> Path:
    """Legacy function - now handled by train_models with timestamped directories."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "best_model.pkl"
    joblib.dump(model, path)
    return path
