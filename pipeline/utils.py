"""
File di utilità per la pipeline di elaborazione dei dati. Contiene funzioni per la gestione dei percorsi, il caricamento delle configurazioni,
la configurazione del logging e l'impostazione del seed per la riproducibilità.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json
import yaml
import logging
import numpy as np
import random


@dataclass
class Paths:
    data_dir: Path
    output_dir: Path
    preprocessed_dir: Path
    models_dir: Path
    metrics_dir: Path
    plots_dir: Path


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_paths(cfg: Dict[str, Any]) -> Paths:
    data_dir = Path(cfg["data_dir"]).expanduser().resolve()
    output_dir = Path(cfg["output_dir"]).expanduser().resolve()
    preprocessed_dir = output_dir / "preprocessed"
    models_dir = output_dir / "models"
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    for p in [output_dir, preprocessed_dir, models_dir, metrics_dir, plots_dir]:
        p.mkdir(parents=True, exist_ok=True)
    return Paths(
        data_dir=data_dir,
        output_dir=output_dir,
        preprocessed_dir=preprocessed_dir,
        models_dir=models_dir,
        metrics_dir=metrics_dir,
        plots_dir=plots_dir,
    )


def setup_logging(output_dir: Path) -> logging.Logger:
    log = logging.getLogger("tesi-pipeline")
    if not log.handlers:
        log.setLevel(logging.INFO)
        fh = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        log.addHandler(fh)
        log.addHandler(ch)
    return log


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)


