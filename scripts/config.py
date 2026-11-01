"""All configured relative paths are resolved against the configuration file."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.json"
SPLITS = {"random_5fold_5": 5, "prot_5fold_5": 5, "pep_5fold_5": 5, "pair_9fold_5": 9}

def load_config(path=DEFAULT_CONFIG):
    path = Path(path).expanduser().resolve()
    cfg = json.loads(path.read_text())
    for section in ("data", "features", "checkpoints"):
        for key, value in cfg[section].items():
            p = Path(value).expanduser()
            cfg[section][key] = str((path.parent / p).resolve() if not p.is_absolute() else p)
    out = Path(cfg["output_dir"]).expanduser()
    if out.is_absolute() or ".." in out.parts:
        raise ValueError("output_dir must be a project-relative path")
    cfg["output_dir"] = str((path.parent / out).resolve() if not out.is_absolute() else out)
    if cfg["model"]["d_model"] != 300:
        raise ValueError("The released model uses 300-dimensional residue representations.")
    if cfg["training"]["selection"] not in ("val_loss", "peptide_mcc"):
        raise ValueError("selection must be val_loss or peptide_mcc")
    return cfg

def split_file(cfg, split, fold, part):
    if split not in SPLITS or not 1 <= fold <= SPLITS[split]:
        raise ValueError(f"Invalid split/fold: {split}/{fold}")
    if part not in ("train", "val", "test"):
        raise ValueError(part)
    return Path(cfg["data"]["splits_dir"]) / split / f"{part}_data_5_fold_{fold}.pkl"
