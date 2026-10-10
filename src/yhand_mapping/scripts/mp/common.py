import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if p.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if p.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file suffix: {p.suffix}. Use .csv or .parquet")


def ensure_required_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {name} columns: {missing}")


def resolve_existing_column(df: pd.DataFrame, candidates: List[Optional[str]], what: str) -> str:
    for c in candidates:
        if c and c in df.columns:
            return c
    raise ValueError(f"Cannot resolve {what}. Tried: {candidates}")


def parse_active_finger(v) -> Optional[int]:
    if pd.isna(v):
        return None
    if isinstance(v, (int, np.integer)):
        idx = int(v)
        if 0 <= idx < 5:
            return idx
        raise ValueError(f"active_finger int out of range [0,4]: {v}")
    s = str(v).strip().lower()
    if s.isdigit():
        idx = int(s)
        if 0 <= idx < 5:
            return idx
        raise ValueError(f"active_finger str int out of range [0,4]: {v}")
    aliases = {
        "thumb": 0,
        "index": 1,
        "middle": 2,
        "ring": 3,
        "little": 4,
        "pinky": 4,
    }
    if s not in aliases:
        raise ValueError(f"Unknown active_finger value: {v}")
    return aliases[s]


def build_active_onehot(df: pd.DataFrame, active_col: Optional[str]) -> np.ndarray:
    if active_col is None:
        return np.zeros((len(df), 0), dtype=np.float32)
    if active_col not in df.columns:
        raise ValueError(f"active_finger column not found: {active_col}")
    out = np.zeros((len(df), 5), dtype=np.float32)
    for i, v in enumerate(df[active_col].tolist()):
        idx = parse_active_finger(v)
        if idx is not None:
            out[i, idx] = 1.0
    return out


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = x.mean(axis=0).astype(np.float32)
        std = x.std(axis=0).astype(np.float32)
        std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> Dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Standardizer":
        return cls(
            mean=np.array(d["mean"], dtype=np.float32),
            std=np.array(d["std"], dtype=np.float32),
        )


def compute_zero_reference_per_session(
    df: pd.DataFrame,
    session_col: str,
    q_cols: List[str],
    m_cols: List[str],
    reset_col: Optional[str] = None,
    infer_zero_quantile: float = 0.05,
) -> pd.DataFrame:
    """
    For each session:
    - if reset_col exists and has positive rows, use their median as zero reference
    - else infer zero reference from the lowest motor-norm quantile
    """
    ref_rows = []
    for sid, sdf in df.groupby(session_col, sort=False):
        if reset_col is not None and reset_col in sdf.columns and sdf[reset_col].astype(bool).any():
            zdf = sdf[sdf[reset_col].astype(bool)]
        else:
            motor_norm = np.linalg.norm(sdf[m_cols].to_numpy(dtype=np.float32), axis=1)
            k = max(1, int(math.ceil(len(sdf) * infer_zero_quantile)))
            zdf = sdf.iloc[np.argsort(motor_norm)[:k]]

        q0 = zdf[q_cols].median(axis=0).to_numpy(dtype=np.float32)
        m0 = zdf[m_cols].median(axis=0).to_numpy(dtype=np.float32)

        row = {session_col: sid}
        for i, c in enumerate(q_cols):
            row[f"{c}__zero"] = float(q0[i])
        for i, c in enumerate(m_cols):
            row[f"{c}__zero"] = float(m0[i])
        ref_rows.append(row)

    ref_df = pd.DataFrame(ref_rows)
    out = df.merge(ref_df, on=session_col, how="left", validate="many_to_one")

    for c in q_cols:
        out[f"d_{c}"] = out[c].astype(np.float32) - out[f"{c}__zero"].astype(np.float32)
    for c in m_cols:
        out[f"d_{c}"] = out[c].astype(np.float32) - out[f"{c}__zero"].astype(np.float32)

    return out


def downsample_reset_rows(
    df: pd.DataFrame,
    reset_col: Optional[str],
    keep_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """
    Keep all non-reset rows.
    Randomly keep only keep_ratio of reset rows.
    """
    if reset_col is None or reset_col not in df.columns:
        return df
    if keep_ratio >= 1.0:
        return df

    rng = np.random.default_rng(seed)
    reset_mask = df[reset_col].astype(bool).to_numpy()
    keep = np.ones(len(df), dtype=bool)

    reset_indices = np.where(reset_mask)[0]
    n_keep = max(1, int(round(len(reset_indices) * keep_ratio))) if len(reset_indices) > 0 else 0
    keep_reset_idx = set(rng.choice(reset_indices, size=n_keep, replace=False).tolist()) if n_keep > 0 else set()

    for idx in reset_indices:
        if idx not in keep_reset_idx:
            keep[idx] = False
    return df.iloc[np.where(keep)[0]].reset_index(drop=True)


def make_split_info(
    df: pd.DataFrame,
    split_col: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict:
    """
    Prefer group split by split_col.
    If unique groups < 3, fallback to row split.
    """
    if split_col in df.columns and df[split_col].nunique() >= 3:
        groups = df[split_col].astype(str).unique().tolist()
        rng = np.random.default_rng(seed)
        rng.shuffle(groups)

        n_total = len(groups)
        n_test = max(1, int(round(n_total * test_ratio)))
        n_val = max(1, int(round(n_total * val_ratio)))
        n_test = min(n_test, n_total - 2)
        n_val = min(n_val, n_total - n_test - 1)

        test_groups = groups[:n_test]
        val_groups = groups[n_test:n_test + n_val]
        train_groups = groups[n_test + n_val:]

        return {
            "mode": "group",
            "split_col": split_col,
            "train_groups": train_groups,
            "val_groups": val_groups,
            "test_groups": test_groups,
        }

    # fallback: row split
    row_ids = df["row_id"].tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(row_ids)

    n_total = len(row_ids)
    n_test = max(1, int(round(n_total * test_ratio)))
    n_val = max(1, int(round(n_total * val_ratio)))
    n_test = min(n_test, n_total - 2)
    n_val = min(n_val, n_total - n_test - 1)

    test_rows = row_ids[:n_test]
    val_rows = row_ids[n_test:n_test + n_val]
    train_rows = row_ids[n_test + n_val:]

    return {
        "mode": "row",
        "split_col": None,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
    }


def apply_split_info(df: pd.DataFrame, split_info: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mode = split_info["mode"]
    if mode == "group":
        split_col = split_info["split_col"]
        train_df = df[df[split_col].astype(str).isin(split_info["train_groups"])].copy()
        val_df = df[df[split_col].astype(str).isin(split_info["val_groups"])].copy()
        test_df = df[df[split_col].astype(str).isin(split_info["test_groups"])].copy()
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    if mode == "row":
        train_df = df[df["row_id"].isin(split_info["train_rows"])].copy()
        val_df = df[df["row_id"].isin(split_info["val_rows"])].copy()
        test_df = df[df["row_id"].isin(split_info["test_rows"])].copy()
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    raise ValueError(f"Unknown split mode: {mode}")


class StaticMappingDataset(Dataset):
    def __init__(
        self,
        q_norm: np.ndarray,
        m_norm: np.ndarray,
        active_onehot: Optional[np.ndarray] = None,
    ):
        self.q = torch.tensor(q_norm, dtype=torch.float32)
        self.m = torch.tensor(m_norm, dtype=torch.float32)
        if active_onehot is None:
            active_onehot = np.zeros((len(q_norm), 0), dtype=np.float32)
        self.active = torch.tensor(active_onehot, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.q)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "q": self.q[idx],
            "m": self.m[idx],
            "active": self.active[idx],
        }


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def concat_if_needed(x: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
    if extra is None or extra.numel() == 0:
        return x
    return torch.cat([x, extra], dim=1)


def save_json(path: str, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def init_wandb(args: Any, out_dir: Path, default_run_name: str) -> Optional[Any]:
    if not getattr(args, "wandb", False):
        return None

    try:
        import wandb
    except Exception as exc:
        raise RuntimeError(
            "wandb logging requested, but wandb is not installed. "
            "Install it with: pip install wandb"
        ) from exc

    run_name = getattr(args, "wandb_run_name", "") or default_run_name
    mode = getattr(args, "wandb_mode", "") or None
    run = wandb.init(
        project=getattr(args, "wandb_project", "mapping"),
        name=run_name,
        dir=str(out_dir),
        config=vars(args),
        mode=mode,
    )
    print(f"wandb logging: project={getattr(args, 'wandb_project', 'mapping')} run={run_name}")
    return run


def log_wandb_metrics(wandb_run: Optional[Any], metrics: Dict[str, float], step: int) -> None:
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)


def close_wandb(wandb_run: Optional[Any]) -> None:
    if wandb_run is not None:
        wandb_run.finish()


@torch.no_grad()
def regression_metrics_raw(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    scaler: Standardizer,
) -> Dict[str, np.ndarray]:
    pred = scaler.inverse(pred_norm.detach().cpu().numpy())
    target = scaler.inverse(target_norm.detach().cpu().numpy())
    abs_err = np.abs(pred - target)
    sq_err = (pred - target) ** 2
    mae_per_dim = abs_err.mean(axis=0)
    rmse_per_dim = np.sqrt(sq_err.mean(axis=0))
    return {
        "mae_per_dim": mae_per_dim,
        "rmse_per_dim": rmse_per_dim,
        "mae_mean": float(mae_per_dim.mean()),
        "rmse_mean": float(rmse_per_dim.mean()),
    }
