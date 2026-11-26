"""Herramientas de la Unidad 8 · Autoencoders en finanzas cuantitativas.

Cubre lo que la infografía deja como "bucle completo": validación,
transferencias CPU/GPU (Colab T4), controles numéricos, reproducibilidad,
persistencia y evaluación. Cada función indica el segmento (01–10) al que sirve.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

FEATURES = [f"F{i}" for i in range(1, 12)]
TARGET = "y_bps"


# ── Entorno: reproducibilidad y dispositivo ───────────────────────────────────
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer: str | None = None) -> torch.device:
    """cuda (Colab T4) > mps (Apple) > cpu. `prefer` o env AE_DEVICE fuerzan uno."""
    prefer = prefer or os.environ.get("AE_DEVICE")
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda · {torch.cuda.get_device_name(device)}"
    return device.type


def to_tensor(x, device: torch.device) -> torch.Tensor:
    """Transferencia host→dispositivo en float32 contiguo."""
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    return torch.as_tensor(np.ascontiguousarray(x, dtype=np.float32), device=device)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    """Transferencia dispositivo→host."""
    return t.detach().cpu().numpy()


# ── Datos y validación de entradas ────────────────────────────────────────────
def make_synthetic_data(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """11 indicadores con 3 factores latentes + no linealidad; y_bps en pb."""
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(n, 3))
    W = rng.normal(size=(3, len(FEATURES)))
    X = f @ W + 0.3 * np.tanh(f[:, [0]] * f[:, [1]]) + 0.2 * rng.normal(size=(n, len(FEATURES)))
    df = pd.DataFrame(X, columns=FEATURES)
    df.index = pd.bdate_range("2010-01-01", periods=n, name="date")
    df[TARGET] = 25 * f[:, 0] - 10 * f[:, 2] + 5 * rng.normal(size=n)
    return df


def load_data(path: str | Path) -> pd.DataFrame:
    """CSV/Parquet con columna de fecha opcional ('date') + FEATURES + y_bps."""
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "date" in df.columns:
        df = df.assign(date=pd.to_datetime(df["date"])).set_index("date").sort_index()
    return df


def validate_frame(df: pd.DataFrame, features=FEATURES, target: str | None = TARGET) -> None:
    """Controles de entrada: columnas, tipos, finitos, varianza y orden temporal."""
    cols = list(features) + ([target] if target else [])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")
    vals = df[cols].apply(pd.to_numeric, errors="coerce")
    bad = vals.columns[~np.isfinite(vals.to_numpy()).all(axis=0)].tolist()
    if bad:
        raise ValueError(f"Valores NaN/inf o no numéricos en: {bad}")
    const = [c for c in features if vals[c].std() == 0]
    if const:
        raise ValueError(f"Columnas constantes (no estandarizables): {const}")
    if isinstance(df.index, pd.DatetimeIndex) and not df.index.is_monotonic_increasing:
        raise ValueError("El índice temporal no está ordenado")


def temporal_split(df: pd.DataFrame, train: float = .6, validation: float = .2, gap: int = 0):
    """División cronológica sin barajar; `gap` filas de embargo entre bloques."""
    if not 0 < train < 1 or not 0 < validation < 1 or train + validation >= 1:
        raise ValueError("Proporciones inválidas")
    n = len(df)
    a, b = int(train * n), int((train + validation) * n)
    return df.iloc[:a], df.iloc[a + gap:b], df.iloc[b + gap:]


# ── 01 Preparar entradas comparables ──────────────────────────────────────────
def standardize(tr, va, te, features=FEATURES):
    """Ajusta el escalador SOLO con train y transforma los tres bloques."""
    x_scaler = StandardScaler().fit(tr[features])
    X = {k: x_scaler.transform(d[features]).astype(np.float32)
         for k, d in [("train", tr), ("validation", va), ("test", te)]}
    return x_scaler, X


# ── 02 Referencia PCA ─────────────────────────────────────────────────────────
def fit_pca(X_train, k: int = 3) -> PCA:
    return PCA(n_components=k).fit(X_train)


# ── 03–05 Modelo ──────────────────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, d: int, k: int = 3, h: int = 32):
        super().__init__()
        self.d, self.k, self.h = d, k, h
        self.encoder = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, k))   # 03
        self.decoder = nn.Sequential(nn.Linear(k, h), nn.ReLU(), nn.Linear(h, d))   # 04

    def forward(self, x):                                                          # 05
        return self.decoder(self.encoder(x))


# ── 06 Ruido ──────────────────────────────────────────────────────────────────
def add_noise(xb: torch.Tensor, sigma: float) -> torch.Tensor:
    return xb + sigma * torch.randn_like(xb) if sigma > 0 else xb


# ── 07 Entrenamiento con validación y controles numéricos ─────────────────────
@dataclass
class TrainConfig:
    epochs: int = 300
    batch: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 25
    min_delta: float = 1e-6
    grad_clip: float = 1.0
    noise: float = 0.0
    seed: int = 42


@dataclass
class History:
    train: list = field(default_factory=list)
    validation: list = field(default_factory=list)
    best_epoch: int = -1
    best_validation: float = float("inf")


class NumericalError(RuntimeError):
    pass


def check_finite(name: str, value) -> None:
    v = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if not torch.isfinite(v).all():
        raise NumericalError(f"{name} contiene NaN/inf: revisa escala, lr o datos")


def fit(model: nn.Module, X_in, X_target, X_val_in, X_val_target,
        cfg: TrainConfig | None = None, device: torch.device | None = None, verbose=False,
        reinit: bool = True):
    """Entrena AE (noise=0) o DAE (noise>0) con early stopping por MSE de validación.

    Controles: entradas finitas, pérdida y gradientes finitos, clipping de norma,
    y restauración del mejor estado. Devuelve (modelo en CPU, History).
    """
    cfg = cfg or TrainConfig()
    device = device or get_device()
    set_seed(cfg.seed)
    if reinit:  # pesos iniciales dependientes solo de cfg.seed → reproducible
        for mod in model.modules():
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()
    xi, xt = to_tensor(X_in, device), to_tensor(X_target, device)
    vi, vt = to_tensor(X_val_in, device), to_tensor(X_val_target, device)
    for n, t in [("X_train", xi), ("X_target", xt), ("X_val", vi), ("X_val_target", vt)]:
        check_finite(n, t)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()
    hist, best_state, wait = History(), None, 0
    gen = torch.Generator().manual_seed(cfg.seed)
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for idx in torch.randperm(len(xi), generator=gen).split(cfg.batch):
            idx = idx.to(device)
            xb, yb = xi[idx], xt[idx]
            pred = model(add_noise(xb, cfg.noise))                 # 06
            loss = loss_fn(pred, yb)
            check_finite("loss", loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip or float("inf"))
            check_finite("grad_norm", gnorm)
            opt.step()
            running += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(vi), vt).item()
        check_finite("validation_loss", torch.tensor(v))
        hist.train.append(running / len(xi))
        hist.validation.append(v)
        if verbose and epoch % 20 == 0:
            print(f"epoch {epoch:3d} · train {hist.train[-1]:.4f} · val {v:.4f}")
        if v < hist.best_validation - cfg.min_delta:
            hist.best_validation, hist.best_epoch, wait = v, epoch, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model.cpu().eval(), hist


# ── Inferencia ────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer(model: nn.Module, x, device: torch.device | None = None, batch: int = 4096):
    """Reconstrucción por lotes; devuelve numpy en CPU."""
    device = device or torch.device("cpu")
    model = model.to(device).eval()
    out = [to_numpy(model(to_tensor(x[i:i + batch], device))) for i in range(0, len(x), batch)]
    model.cpu()
    return np.concatenate(out)


@torch.no_grad()
def encode_ae(model: Autoencoder, x) -> np.ndarray:
    model.eval()
    return to_numpy(model.encoder(to_tensor(x, torch.device("cpu"))))


class Compressor:
    """Interfaz común PCA / AE / DAE: encode() y reconstruct()."""

    def __init__(self, name: str, model):
        self.name, self.model = name, model

    def encode(self, x):
        return self.model.transform(x) if isinstance(self.model, PCA) else encode_ae(self.model, x)

    def reconstruct(self, x):
        if isinstance(self.model, PCA):
            return self.model.inverse_transform(self.model.transform(x)).astype(np.float32)
        return infer(self.model, x)


# ── 08 Selección por reconstrucción ───────────────────────────────────────────
def reconstruction_mse(recon, x) -> float:
    return float(np.mean((np.asarray(recon) - np.asarray(x)) ** 2))


def per_feature_mse(recon, x, features=FEATURES) -> pd.Series:
    return pd.Series(np.mean((recon - x) ** 2, axis=0), index=features, name="MSE")


def compare_models(compressors: dict[str, Compressor], X_val) -> pd.DataFrame:
    rows = [{"modelo": n, "validation_reconstruction_MSE": reconstruction_mse(c.reconstruct(X_val), X_val)}
            for n, c in compressors.items()]
    return pd.DataFrame(rows).sort_values("validation_reconstruction_MSE").reset_index(drop=True)


def select_best(comparison: pd.DataFrame) -> str:
    return comparison.sort_values("validation_reconstruction_MSE").iloc[0].modelo


# ── 09 Utilidad posterior con Ridge ───────────────────────────────────────────
def fit_downstream(compressors: dict[str, Compressor], X_train, y_train, alpha: float = 1.):
    return {n: Ridge(alpha=alpha).fit(c.encode(X_train), y_train) for n, c in compressors.items()}


# ── 10 Medición en test ───────────────────────────────────────────────────────
def evaluate_test(compressors, downstream, X_test, y_test) -> pd.DataFrame:
    rows = []
    for n, c in compressors.items():
        y_hat = downstream[n].predict(c.encode(X_test))
        rows.append({"modelo": n,
                     "test_reconstruction_MSE": reconstruction_mse(c.reconstruct(X_test), X_test),
                     "test_y_bps_RMSE": mean_squared_error(y_test, y_hat) ** .5,
                     "test_y_bps_R2": r2_score(y_test, y_hat)})
    return pd.DataFrame(rows)


# ── Persistencia ──────────────────────────────────────────────────────────────
def save_model(model: Autoencoder, path: str | Path, cfg: TrainConfig | None = None, **meta) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "arch": {"d": model.d, "k": model.k, "h": model.h},
                "config": asdict(cfg) if cfg else None, "meta": meta}, path)


def load_model(path: str | Path) -> Autoencoder:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = Autoencoder(**ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def save_report(path: str | Path, **tables) -> None:
    out = {k: (v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else v) for k, v in tables.items()}
    Path(path).write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
