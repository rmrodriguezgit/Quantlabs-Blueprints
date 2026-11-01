"""Configurable PyTorch sequence models, denoising AE and offline trading DQN."""
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .stats_nodes import PreparedData


@dataclass
class ModelResult:
    prepared: PreparedData
    predicted: np.ndarray
    metrics: dict[str, float]
    history: list[dict[str, float]]
    signals: np.ndarray | None = None
    latent: np.ndarray | None = None
    reconstructed: np.ndarray | None = None


@dataclass
class NetworkSpec:
    input_features: int
    mode: str
    layers: list[dict[str, Any]]
    complete: bool = False


class SelectInput(nn.Module):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -1, :] if self.mode == 'last' else x.flatten(1)


def custom_network(features: int, steps: int, spec: NetworkSpec) -> tuple[nn.Sequential, int]:
    if not spec.complete:
        raise ValueError('Connect Network Output before the trainer.')
    if spec.input_features != features:
        raise ValueError(f'Network expects {spec.input_features} features; preparation provides {features}. Select matching feature columns.')
    width = features if spec.mode == 'last' else features * steps
    modules: list[nn.Module] = [SelectInput(spec.mode)]
    activations: dict[str, type[nn.Module]] = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'GELU': nn.GELU,
                                            'Sigmoid': nn.Sigmoid, 'Swish': nn.SiLU, 'LeakyReLU': nn.LeakyReLU, 'Identity': nn.Identity}
    for layer in spec.layers:
        kind = str(layer.get('layer_type', 'Dense'))
        if kind == 'Dense':
            units = int(layer.get('units', 64))
            if not 1 <= units <= 2048:
                raise ValueError('Dense units must be 1–2048.')
            modules.append(nn.Linear(width, units))
            width = units
            activation = str(layer.get('activation', 'ReLU'))
            if activation not in activations:
                raise ValueError('Unsupported activation.')
            modules.append(activations[activation]())
        elif kind == 'Dropout':
            rate = float(layer.get('rate', .2))
            if not 0 <= rate < 1:
                raise ValueError('Dropout rate must be 0–<1.')
            modules.append(nn.Dropout(rate))
        elif kind == 'BatchNorm':
            modules.append(nn.BatchNorm1d(width))
        elif kind in activations:
            modules.append(activations[kind]())
        else:
            raise ValueError(f'Unsupported layer type: {kind}')
    return nn.Sequential(*modules), width


class SequenceNet(nn.Module):
    def __init__(self, features: int, steps: int, params: dict[str, Any]) -> None:
        super().__init__()
        self.kind = str(params.get('architecture', 'LSTM'))
        spec = params.get('_network')
        if isinstance(spec, NetworkSpec):
            self.kind = 'Custom'
            self.denoise = self.autoencoder = self.dqn = False
            self.conv = nn.Identity()
            self.recurrent = None
            self.encoder, width = custom_network(features, steps, spec)
            self.norm = self.dropout = nn.Identity()
            self.head = nn.Linear(width, 1)
            self.decoder = self.q_head = None
            return
        hidden = int(params.get('hidden', 32))
        layers = int(params.get('layers', 1))
        dropout = float(params.get('dropout', 0.15))
        if not 4 <= hidden <= 256 or not 1 <= layers <= 4 or not 0 <= dropout <= 0.8:
            raise ValueError('Hidden 4–256, layers 1–4, dropout 0–.8 required.')
        self.denoise = self.kind == 'DAE'
        self.autoencoder = self.kind in ('Autoencoder', 'DAE')
        self.dqn = self.kind == 'DQN'
        self.conv: nn.Module = nn.Identity()
        self.recurrent: nn.Module | None = None
        if self.kind in ('CNN', 'CNN-LSTM'):
            kernel = int(params.get('kernel', 3))
            if kernel < 1 or kernel > steps:
                raise ValueError('Convolution kernel must fit the lookback window.')
            self.conv = nn.Sequential(nn.Conv1d(features, hidden, kernel, padding='same'), nn.ReLU())
            features = hidden
        bidirectional = self.kind in ('BiLSTM', 'BiGRU')
        if self.kind in ('LSTM', 'CNN-LSTM', 'BiLSTM', 'GRU', 'BiGRU'):
            cls = nn.GRU if self.kind in ('GRU', 'BiGRU') else nn.LSTM
            self.recurrent = cls(features, hidden, num_layers=layers, batch_first=True,
                                 bidirectional=bidirectional, dropout=dropout if layers > 1 else 0)
            width = hidden * (2 if bidirectional else 1)
            self.encoder = nn.Identity()
        else:
            width = hidden
            blocks: list[nn.Module] = [nn.Flatten(), nn.Linear(features * steps, hidden), nn.ReLU()]
            for _ in range(layers - 1):
                blocks.extend([nn.Linear(hidden, hidden), nn.ReLU()])
            self.encoder = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(width) if not params.get('batch_norm', False) else nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, 1)
        self.decoder = nn.Linear(width, features * steps) if self.autoencoder else None
        self.q_head = nn.Linear(width, 3) if self.dqn else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind in ('CNN', 'CNN-LSTM'):
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        if self.recurrent is not None:
            _, state = self.recurrent(x)
            hidden = state[0] if isinstance(state, tuple) else state
            x = torch.cat([hidden[-2], hidden[-1]], dim=1) if self.recurrent.bidirectional else hidden[-1]
        else:
            x = self.encoder(x)
        return self.dropout(self.norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x)).squeeze(-1)


def train(data: PreparedData, params: dict[str, Any], progress: Callable[[dict[str, Any]], None] | None = None) -> ModelResult:
    architecture = str(params.get('architecture', 'LSTM'))
    if architecture == 'Custom' and not isinstance(params.get('_network'), NetworkSpec):
        raise ValueError('Custom architecture requires a connected Network Input → Layers → Network Output chain.')
    if architecture not in ('Custom', 'Dense', 'CNN', 'LSTM', 'GRU', 'BiLSTM', 'BiGRU', 'CNN-LSTM', 'Autoencoder', 'DAE', 'DQN'):
        raise ValueError('Unsupported architecture.')
    if architecture == 'DQN' and not isinstance(params.get('_network'), NetworkSpec) and data.target != 'LogReturn':
        raise ValueError('DQN requires LogReturn target.')
    torch.manual_seed(int(params.get('seed', 42)))
    epochs, batch = int(params.get('epochs', 12)), int(params.get('batch_size', 32))
    patience = int(params.get('patience', 5))
    lr, l1, l2 = (float(params.get(k, v)) for k, v in [('lr', .001), ('l1', 0), ('l2', .0001)])
    if not 1 <= epochs <= 300 or not 2 <= batch <= 1024 or not 1 <= patience <= 100 or not 0 < lr <= 1 or min(l1, l2) < 0:
        raise ValueError('Invalid training hyperparameters.')
    model = SequenceNet(data.x_train.shape[2], data.x_train.shape[1], params)
    optimizer_name = str(params.get('optimizer', 'Adam'))
    if optimizer_name not in ('Adam', 'SGD'):
        raise ValueError('Optimizer must be Adam or SGD.')
    optimizer_cls = torch.optim.Adam if optimizer_name == 'Adam' else torch.optim.SGD
    optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=l2)
    x = torch.from_numpy(data.x_train)
    xv = torch.from_numpy(data.x_val)
    mean, std = float(data.y_train.mean()), max(float(data.y_train.std()), 1e-8)
    y = torch.from_numpy((data.y_train - mean) / std)
    yv = torch.from_numpy((data.y_val - mean) / std)
    criterion = nn.MSELoss()
    best, stale, history = float('inf'), 0, []
    checkpoint = deepcopy(model.state_dict())
    target_model = deepcopy(model).eval()
    actions = torch.tensor([-1., 0., 1.])
    noise = float(params.get('noise', .1))
    if not 0 <= noise <= 1:
        raise ValueError('Denoising noise must be 0–1.')
    for epoch in range(epochs):
        model.train()
        total, seen = 0., 0
        for indices in torch.randperm(len(x)).split(batch):
            if len(indices) < 2:  # BatchNorm requires at least two samples.
                continue
            xb, yb = x[indices], y[indices]
            noisy = xb + noise * torch.randn_like(xb) if model.denoise else xb
            encoded = model.encode(noisy)
            loss = criterion(model.head(encoded).squeeze(-1), yb)
            if model.decoder is not None:
                loss = loss + criterion(model.decoder(encoded), xb.flatten(1))
            if model.q_head is not None and target_model.q_head is not None:
                # Offline replay of all actions; market transitions are action independent.
                next_indices = torch.clamp(indices + 1, max=len(x) - 1)
                with torch.no_grad():
                    next_q = target_model.q_head(target_model.encode(x[next_indices])).max(1).values
                    terminal = (indices == len(x) - 1).float()
                    reward = torch.from_numpy(data.y_train[indices.numpy()])[:, None] * actions[None, :]
                    q_target = reward + .95 * next_q[:, None] * (1 - terminal[:, None])
                loss = loss + criterion(model.q_head(encoded), q_target)
            loss = loss + l1 * sum(p.abs().sum() for p in model.parameters())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            seen += len(indices)
        model.eval()
        with torch.no_grad():
            encoded_val = model.encode(xv)
            validation = criterion(model.head(encoded_val).squeeze(-1), yv)
            if model.decoder is not None:
                validation += criterion(model.decoder(encoded_val), xv.flatten(1))
            if model.q_head is not None:
                rewards = torch.from_numpy(data.y_val)[:, None] * actions[None, :]
                next_states = torch.cat([xv[1:], xv[-1:]])
                next_values = model.q_head(model.encode(next_states)).max(1).values
                next_values[-1] = 0
                validation += criterion(model.q_head(encoded_val), rewards + .95 * next_values[:, None])
            val_loss = float(validation)
        history.append({'epoch': epoch + 1, 'train': total / max(seen, 1), 'validation': val_loss})
        if progress and np.isfinite(val_loss):
            progress({**history[-1], 'epochs': epochs})
        target_model.load_state_dict(model.state_dict())
        if not np.isfinite(val_loss):
            raise ValueError('Training diverged; reduce learning rate.')
        if val_loss < best:
            best, stale, checkpoint = val_loss, 0, deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(checkpoint)
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(data.x_test)
        encoded_test = model.encode(xt)
        prediction = model.head(encoded_test).squeeze(-1).numpy() * std + mean
        latent = encoded_test.numpy() if model.autoencoder else None
        reconstructed = model.decoder(encoded_test).reshape(xt.shape).numpy() if model.decoder is not None else None
        signals = model.q_head(model.encode(xt)).argmax(1).numpy() - 1 if model.q_head is not None else None
    mse = float(mean_squared_error(data.y_test, prediction))
    metrics = {'MSE': mse, 'RMSE': float(np.sqrt(mse)), 'MAE': float(mean_absolute_error(data.y_test, prediction)),
               'R2': float(r2_score(data.y_test, prediction))}
    return ModelResult(data, prediction, metrics, history, signals, latent, reconstructed)


def describe_network(model: nn.Module, example: torch.Tensor) -> dict[str, Any]:
    """Report actual executed leaf layers without changing weights, BN stats or RNG."""
    layers: list[dict[str, Any]] = []
    hooks = []
    training = model.training
    def capture(name: str, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        layers.append({'name': name, 'operation': 'Swish' if isinstance(module, nn.SiLU) else type(module).__name__,
                       'input_shape': list(inputs[0].shape[1:]), 'output_shape': list(output.shape[1:]),
                       'parameters': sum(p.numel() for p in module.parameters(recurse=False))})
    for name, module in model.named_modules():
        if name and not list(module.children()):
            hooks.append(module.register_forward_hook(lambda m, x, y, name=name: capture(name, m, x, y)))
    try:
        model.eval()
        with torch.no_grad(): model(example)
    finally:
        for hook in hooks: hook.remove()
        model.train(training)
    return {'layers': layers, 'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
