"""Training the proposed receiver once on simulated asynchronous cochannel frames.

The only optimized neural model is DirectionConditionedReceiver. No external
baselines, ablation variants, oracle-timing evaluation, or sweep code are here.
Simulated channel coefficients and symbol positions are labels for generating
supervised training examples; they are NEVER inputs to neural inference.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import CochannelDataset, make_dataset
from .model import DirectionConditionedReceiver
from .utils import ensure_dir, resolve_device, set_seed, stable_int


def mix_batch(
    sources: torch.Tensor, channel: torch.Tensor,
    snr_db: float | torch.Tensor, *, noise_seed: int | None = None,
) -> torch.Tensor:
    """Return stacked array IQ [B,2M,N] from training-only physical simulation.

    Implements the original array mixing and complex AWGN convention.
    ``channel`` is the simulator's channel matrix and is not fed to the model.
    """
    b = sources.shape[0]
    source_complex = torch.complex(sources[:, :, 0], sources[:, :, 1])
    clean = torch.einsum('bmk,bkl->bml', channel, source_complex)
    if isinstance(snr_db, torch.Tensor):
        snr = snr_db.to(sources.device).float().view(b, 1, 1)
    else:
        snr = torch.full((b, 1, 1), float(snr_db), device=sources.device)
    signal_power = torch.mean(torch.abs(clean) ** 2, dim=(1, 2), keepdim=True)
    noise_power = signal_power / (10.0 ** (snr / 10.0))
    if noise_seed is None:
        noise_real = torch.randn_like(clean.real)
        noise_imag = torch.randn_like(clean.real)
    else:
        generator1 = torch.Generator(device='cpu').manual_seed(int(noise_seed))
        generator2 = torch.Generator(device='cpu').manual_seed(int(noise_seed) + 1)
        noise_real = torch.randn(clean.shape, generator=generator1).to(sources.device)
        noise_imag = torch.randn(clean.shape, generator=generator2).to(sources.device)
    received = clean + torch.sqrt(noise_power / 2.0) * (noise_real + 1j * noise_imag)
    return torch.stack([received.real, received.imag], dim=2).reshape(b, 2 * channel.shape[1], -1).float()


def direction_features(doa_deg: torch.Tensor, target: int) -> torch.Tensor:
    angle = torch.deg2rad(doa_deg[:, target].float())
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)


def extract_all_users(model: DirectionConditionedReceiver, observation: torch.Tensor,
                      doa_deg: torch.Tensor) -> torch.Tensor:
    """All K users via K forward passes of ONE shared trained receiver."""
    return torch.stack([model(observation, direction_features(doa_deg, user))
                        for user in range(doa_deg.shape[1])], dim=1)


def pilot_calibrated_waveform_loss(pred: torch.Tensor, target: torch.Tensor,
                                   pilot_positions: torch.Tensor) -> torch.Tensor:
    """Eq. (pilot-calibrated waveform loss) in the LWC manuscript.

    Supervised training positions are known from simulated training labels.
    The deployment inference pipeline estimates offsets from observed pilots.
    """
    pred_complex = torch.complex(pred[:, :, 0], pred[:, :, 1])
    target_complex = torch.complex(target[:, :, 0], target[:, :, 1])
    pred_p = torch.gather(pred_complex, 2, pilot_positions.long())
    target_p = torch.gather(target_complex, 2, pilot_positions.long())
    numerator = torch.sum(torch.conj(pred_p) * target_p, dim=2)
    denominator = torch.sum(torch.abs(pred_p) ** 2, dim=2).clamp_min(1e-8)
    alpha = numerator / denominator
    return torch.mean(torch.abs(alpha.unsqueeze(-1) * pred_complex - target_complex) ** 2)


def _channel_from_batch(batch: dict, device: torch.device) -> torch.Tensor:
    return torch.complex(batch['h_real'].to(device).float(),
                         batch['h_imag'].to(device).float())


@torch.no_grad()
def _validation_loss(model: DirectionConditionedReceiver, loader: DataLoader,
                     cfg: ExperimentConfig, device: torch.device) -> float:
    model.eval()
    total, num = 0.0, 0
    for batch_index, batch in enumerate(loader):
        sources = batch['sources_ch'].to(device).float()
        channel = _channel_from_batch(batch, device)
        doa_deg = batch['doa'].to(device).float()
        observed = mix_batch(sources, channel, cfg.val_snr_db,
                             noise_seed=stable_int('validation', batch_index))
        pred = extract_all_users(model, observed, doa_deg)
        value = pilot_calibrated_waveform_loss(
            pred, sources, batch['symbol_positions'][:, :, :cfg.pilot_symbols].to(device))
        total += value.item() * sources.shape[0]
        num += sources.shape[0]
    return total / max(num, 1)


def checkpoint_path(cfg: ExperimentConfig, *, quick: bool = False,
                    root: str | Path = 'outputs') -> Path:
    mode = 'SMOKE_ONLY' if quick else 'full'
    return Path(root) / 'checkpoints' / f'dcpanr_{cfg.modulation.lower()}_{mode}_seed{cfg.run_seed}.pt'


def train_once(cfg: ExperimentConfig, *, checkpoint: str | Path | None = None,
               force: bool = False, quick: bool = False) -> tuple[Path, dict]:
    """Train (or load) ONE shared receiver, once; save state and JSON training summary."""
    set_seed(cfg.run_seed)
    device = resolve_device(cfg.device)
    path = Path(checkpoint) if checkpoint is not None else checkpoint_path(cfg, quick=quick)
    ensure_dir(path.parent)

    if path.exists() and not force:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload['config'] != asdict(cfg):
            raise ValueError(f'Existing checkpoint configuration differs: {path}; use --force or a new path')
        return path, {'loaded_existing': True, 'best_epoch': payload['best_epoch'],
                      'best_val_loss': payload['best_val_loss']}

    train_path = make_dataset(cfg, 'train', cfg.num_train, cfg.train_min_sep_deg,
                              cfg.train_power_spread_db, 'main_train')
    val_path = make_dataset(cfg, 'val', cfg.num_val, cfg.test_min_sep_deg,
                            0.0, 'main_val')
    generator = torch.Generator().manual_seed(cfg.run_seed)
    train_loader = DataLoader(CochannelDataset(train_path), batch_size=cfg.batch_size,
                              shuffle=True, generator=generator, num_workers=cfg.num_workers)
    val_loader = DataLoader(CochannelDataset(val_path), batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers)
    model = DirectionConditionedReceiver(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=cfg.lr_plateau_patience, min_lr=cfg.min_lr)
    best, best_epoch, stale = float('inf'), 0, 0
    start = time.time()
    history = []
    print(f'Training one {cfg.modulation} receiver: device={device}, '
          f'epochs<= {cfg.max_epochs}, samples={cfg.num_train}, '
          f'parameters={sum(p.numel() for p in model.parameters()):,}', flush=True)

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total, num = 0.0, 0
        for batch in train_loader:
            sources = batch['sources_ch'].to(device).float()
            channel = _channel_from_batch(batch, device)
            doa = batch['doa'].to(device).float()
            snr = torch.empty(sources.shape[0], device=device).uniform_(
                cfg.train_snr_min, cfg.train_snr_max)
            observation = mix_batch(sources, channel, snr)
            if cfg.train_doa_jitter_std_deg > 0:
                doa_est = doa + torch.randn_like(doa) * cfg.train_doa_jitter_std_deg
            else:
                doa_est = doa
            pred = extract_all_users(model, observation, doa_est)
            loss = pilot_calibrated_waveform_loss(
                pred, sources, batch['symbol_positions'][:, :, :cfg.pilot_symbols].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * sources.shape[0]
            num += sources.shape[0]
        train_value = total / max(1, num)
        val_value = _validation_loss(model, val_loader, cfg, device)
        scheduler.step(val_value)
        learning_rate = optimizer.param_groups[0]['lr']
        history.append({'epoch': epoch, 'train_loss': train_value,
                        'val_loss': val_value, 'learning_rate': learning_rate})
        print(f'epoch {epoch:03d}: train={train_value:.6e} '
              f'val={val_value:.6e} lr={learning_rate:.3g}', flush=True)
        if val_value < best - 1e-6:
            best, best_epoch, stale = val_value, epoch, 0
            torch.save({'model_state': model.state_dict(), 'config': asdict(cfg),
                        'best_epoch': best_epoch, 'best_val_loss': best}, path)
        else:
            stale += 1
        if epoch >= cfg.min_epochs and stale >= cfg.early_stop_patience:
            break

    with open(path.with_suffix('.history.csv'), 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 'learning_rate'])
        writer.writeheader()
        writer.writerows(history)
    summary = {'loaded_existing': False, 'best_epoch': best_epoch,
               'best_val_loss': best, 'elapsed_seconds': time.time() - start,
               'train_frames': cfg.num_train, 'val_frames': cfg.num_val, 'quick': bool(quick)}
    path.with_suffix('.training.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return path, summary
