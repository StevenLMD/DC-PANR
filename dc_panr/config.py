"""Reproducible single-run settings for the proposed LWC receiver.

The default architecture and training hyperparameters match the accompanying
article. ``small_smoke_config`` intentionally reduces compute for plumbing tests
and cannot reproduce the article's performance numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ExperimentConfig:
    modulation: str = 'QPSK'
    num_sources: int = 3
    num_antennas: int = 3
    seq_len: int = 1024
    sps: int = 8
    pilot_symbols: int = 20
    num_train: int = 16000
    num_val: int = 2400
    dataset_seed: int = 20260710
    run_seed: int = 20260711
    doa_min: float = -60.0
    doa_max: float = 60.0
    train_min_sep_deg: float = 5.0
    test_min_sep_deg: float = 10.0
    train_power_spread_db: float = 6.0
    train_doa_jitter_std_deg: float = 1.0
    random_channel_phase: bool = True
    refine_max_gain: float = 0.12
    refine_max_phase: float = 0.15
    max_epochs: int = 100
    min_epochs: int = 30
    early_stop_patience: int = 12
    lr_plateau_patience: int = 4
    batch_size: int = 32
    base_ch: int = 48
    weight_hidden: int = 64
    lr: float = 8e-4
    min_lr: float = 1e-5
    weight_decay: float = 1e-4
    train_snr_min: float = -10.0
    train_snr_max: float = 12.0
    val_snr_db: float = 0.0
    num_workers: int = 0
    device: str = 'auto'
    data_root: str = 'outputs/data_cache'


def config_for_modulation(modulation: str) -> ExperimentConfig:
    name = modulation.upper()
    cfg = ExperimentConfig(modulation=name)
    if name in {'16QAM', 'QAM16'}:
        return replace(cfg, modulation='16QAM', train_snr_min=0.0,
                       train_snr_max=24.0, val_snr_db=12.0)
    if name != 'QPSK':
        raise ValueError('The paper single-run training presets support QPSK or 16QAM')
    return cfg


def small_smoke_config(cfg: ExperimentConfig) -> ExperimentConfig:
    """Small fast test ONLY; reduced width, frame size and dataset."""
    return replace(cfg, seq_len=128, sps=4, pilot_symbols=6,
                   num_train=32, num_val=16, batch_size=4,
                   base_ch=8, weight_hidden=16, max_epochs=1,
                   min_epochs=1, early_stop_patience=1, lr_plateau_patience=1)
