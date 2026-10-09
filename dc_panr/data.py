from __future__ import annotations

import json
import math
from functools import lru_cache
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import ExperimentConfig
from .utils import ensure_dir, project_root, stable_hash, stable_int


def constellation(modulation: str) -> np.ndarray:
    mod = modulation.upper()
    if mod == "BPSK":
        return np.asarray([-1 + 0j, 1 + 0j], dtype=np.complex64)
    if mod == "QPSK":
        c = np.asarray([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex64)
        return c / np.sqrt(2.0)
    if mod == "8PSK":
        return np.exp(1j * 2.0 * np.pi * np.arange(8) / 8.0).astype(np.complex64)
    if mod in {"16QAM", "QAM16"}:
        vals = [-3, -1, 1, 3]
        c = np.asarray([r + 1j * im for im in vals[::-1] for r in vals], dtype=np.complex64)
        return c / np.sqrt(np.mean(np.abs(c) ** 2))
    raise ValueError(f"Unsupported modulation: {modulation}")


def bits_table(modulation: str) -> np.ndarray:
    """Return the standard Gray labels aligned with ``constellation``.

    QPSK follows the circular order 00, 01, 11, 10. For 16QAM, the
    in-phase and quadrature axes independently use the 4-PAM Gray order
    -3,-1,+1,+3 -> 00,01,11,10. The output bit order is [I bits, Q bits].
    """

    mod = modulation.upper()
    if mod == "BPSK":
        return np.asarray([[0], [1]], dtype=np.uint8)
    if mod == "QPSK":
        return np.asarray([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)
    if mod == "8PSK":
        gray = np.arange(8, dtype=np.uint8) ^ (np.arange(8, dtype=np.uint8) >> 1)
        return np.asarray([[(value >> shift) & 1 for shift in (2, 1, 0)] for value in gray], dtype=np.uint8)
    if mod in {"16QAM", "QAM16"}:
        axis_gray = {
            -3: (0, 0),
            -1: (0, 1),
            1: (1, 1),
            3: (1, 0),
        }
        labels = []
        for q_level in (3, 1, -1, -3):
            for i_level in (-3, -1, 1, 3):
                labels.append(axis_gray[i_level] + axis_gray[q_level])
        return np.asarray(labels, dtype=np.uint8)
    raise ValueError(f"Unsupported modulation: {modulation}")


def sample_doa_set(rng: np.random.Generator, k: int, lo: float, hi: float, min_sep: float) -> np.ndarray:
    if hi - lo < (k - 1) * min_sep:
        raise ValueError("DOA range is too narrow for the requested minimum separation")
    for _ in range(20000):
        x = np.sort(rng.uniform(lo, hi, size=k))
        if np.min(np.diff(x)) >= min_sep:
            return x.astype(np.float32)
    raise RuntimeError("Unable to sample a valid DOA set; reduce min_sep_deg")


def steering_matrix(doa: np.ndarray, num_antennas: int) -> np.ndarray:
    ant = np.arange(num_antennas, dtype=np.float32)[:, None]
    th = np.deg2rad(doa.astype(np.float32))[None, :]
    return (np.exp(-1j * np.pi * ant * np.sin(th)) / np.sqrt(num_antennas)).astype(np.complex64)


def _maximum_cyclic_correlation(candidate: np.ndarray, selected: list[np.ndarray], const: np.ndarray) -> float:
    if not selected:
        return 0.0
    x = const[candidate].astype(np.complex128, copy=False)
    x_norm = max(float(np.vdot(x, x).real), 1e-12)
    x_fft = np.fft.fft(x)
    worst = 0.0
    for previous in selected:
        y = const[previous].astype(np.complex128, copy=False)
        y_norm = max(float(np.vdot(y, y).real), 1e-12)
        cyclic = np.fft.ifft(np.conj(x_fft) * np.fft.fft(y))
        worst = max(worst, float(np.max(np.abs(cyclic)) / math.sqrt(x_norm * y_norm)))
    return worst


@lru_cache(maxsize=64)
def _pilot_master(k: int, modulation: str, master_length: int = 64) -> tuple[tuple[int, ...], ...]:
    """Build deterministic, distinct low-correlation PN-like pilot sequences.

    A fixed master sequence ensures user pilot codes remain comparable across
    changing pilot-prefix lengths without changing the code family.
    """

    const = constellation(modulation)
    q = len(const)
    rng = np.random.default_rng(stable_int("pilot_master", modulation.upper(), k, master_length))
    selected: list[np.ndarray] = []
    candidate_count = 512 if q <= 4 else 256
    for _ in range(k):
        best: np.ndarray | None = None
        best_score = float("inf")
        for _ in range(candidate_count):
            candidate = rng.integers(0, q, size=master_length, endpoint=False, dtype=np.int64)
            if any(np.array_equal(candidate, previous) for previous in selected):
                continue
            score = _maximum_cyclic_correlation(candidate, selected, const)
            if score < best_score:
                best = candidate
                best_score = score
        if best is None:
            raise RuntimeError("Unable to construct distinct pilot sequences")
        selected.append(best)
    return tuple(tuple(int(value) for value in sequence) for sequence in selected)


def make_pilots(k: int, p: int, modulation: str) -> np.ndarray:
    if p <= 0:
        raise ValueError("pilot symbol count must be positive")
    master_length = max(64, p)
    master = np.asarray(_pilot_master(k, modulation.upper(), master_length), dtype=np.int64)
    pilots = master[:, :p].copy()
    if len({tuple(row.tolist()) for row in pilots}) != k:
        raise RuntimeError("Pilot prefixes are not unique; increase pilot length")
    return pilots


def num_symbols(cfg: ExperimentConfig) -> int:
    return max(cfg.pilot_symbols + 8, (cfg.seq_len - (cfg.sps - 1) - cfg.sps // 2) // cfg.sps)


def generate_dataset_arrays(
    cfg: ExperimentConfig,
    n: int,
    seed: int,
    min_sep_deg: float,
    power_spread_db: float,
) -> dict[str, np.ndarray]:
    """Generate asynchronous co-channel frames with one H per sample.

    Symbols are canonical constellation points. Random common phase and source
    power are part of the channel. The direction-conditioned neural receiver receives only the
    target DOA; the deterministic pilots allow it to learn residual phase/gain
    calibration from the observed frame without being given target CSI.
    """

    rng = np.random.default_rng(seed)
    k, m, length, sps = cfg.num_sources, cfg.num_antennas, cfg.seq_len, cfg.sps
    const = constellation(cfg.modulation)
    q = len(const)
    nsym = num_symbols(cfg)
    if cfg.pilot_symbols >= nsym:
        raise ValueError("pilot_symbols must be smaller than the frame symbol count")
    pilot_idx = make_pilots(k, cfg.pilot_symbols, cfg.modulation)

    sources = np.zeros((n, k, length), dtype=np.complex64)
    sym_idx = np.zeros((n, k, nsym), dtype=np.int64)
    offsets = rng.integers(0, sps, size=(n, k), endpoint=False, dtype=np.int64)
    positions = np.zeros((n, k, nsym), dtype=np.int64)
    doa = np.zeros((n, k), dtype=np.float32)
    gains = np.zeros((n, k), dtype=np.complex64)
    h = np.zeros((n, m, k), dtype=np.complex64)
    p_db = np.zeros((n, k), dtype=np.float32)
    channel_phase = np.zeros((n, k), dtype=np.float32)

    time_index = np.arange(length, dtype=np.int64)
    for i in range(n):
        doa[i] = sample_doa_set(rng, k, cfg.doa_min, cfg.doa_max, min_sep_deg)
        if power_spread_db > 0:
            p = rng.uniform(-power_spread_db / 2.0, power_spread_db / 2.0, size=k).astype(np.float32)
            p -= np.mean(p)
        else:
            p = np.zeros(k, dtype=np.float32)
        phase = (
            rng.uniform(-np.pi, np.pi, size=k).astype(np.float32)
            if cfg.random_channel_phase
            else np.zeros(k, dtype=np.float32)
        )
        p_db[i] = p
        channel_phase[i] = phase
        gains[i] = (10.0 ** (p / 20.0) * np.exp(1j * phase)).astype(np.complex64)
        h[i] = steering_matrix(doa[i], m) * gains[i][None, :]

        for user in range(k):
            payload = rng.integers(0, q, size=nsym - cfg.pilot_symbols, endpoint=False, dtype=np.int64)
            idx = np.concatenate([pilot_idx[user], payload])
            sym_idx[i, user] = idx
            off = int(offsets[i, user])
            pos = off + cfg.sps // 2 + np.arange(nsym, dtype=np.int64) * sps
            positions[i, user] = pos
            slot = np.floor((time_index - off) / sps).astype(np.int64)
            slot = np.clip(slot, 0, nsym - 1)
            sources[i, user] = const[idx[slot]]

    return {
        "sources_ch": np.stack([sources.real, sources.imag], axis=2).astype(np.float32),
        "symbol_indices": sym_idx,
        "offsets": offsets,
        "symbol_positions": positions,
        "doa": doa,
        "gain_real": gains.real.astype(np.float32),
        "gain_imag": gains.imag.astype(np.float32),
        "channel_phase_rad": channel_phase,
        "h_real": h.real.astype(np.float32),
        "h_imag": h.imag.astype(np.float32),
        "power_db": p_db,
        "pilot_indices": pilot_idx.astype(np.int64),
    }


def dataset_path(
    cfg: ExperimentConfig,
    split: str,
    n: int,
    min_sep_deg: float,
    power_spread_db: float,
    tag: str,
) -> Path:
    meta = {
        "cfg": asdict(cfg),
        "split": split,
        "n": n,
        "min_sep": min_sep_deg,
        "power_spread": power_spread_db,
        "tag": tag,
    }
    root = ensure_dir(project_root() / cfg.data_root / cfg.modulation.lower())
    return root / f"{split}_{tag}_{stable_hash(meta)}.npz"


def make_dataset(
    cfg: ExperimentConfig,
    split: str,
    n: int,
    min_sep_deg: float,
    power_spread_db: float,
    tag: str,
    force: bool = False,
) -> Path:
    path = dataset_path(cfg, split, n, min_sep_deg, power_spread_db, tag)
    if path.exists() and not force:
        return path
    seed = cfg.dataset_seed + stable_int(
        cfg.modulation, split, tag, min_sep_deg, power_spread_db, False, modulo=10_000_000
    )
    arrays = generate_dataset_arrays(cfg, n, seed, min_sep_deg, power_spread_db)
    np.savez_compressed(path, config_json=np.asarray(json.dumps(asdict(cfg))), **arrays)
    return path


class CochannelDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        with np.load(path, allow_pickle=False) as archive:
            self.data = {key: archive[key] for key in archive.files if key != "config_json"}

    def __len__(self) -> int:
        return int(self.data["sources_ch"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        n = len(self)
        return {
            key: torch.from_numpy(value[idx] if key != "pilot_indices" and value.shape[0] == n else value)
            for key, value in self.data.items()
        }
