"""Pilot-correlation synchronization and pilot-LS corrected hard demodulation.

Inference uses ONLY array observations, known DOAs, known pilot indices,
and trained neural network parameters. No oracle delays or channel matrices.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ExperimentConfig
from .data import bits_table, constellation, num_symbols, steering_matrix
from .model import DirectionConditionedReceiver
from .utils import resolve_device

def _pilot_waveforms(pilot_indices: np.ndarray, modulation: str) -> np.ndarray:
    return constellation(modulation)[np.asarray(pilot_indices, dtype=np.int64)]


def estimate_user_offsets(
    received: np.ndarray,
    doa_deg: np.ndarray,
    pilot_indices: np.ndarray,
    modulation: str,
    sps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate one integer sample offset per user using only DOA, pilots and received samples.

    A DOA steering projection is used only to form a target-biased scalar stream; no true
    channel coefficient, simulated delay or transmitted payload is used by this estimator.
    """
    y = np.asarray(received)
    doa = np.asarray(doa_deg, dtype=float)
    pilots = _pilot_waveforms(pilot_indices, modulation)
    m, length = y.shape
    k, pcount = pilots.shape
    offsets = np.zeros(k, dtype=np.int64)
    scores = np.zeros((k, sps), dtype=np.float64)
    for user in range(k):
        a = steering_matrix(np.asarray([doa[user]], dtype=np.float32), m)[:, 0].astype(np.complex128)
        beam = np.conj(a) @ y
        pilot = pilots[user].astype(np.complex128)
        # Rectangular pulse shaping has no unique within-symbol peak. Timing is therefore
        # recovered from the pilot transition pattern: compare the complete oversampled
        # pilot waveform (each pilot symbol repeated sps times) for every candidate offset.
        template = np.repeat(pilot, sps)
        for q in range(sps):
            pos = q + np.arange(len(template), dtype=np.int64)
            valid = pos < length
            if np.count_nonzero(valid) < max(2 * sps, min(len(template), k * sps)):
                scores[user, q] = -np.inf
                continue
            z = beam[pos[valid]]
            pv = template[valid]
            denom = max(float(np.vdot(z, z).real) * float(np.vdot(pv, pv).real), 1e-12)
            scores[user, q] = float(abs(np.vdot(pv, z)) ** 2 / denom)
        offsets[user] = int(np.argmax(scores[user]))
    return offsets, scores



def estimate_user_offsets_pulse_shaped(
    received: np.ndarray,
    doa_deg: np.ndarray,
    pilot_indices: np.ndarray,
    modulation: str,
    sps: int,
    beta: float = 0.35,
    span_symbols: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Pilot-correlation timing for RRC-Tx/RRC-Rx matched-filter-equivalent data.

    Candidate offsets are scored using only received samples, target DOA, known
    pilot symbols and the known RRC pulse specification. Neither simulated
    offsets nor payload symbols are used. Multipath taps are deliberately not
    supplied to the timing estimator.
    """
    from .rrc import rrc_taps

    y = np.asarray(received)
    doa = np.asarray(doa_deg, dtype=float)
    pilots = _pilot_waveforms(pilot_indices, modulation)
    m, length = y.shape
    k, pcount = pilots.shape
    rrc = rrc_taps(beta, span_symbols, sps)
    pulse = np.convolve(rrc, rrc)
    pulse = pulse / max(abs(pulse[len(pulse) // 2]), 1e-12)
    offsets = np.zeros(k, dtype=np.int64)
    scores = np.full((k, sps), -np.inf, dtype=np.float64)
    margin = int(span_symbols * sps)
    for user in range(k):
        a = steering_matrix(np.asarray([doa[user]], dtype=np.float32), m)[:, 0].astype(np.complex128)
        beam = np.conj(a) @ y
        pilot = pilots[user].astype(np.complex128)
        for q in range(sps):
            impulse = np.zeros(length, dtype=np.complex128)
            pos = q + sps // 2 + np.arange(pcount, dtype=np.int64) * sps
            valid = (pos >= 0) & (pos < length)
            impulse[pos[valid]] = pilot[valid]
            template = np.convolve(impulse, pulse, mode="same")
            lo = max(0, int(q + sps // 2 - margin))
            hi = min(length, int(q + sps // 2 + pcount * sps + margin))
            z = beam[lo:hi]
            pv = template[lo:hi]
            denom = max(float(np.vdot(z, z).real) * float(np.vdot(pv, pv).real), 1e-12)
            scores[user, q] = float(abs(np.vdot(pv, z)) ** 2 / denom)
        offsets[user] = int(np.argmax(scores[user]))
    return offsets, scores

def symbol_positions_from_offsets(offsets: np.ndarray, nsym: int, sps: int) -> np.ndarray:
    off = np.asarray(offsets, dtype=np.int64)
    return off[:, None] + sps // 2 + np.arange(nsym, dtype=np.int64)[None, :] * sps



def load_receiver(checkpoint: str | Path, device: str = 'auto') -> tuple[DirectionConditionedReceiver, ExperimentConfig]:
    """Load the saved proposed model and its exactly matching architecture configuration.

    Only load checkpoints from sources you trust.
    """
    payload = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
    cfg = ExperimentConfig(**payload['config'])
    receiver = DirectionConditionedReceiver(cfg)
    receiver.load_state_dict(payload['model_state'], strict=True)
    receiver = receiver.to(resolve_device(device)).eval()
    return receiver, cfg


@torch.inference_mode()
def infer_frame(
    receiver: DirectionConditionedReceiver,
    cfg: ExperimentConfig,
    iq: np.ndarray,
    doa_deg: np.ndarray,
    pilot_indices: np.ndarray,
    *,
    timing: str = 'rectangular',
    rrc_beta: float = 0.35,
    rrc_span_symbols: int = 8,
) -> dict[str, np.ndarray]:
    """Extract K cochannel user waveforms, estimate timing and demodulate one frame.

    Parameters
    ----------
    iq:
        Complex64/complex128 array shaped [M, N], antenna rows and time columns.
        Use the same receiver-array ordering and normalization as training.
    doa_deg:
        Desired user DOAs in degrees [K], supplied by an external estimator.
        DOA order must equal the order of ``pilot_indices``.
    pilot_indices:
        Known indices [K, P] into ``constellation(cfg.modulation)``; the
        transmit pilots are the first P symbols of every user frame.
    timing:
        'rectangular' (paper primary model) or 'rrc' (known RRC pulse timing).
        An RRC-trained model is required for dependable RRC data performance.

    Returns
    -------
    Dictionaries of numpy arrays, including estimated offsets, frame waveforms,
    corrected symbols, hard payload symbols and Gray-coded bits. NEVER uses
    the transmitted payload, simulated offsets, or true complex channel.
    """
    y = np.asarray(iq)
    angles = np.asarray(doa_deg, dtype=np.float32)
    pilots = np.asarray(pilot_indices)
    if y.ndim != 2 or y.shape != (cfg.num_antennas, cfg.seq_len):
        raise ValueError(f'Expected iq shape [M,N]={cfg.num_antennas,cfg.seq_len}, got {y.shape}')
    if not np.iscomplexobj(y) or not np.isfinite(y).all():
        raise ValueError('iq must contain finite complex-valued observations')
    if angles.shape != (cfg.num_sources,) or not np.isfinite(angles).all():
        raise ValueError(f'doa_deg must have shape ({cfg.num_sources},) and be finite')
    if pilots.shape != (cfg.num_sources, cfg.pilot_symbols):
        raise ValueError(f'pilot_indices must have shape ({cfg.num_sources},{cfg.pilot_symbols})')
    constellation_values = constellation(cfg.modulation)
    if not np.issubdtype(pilots.dtype, np.integer):
        raise ValueError('pilot_indices must be integer constellation indices')
    if np.any((pilots < 0) | (pilots >= len(constellation_values))):
        raise ValueError('pilot_indices out of constellation range')
    if timing == 'rectangular':
        estimated_offsets, timing_scores = estimate_user_offsets(
            y, angles, pilots, cfg.modulation, cfg.sps)
    elif timing == 'rrc':
        estimated_offsets, timing_scores = estimate_user_offsets_pulse_shaped(
            y, angles, pilots, cfg.modulation, cfg.sps,
            beta=rrc_beta, span_symbols=rrc_span_symbols)
    else:
        raise ValueError("timing must be 'rectangular' or 'rrc'")

    device = next(receiver.parameters()).device
    iq32 = np.asarray(y, dtype=np.complex64)
    observation = np.stack([iq32.real, iq32.imag], axis=1).reshape(
        1, 2 * cfg.num_antennas, cfg.seq_len)
    observation_tensor = torch.from_numpy(observation.copy()).to(device)
    waveforms = []
    for angle in angles:
        rad = np.deg2rad(float(angle))
        direction = torch.tensor([[np.sin(rad), np.cos(rad)]],
                                 device=device, dtype=torch.float32)
        predicted = receiver(observation_tensor, direction).squeeze(0).cpu().numpy()
        waveforms.append((predicted[0] + 1j * predicted[1]).astype(np.complex64))
    waveforms = np.stack(waveforms, axis=0)

    nsym = num_symbols(cfg)
    positions = symbol_positions_from_offsets(estimated_offsets, nsym, cfg.sps)
    if not np.all((positions >= 0) & (positions < cfg.seq_len)):
        raise RuntimeError('Estimated sampling positions exceed frame bounds')
    sampled = np.take_along_axis(waveforms, positions, axis=1)
    calibrated = np.empty_like(sampled)
    gains = np.empty(cfg.num_sources, dtype=np.complex64)
    payload_ids = np.zeros((cfg.num_sources, nsym - cfg.pilot_symbols), dtype=np.int64)
    labels = bits_table(cfg.modulation)
    for user in range(cfg.num_sources):
        pred_pilots = sampled[user, :cfg.pilot_symbols]
        known_pilots = constellation_values[pilots[user]]
        denominator = np.vdot(pred_pilots, pred_pilots)
        gain = np.vdot(pred_pilots, known_pilots) / denominator if abs(denominator) > 1e-12 else (1.0 + 0j)
        gains[user] = gain
        calibrated[user] = gain * sampled[user]
        payload = calibrated[user, cfg.pilot_symbols:]
        payload_ids[user] = np.argmin(np.abs(payload[:, None] - constellation_values[None, :]) ** 2, axis=1)
    return {
        'estimated_offsets': estimated_offsets.astype(np.int64),
        'timing_scores': timing_scores,
        'raw_waveforms': waveforms,
        'symbol_positions': positions,
        'estimated_complex_gain': gains,
        'calibrated_symbols': calibrated,
        'payload_symbol_indices': payload_ids,
        'payload_constellation_symbols': constellation_values[payload_ids],
        'payload_bits': labels[payload_ids],
    }
