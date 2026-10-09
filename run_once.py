"""One command: train/load ONE proposed receiver and infer ONE example frame."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from dc_panr.config import config_for_modulation, small_smoke_config
from dc_panr.data import generate_dataset_arrays
from dc_panr.inference import infer_frame, load_receiver
from dc_panr.training import mix_batch, train_once
from dc_panr.utils import ensure_dir, set_seed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quick', action='store_true', help='SMOKE TEST ONLY: tiny non-paper settings')
    p.add_argument('--modulation', choices=['QPSK', '16QAM'], default='QPSK')
    p.add_argument('--snr-db', type=float, default=0.0, help='Single simulated inference frame SNR')
    p.add_argument('--force', action='store_true', help='Retrain even if matching checkpoint exists')
    p.add_argument('--checkpoint', type=Path, default=None, help='Optional checkpoint location')
    p.add_argument('--out', type=Path, default=None)
    args = p.parse_args()
    cfg = config_for_modulation(args.modulation)
    if args.quick:
        cfg = small_smoke_config(cfg)
        print('*** QUICK SMOKE TEST: shortened sequence/smaller network; NOT paper results ***', flush=True)
    root = args.out or Path('outputs') / ('quick_smoke' if args.quick else f'one_run_{cfg.modulation.lower()}')
    ensure_dir(root)
    path, summary = train_once(cfg, checkpoint=args.checkpoint, quick=args.quick, force=args.force)
    receiver, saved_cfg = load_receiver(path, cfg.device)
    set_seed(cfg.run_seed + 101)
    arr = generate_dataset_arrays(cfg, 1, cfg.dataset_seed + 909,
                                  cfg.test_min_sep_deg, cfg.train_power_spread_db)
    source = torch.from_numpy(arr['sources_ch'])
    channel = torch.complex(torch.from_numpy(arr['h_real']), torch.from_numpy(arr['h_imag']))
    iq_channels = mix_batch(source, channel, args.snr_db, noise_seed=cfg.run_seed + 171)
    stacked = iq_channels[0].numpy().reshape(cfg.num_antennas, 2, cfg.seq_len)
    iq = stacked[:, 0] + 1j * stacked[:, 1]
    doa = arr['doa'][0]
    pilots = arr['pilot_indices']
    np.savez_compressed(root / 'example_input.npz', iq=iq, doa_deg=doa, pilot_indices=pilots)
    result = infer_frame(receiver, saved_cfg, iq, doa, pilots)
    np.savez_compressed(root / 'one_frame_output.npz', **result)
    report = {
        'status': 'completed', 'mode': 'SMOKE_NOT_FOR_PAPER' if args.quick else 'single_run',
        'method': 'DC-PANR', 'modulation': cfg.modulation,
        'snr_db': args.snr_db, 'checkpoint': str(path),
        'model_parameters': sum(x.numel() for x in receiver.parameters()),
        'num_users': cfg.num_sources, 'estimated_offsets': result['estimated_offsets'].tolist(),
        'payload_symbols_per_user': int(result['payload_symbol_indices'].shape[1]),
        'training': summary,
    }
    (root / 'run_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('Saved example_input.npz, one_frame_output.npz, run_summary.json in', root.resolve())


if __name__ == '__main__':
    main()
