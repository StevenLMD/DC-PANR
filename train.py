"""Standalone one-model training entry point (no experiment sweeps)."""
import argparse
from pathlib import Path

from dc_panr.config import config_for_modulation, small_smoke_config
from dc_panr.training import train_once


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--modulation', choices=['QPSK', '16QAM'], default='QPSK')
    p.add_argument('--quick', action='store_true', help='Tiny plumbing test only')
    p.add_argument('--force', action='store_true')
    p.add_argument('--checkpoint', type=Path)
    args = p.parse_args()
    cfg = config_for_modulation(args.modulation)
    if args.quick:
        cfg = small_smoke_config(cfg)
        print('WARNING: quick model/settings are not paper-equivalent')
    checkpoint, stats = train_once(cfg, checkpoint=args.checkpoint, force=args.force, quick=args.quick)
    print('Checkpoint:', checkpoint.resolve())
    print('Training result:', stats)


if __name__ == '__main__':
    main()
