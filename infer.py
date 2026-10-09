"""Standalone inference on YOUR frame without ground-truth channel or timing."""
import argparse
from pathlib import Path

import numpy as np

from dc_panr.inference import infer_frame, load_receiver


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True,
                   help='NPZ with complex iq [M,N], doa_deg [K], pilot_indices [K,P]')
    p.add_argument('--output', type=Path, default=Path('outputs/inference_result.npz'))
    p.add_argument('--timing', choices=['rectangular', 'rrc'], default='rectangular')
    p.add_argument('--device', default='auto')
    args = p.parse_args()
    with np.load(args.input, allow_pickle=False) as file:
        iq, doa_deg, pilots = (file[k] for k in ('iq', 'doa_deg', 'pilot_indices'))
    model, cfg = load_receiver(args.checkpoint, args.device)
    result = infer_frame(model, cfg, iq, doa_deg, pilots, timing=args.timing)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(args.output, **result)
    print('Saved:', args.output.resolve())
    print('Estimated symbol offsets:', result['estimated_offsets'].tolist())
    print('Hard payload bits shape:', result['payload_bits'].shape)


if __name__ == '__main__':
    main()
