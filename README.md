# DC-PANR: Direction-Conditioned Pilot-Aided Neural Receiver

**Reference implementation of the proposed receiver only** for the IEEE *Wireless Communications Letters* paper:

> M. Li, Y. Li, J. Xie, and Q. Feng, "Direction-Conditioned Pilot-Aided Neural Receiver for Cochannel Signal Separation," *IEEE Wireless Communications Letters*, 2026. DOI: [10.1109/LWC.2026.3741686](https://doi.org/10.1109/LWC.2026.3741686).

[中文说明 / Chinese README](README_CN.md)

## What is included

A **single trained/shared target-user extractor** that uses array observations, a desired-user direction of arrival (DOA) and known pilots. The complete processing chain is:

1. DOA features `[sin(theta), cos(theta)]` and learned normalized complex antenna weights.
2. Stream-preserving complex front end with direction-conditioned dual-scale temporal U-Net.
3. Bounded complex gain/phase refinement.
4. Supervised training with a **pilot-calibrated complex waveform loss**.
5. **Observation-based pilot-correlation timing** (no true offsets), pilot least-squares complex calibration and Gray-coded hard demodulation.

Run the same shared extractor **once per desired user**; there is no separate network per user. Neither the true source channel vector nor its true timing offset is provided during inference. See [algorithm notes](docs/METHOD.md).

**Excluded on purpose:** other receivers, comparisons, baselines, ablations, parameter sweeps, plots, referee responses and manuscript files. Synthetic generation supports initial training and a single-frame example. The source includes no pretrained weights or published numerical results.

## Requirements

- Python 3.10+ (prefer 3.11 or 3.12)
- PyTorch 2.4+ and NumPy 1.24+
- CPU is sufficient for the fast smoke test; a CUDA-capable GPU is **strongly recommended** for the full 16,000-frame training configuration.

### Install

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

If GPU support is important, install a PyTorch wheel for your hardware following the [official PyTorch installer](https://pytorch.org/get-started/locally/) before `pip install -e .`.

## One-command run

### Minimal end-to-end software check (not paper performance)

```bash
python run_once.py --quick
```

It trains a **reduced-size** receiver for one short epoch, synthesizes **one** test observation, estimates its user timing from received pilots, and writes:

```text
outputs/quick_smoke/example_input.npz
outputs/quick_smoke/one_frame_output.npz
outputs/quick_smoke/run_summary.json
outputs/checkpoints/dcpanr_qpsk_SMOKE_ONLY_seed20260711.pt
```

**IMPORTANT:** `--quick` changes network width, sequence size, pilot length and training sample count. Its output is *only a program-integrity test*, NOT a performance number from the article.

### Full single-model run

```bash
python run_once.py --modulation QPSK --snr-db 0
# Or train a single 16QAM model:
python run_once.py --modulation 16QAM --snr-db 12
```

Defaults: 3 users, 3 antennas, 1024 complex samples, 8 samples/symbol, 20 pilots, 16,000 training frames, 2,400 validation frames, model width 48, and up to 100 epochs. A trained model is cached under `outputs/checkpoints/`; repeating the command loads it. To retrain, use `--force`.

The **full** model has **3,363,283 trainable parameters**. A full run may require substantial GPU time and RAM. It trains ONE model seed; it **does not reproduce the three-seed averages and all figures from the paper**.

You may train only with `python train.py --modulation QPSK` and infer on your own data with `python infer.py`.

### Inference on your own array observation

Save one `.npz` file with:

| Key | Shape / dtype | Description |
| --- | --- | --- |
| `iq` | `[M, N]` complex64/complex128 | Received complex IQ: antenna axis first |
| `doa_deg` | `[K]` float | Estimated target DOAs in degrees, one per user |
| `pilot_indices` | `[K, P]` int | Known pilot symbol indices into `dc_panr.data.constellation(modulation)` |

For the full default QPSK checkpoint: `M=K=3, N=1024, P=20`. The frame must use the training scenario's timing/normalization conventions. DOA and pilot order must match. The receiver **does not** itself estimate DOAs.

Run:

```bash
python infer.py \
  --checkpoint outputs/checkpoints/dcpanr_qpsk_full_seed20260711.pt \
  --input your_frame.npz \
  --output outputs/your_result.npz
```

On Windows, replace line-continuation `\` with a single-line command. To examine the input convention, run `python run_once.py --quick` and inspect `outputs/quick_smoke/example_input.npz`.

Outputs in `your_result.npz`:

- `raw_waveforms [K,N]`: neural complex waveform estimates before pilot correction;
- `estimated_offsets [K]`: estimated **integer** sample offsets from the received pilots;
- `timing_scores [K,sps]`: pilot correlation metrics;
- `estimated_complex_gain [K]`: pilot least-squares calibration;
- `symbol_positions [K,nsym]` and `calibrated_symbols [K,nsym]`;
- `payload_symbol_indices [K,nsym-P]`, `payload_constellation_symbols`, `payload_bits [K,nsym-P,bits_per_symbol]`.

**No ground-truth source symbols, hidden gains or simulated offsets are required by the inference API.** The synthesized example generates them solely to create an observable test mixture; they are not passed into `infer_frame`.

## Public Python API

```python
from dc_panr import load_receiver, infer_frame
import numpy as np

model, cfg = load_receiver("outputs/checkpoints/dcpanr_qpsk_full_seed20260711.pt")
with np.load("your_frame.npz", allow_pickle=False) as frame:
    result = infer_frame(model, cfg, frame["iq"], frame["doa_deg"], frame["pilot_indices"])
print(result["estimated_offsets"])
print(result["payload_bits"].shape)
```

## Scope and limitations

This is a **pilot-aided, direction-conditioned, multi-antenna** receiver, **not blind single-channel separation**. It assumes a known ULA manifold, supplied DOAs, frame-constant narrowband channel coefficients, integer-sample timing, and the stated frame/pilot setup. The default training simulator uses a rectangular pulse; optional `timing='rrc'` supports a known RRC pulse for timing **only if the neural model has been trained on appropriately pulse-shaped data**. No claim is made for untrained deployment on arbitrary RF recordings, CFO, fractional timing, strong multipath, or unknown hardware effects.

Training uses true synthetic labels (including symbol positions) to form the **supervised** pilot-calibrated loss; this does not make the inference procedure oracle-aided. There are no pretrained weights here. The program's quick outputs must not be cited as published BER results.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Citation, license and releasing

- Please cite the paper above and consult [CITATION.cff](CITATION.cff).
- An **MIT license** is provided. **Before publishing**, the release owner must confirm copyright/permission with coauthors and any institution owning code/IP, review third-party provenance, and check possible patent or contractual restrictions.
- Recommended repository name: `DC-PANR`.
- To obtain a citable **software DOI**, connect a public GitHub repository to Zenodo and then publish a tagged release (see [release checklist](docs/OPEN_SOURCE_CHECKLIST.md)).

## Repository contents

```text
dc_panr/
  model.py      # proposed architecture only, preserved original state-dict keys
  training.py   # one-model training and pilot-calibrated loss
  inference.py  # observation-based timing, LS correction, demodulation
  data.py       # synthetic signal and known-pilot construction
  config.py     # original full settings + separate smoke-test settings
  rrc.py        # optional known-pulse pilot timing coefficients
  utils.py
run_once.py     # one train/load + one frame inference
train.py        # train only
infer.py        # inference from external .npz
```
