# Method implementation map

Paper: "Direction-Conditioned Pilot-Aided Neural Receiver for Cochannel Signal Separation", IEEE WCL, 2026, DOI `10.1109/LWC.2026.3741686`.

The implementation preserves the original proposed `M80Receiver` model module names and its trainable parameter tensors. It has no baselines, ablation construction code or paper figure pipelines.

## Assumed physical observations

Frame `Y` is `M x N` complex IQ data from a half-wavelength ULA. `K` cochannel users use frame-constant complex coefficients, differing integer-sample offsets, and known distinct pilot prefixes. External target DOAs are known or estimated upstream. The receiver is given only `Y`, `DOA` and pilots.

## Trainable estimator (`dc_panr/model.py`)

- `LearnedSpatialWeightNet`: normalized complex weights from target `[sinθ,cosθ]`.
- Each antenna stream is rotated independently by conjugated complex weight; streams are not collapsed into an analytic beam sum.
- `ComplexConv1d`: complex convolution (length 9), GroupNorm, PReLU.
- `CondAwareNoConcatAdditiveUNet`: four-level U-Net, base width 48, dual length-3/17 depthwise temporal branches, condition-weighted softmax selection, FiLM conditioning and additive gated skip fusion.
- `PhaseRotationRefineHead`: bounded gain/phase correction with default bounds `(0.12, 0.15 rad)`.
- Model output: estimated target complex waveform `[B,2,N]`. It is the same shared model evaluated for each user independently.

## Optimization (`dc_panr/training.py`)

- Training simulator generates target waveforms, true H, random DOAs, power imbalance, channel phase, integer timing and training pilot indices. These physical truth variables generate supervised *labels* and mixture observations, not model input features.
- `pilot_calibrated_waveform_loss`: use simulated known pilot sample positions for the supervised loss and compute the closed-form complex scalar `α = Σ conj(pred_p) target_p / (Σ |pred_p|² + ε)`. Waveform error uses `|α pred - target|²`.
- AdamW, plateau scheduler, gradient clipping, early stopping, DOA perturbation, and online noise are implemented with the default paper settings.
- Validation frames are used for early stopping of **one model**. No multi-method or multi-condition sweeps are included.

## Deployment (`dc_panr/inference.py`)

- `estimate_user_offsets` correlates the steering-projected received IQ with candidate time-shifted repetitions of the **known pilots**. No oracle offset is used.
- `infer_frame` samples neural waveforms at the **estimated** integer timing grid, computes pilot LS phase/gain correction, applies it to all symbols, then hard-demodulates the nonpilot payload and exports Gray bits.
- The API has no argument for true H, power ratios, transmitted payload, or oracle offsets.
- `estimate_user_offsets_pulse_shaped` is an optional RRC-template timing implementation. Deploy it only with a model trained on pulse-shaped frames and known RRC pulse parameters.

## Output and interpretation

- The smoke test establishes execution and interface integrity, not BER generalization.
- The one-run code is not a three-seed reproduction of all journal plots.
- Actual signals require compatible sample rate, antenna order, IQ convention, frame design, DOA, and pilot definition. DOA estimation and carrier/fractional clock recovery are outside this paper's proposed implementation.
