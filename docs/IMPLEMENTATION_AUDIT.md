# Technical release audit (2026-10-09)

The source was extracted from the user's `修改文章1(1).zip` supplementary research project. It is a standalone **method-only** release and omits conventional/neural baselines, reviewer ablations, graphs, parameter sweeps and manuscript sources.

## Direct parity checks against the original method source

| Check | Result |
| --- | --- |
| Proposed full model parameter count | Original 3,363,283; extracted 3,363,283 |
| State-dict key list | Identical |
| Forward result with identical weights and input | Maximum absolute difference: `0.0` |
| Training data generator (identical seed/config) | All generated arrays identical |
| Pilot-calibrated waveform loss (identical tensors) | Absolute difference: `0.0` |
| Observation-based pilot timing scores and offsets | Identical |
| Independent model checkpoint reload and one-frame inference | Passed |
| QPSK and 16QAM smoke training + one-frame inference | Both passed |
| Software unit tests | 6/6 passed |

The program also successfully installed as an editable Python package. These checks verify implementation identity and program integrity; they do **not** establish published BER values or performance on real radio measurements.

## Exactly what is preserved

- `dc_panr/model.py`: proposed learned-complex-weight + conditioned U-Net + bounded refine architecture only; state-dict names preserved.
- `dc_panr/training.py`: online array mixture simulation, direction jitter, supervised pilot-calibrated loss, AdamW/scheduler/early stop, single checkpoint.
- `dc_panr/inference.py`: the original practical pilot-correlation timing routine with the same timing outputs, followed by pilot LS calibration and hard demodulation.
- `dc_panr/data.py`: the original synthetic source, pilot, constellation and ULA definitions; reviewer-specific exact-gap scenario generation omitted.

## Limitations

- Source only; the user-provided research archive did **not** contain a pretrained final model for distribution.
- Formal training was not executed end-to-end during this extraction. A full run is computationally more expensive than software smoke tests.
- The paper reports three-run averages, whereas this public release implements **one** method-training run and **one** inference frame at a time, as requested.
- The `rrc` timing API is exposed for trained pulse-shaped receivers, but the default synthetic training scenario and one-click run use a rectangular pulse; the project does not include the paper's RRC multipath study.
- MIT text is proposed on the assumption that legal rights holders approve open-source publication; verify before public release.
