"""Minimal software correctness tests; NOT method performance experiments."""
from __future__ import annotations

import inspect
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from dc_panr.config import ExperimentConfig, small_smoke_config
from dc_panr.data import bits_table, constellation, generate_dataset_arrays
from dc_panr.inference import infer_frame, load_receiver
from dc_panr.model import DirectionConditionedReceiver
from dc_panr.training import mix_batch, pilot_calibrated_waveform_loss


class SingleMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(91)
        cls.cfg = small_smoke_config(ExperimentConfig())

    def test_full_architecture_parameter_count(self):
        model = DirectionConditionedReceiver(ExperimentConfig())
        self.assertEqual(sum(p.numel() for p in model.parameters()), 3_363_283)

    def test_model_single_user_output(self):
        model = DirectionConditionedReceiver(self.cfg).eval()
        with torch.no_grad():
            out = model(torch.randn(2, self.cfg.num_antennas * 2, self.cfg.seq_len),
                        torch.tensor([[0.0, 1.0], [0.5, 0.8660254]]))
        self.assertEqual(tuple(out.shape), (2, 2, self.cfg.seq_len))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_pilot_calibrated_loss_invariant_to_global_complex_gain(self):
        rng = np.random.default_rng(22)
        target_np = (rng.standard_normal((2, 3, self.cfg.seq_len)) +
                     1j * rng.standard_normal((2, 3, self.cfg.seq_len))).astype(np.complex64)
        predicted = (2.0 + 3.0j) * target_np
        to_tensor = lambda x: torch.from_numpy(np.stack([x.real, x.imag], axis=2).copy())
        pos = torch.tensor(np.broadcast_to(np.arange(2, 2 + self.cfg.pilot_symbols) * self.cfg.sps,
                                           (2, 3, self.cfg.pilot_symbols)).copy())
        loss = pilot_calibrated_waveform_loss(to_tensor(predicted), to_tensor(target_np), pos)
        self.assertLess(float(loss), 1e-9)

    def test_generated_training_data_shape(self):
        arr = generate_dataset_arrays(self.cfg, 2, 234, self.cfg.train_min_sep_deg,
                                      self.cfg.train_power_spread_db)
        self.assertEqual(arr['sources_ch'].shape, (2, 3, 2, self.cfg.seq_len))
        self.assertEqual(arr['pilot_indices'].shape, (3, self.cfg.pilot_symbols))
        self.assertEqual(bits_table('QPSK').shape, (4, 2))
        self.assertEqual(constellation('16QAM').shape, (16,))

    def test_deployment_does_not_request_oracle_channels_or_timing(self):
        params = inspect.signature(infer_frame).parameters
        self.assertEqual({'receiver', 'cfg', 'iq', 'doa_deg', 'pilot_indices'}.issubset(params), True)
        for prohibited in ('h', 'h_true', 'true_offsets', 'symbol_positions', 'true_symbols'):
            self.assertNotIn(prohibited, params)

    def test_infer_and_reload_from_single_checkpoint(self):
        cfg = self.cfg
        model = DirectionConditionedReceiver(cfg).eval()
        with tempfile.TemporaryDirectory() as directory:
            ckpt = Path(directory) / 'saved.pt'
            torch.save({'config': asdict(cfg), 'model_state': model.state_dict(),
                        'best_epoch': 1, 'best_val_loss': 0.0}, ckpt)
            reloaded, saved_cfg = load_receiver(ckpt, device='cpu')
            arr = generate_dataset_arrays(cfg, 1, 445, cfg.test_min_sep_deg, 0.0)
            h = torch.complex(torch.from_numpy(arr['h_real']), torch.from_numpy(arr['h_imag']))
            obs = mix_batch(torch.from_numpy(arr['sources_ch']), h, 5.0,
                            noise_seed=222)[0].numpy().reshape(cfg.num_antennas, 2, cfg.seq_len)
            iq = obs[:, 0] + 1j * obs[:, 1]
            result = infer_frame(reloaded, saved_cfg, iq, arr['doa'][0], arr['pilot_indices'])
            self.assertEqual(result['raw_waveforms'].shape, (cfg.num_sources, cfg.seq_len))
            self.assertEqual(result['estimated_offsets'].shape, (cfg.num_sources,))
            self.assertTrue(np.all(result['estimated_offsets'] < cfg.sps))
            self.assertEqual(result['payload_bits'].shape[0], cfg.num_sources)
            self.assertTrue(np.isfinite(result['calibrated_symbols']).all())


if __name__ == '__main__':
    unittest.main()
