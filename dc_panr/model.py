from __future__ import annotations

import math
import os

import torch

torch.set_num_threads(max(1, int(os.environ.get("DC_PANR_TORCH_THREADS", "1"))))
import torch.nn as nn
import torch.nn.functional as F

from .config import ExperimentConfig


def groups(channels: int, max_groups: int = 8) -> int:
    g = min(channels, max_groups)
    while g > 1 and channels % g:
        g -= 1
    return max(1, g)


def obs_to_complex(obs: torch.Tensor, num_antennas: int) -> torch.Tensor:
    b, channels, length = obs.shape
    expected = 2 * num_antennas
    if channels != expected:
        raise ValueError(f"Expected {expected} observation channels, got {channels}")
    x = obs.view(b, num_antennas, 2, length)
    return torch.complex(x[:, :, 0], x[:, :, 1])


def complex_to_channels(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return torch.stack([x.real, x.imag], dim=2).float()
    if x.ndim == 2:
        return torch.stack([x.real, x.imag], dim=1).float()
    raise ValueError(f"Unsupported complex tensor rank: {x.ndim}")


def flatten_complex_channels(x: torch.Tensor) -> torch.Tensor:
    b, pairs, length = x.shape
    return torch.stack([x.real, x.imag], dim=2).reshape(b, 2 * pairs, length).float()


# -----------------------------------------------------------------------------
# Direction-conditioned receiver architecture from the associated LWC article.
# -----------------------------------------------------------------------------

class LearnedSpatialWeightNet(nn.Module):
    """Original direction-to-complex-weight MLP used by the m80 lineage."""

    def __init__(self, num_antennas: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.PReLU(hidden),
            nn.Linear(hidden, hidden),
            nn.PReLU(hidden),
            nn.Linear(hidden, 2 * num_antennas),
        )

    def forward(self, direction: torch.Tensor) -> torch.Tensor:
        raw = self.net(direction.float())
        real, imag = raw.chunk(2, dim=1)
        weight = torch.complex(real, imag)
        return weight / torch.sqrt(torch.sum(torch.abs(weight) ** 2, dim=1, keepdim=True) + 1e-8)


class ComplexConv1d(nn.Module):
    def __init__(self, in_pairs: int, out_pairs: int, kernel_size: int = 9) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.real = nn.Conv1d(in_pairs, out_pairs, kernel_size, padding=pad)
        self.imag = nn.Conv1d(in_pairs, out_pairs, kernel_size, padding=pad)
        self.norm = nn.GroupNorm(groups(2 * out_pairs), 2 * out_pairs)
        self.act = nn.PReLU(2 * out_pairs)

    def forward(self, pair_channels: torch.Tensor) -> torch.Tensor:
        b, two_pairs, length = pair_channels.shape
        if two_pairs % 2:
            raise ValueError("ComplexConv1d expects interleaved real/imag pairs")
        pairs = two_pairs // 2
        x = pair_channels.view(b, pairs, 2, length)
        xr, xi = x[:, :, 0], x[:, :, 1]
        yr = self.real(xr) - self.imag(xi)
        yi = self.real(xi) + self.imag(xr)
        y = torch.stack([yr, yi], dim=2).reshape(b, 2 * yr.shape[1], length)
        return self.act(self.norm(y))


class LowRankPointwise1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rank_ratio: float = 0.45, act: bool = False) -> None:
        super().__init__()
        rank = max(8, int(round(min(in_ch, out_ch) * rank_ratio)))
        layers: list[nn.Module] = [
            nn.Conv1d(in_ch, rank, 1, bias=False),
            nn.GroupNorm(groups(rank), rank),
        ]
        if act:
            layers.append(nn.SiLU())
        layers.append(nn.Conv1d(rank, out_ch, 1, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CondFiLM1D(nn.Module):
    def __init__(self, channels: int, cond_dim: int, scale: float = 0.20) -> None:
        super().__init__()
        hidden = max(16, channels // 4)
        self.scale = float(scale)
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * channels),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(cond.float()).chunk(2, dim=1)
        gamma = torch.tanh(gamma).unsqueeze(-1).to(x.dtype)
        beta = torch.tanh(beta).unsqueeze(-1).to(x.dtype)
        return x * (1.0 + self.scale * gamma) + self.scale * beta


class CondAwareShortLongLowRankBlock1D(nn.Module):
    """Original m80 condition-aware short/long low-rank block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        expand: int = 2,
        kernel_sizes: tuple[int, ...] = (3, 17),
        rank_ratio: float = 0.45,
    ) -> None:
        super().__init__()
        self.proj = LowRankPointwise1D(in_ch, out_ch, rank_ratio) if in_ch != out_ch else nn.Identity()
        hidden = out_ch * expand
        self.expand = nn.Sequential(
            LowRankPointwise1D(out_ch, hidden, rank_ratio, act=True),
            nn.GroupNorm(groups(hidden), hidden),
            nn.SiLU(),
        )
        self.branches = nn.ModuleList([
            nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2, groups=hidden, bias=False)
            for kernel in kernel_sizes
        ])
        branch_count = len(kernel_sizes)
        gate_hidden = max(8, hidden // 24)
        self.feature_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden, gate_hidden, 1),
            nn.SiLU(),
            nn.Conv1d(gate_hidden, hidden * branch_count, 1),
        )
        self.cond_gate = nn.Linear(cond_dim, hidden * branch_count)
        nn.init.zeros_(self.cond_gate.weight)
        nn.init.zeros_(self.cond_gate.bias)
        self.project = nn.Sequential(
            nn.GroupNorm(groups(hidden), hidden),
            LowRankPointwise1D(hidden, out_ch, rank_ratio),
            nn.GroupNorm(groups(out_ch), out_ch),
        )
        self.act = nn.PReLU(out_ch)
        self.gamma = nn.Parameter(torch.ones(1, out_ch, 1) * 1e-3)
        self.branch_count = branch_count
        self.hidden = hidden

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        h = self.expand(x)
        stack = torch.stack([branch(h) for branch in self.branches], dim=1)
        feature_gate = self.feature_gate(h).view(h.shape[0], self.branch_count, self.hidden, 1)
        cond_gate = self.cond_gate(cond.float()).view(h.shape[0], self.branch_count, self.hidden, 1).to(h.dtype)
        weights = torch.softmax(feature_gate + 0.25 * torch.tanh(cond_gate), dim=1)
        y = torch.sum(stack * weights, dim=1)
        y = self.project(y)
        return self.act(x + self.gamma.to(x.dtype) * y)


class CondAwareDownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, kernel_sizes: tuple[int, ...] = (3, 17)) -> None:
        super().__init__()
        self.down = nn.Conv1d(in_ch, out_ch, 4, stride=2, padding=1)
        self.block = CondAwareShortLongLowRankBlock1D(out_ch, out_ch, cond_dim, kernel_sizes=kernel_sizes)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x), cond)


class CondAwareUpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cond_dim: int, additive: bool = True, kernel_sizes: tuple[int, ...] = (3, 17)) -> None:
        super().__init__()
        self.additive = additive
        self.x_proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.s_proj = nn.Conv1d(skip_ch, out_ch, 1) if skip_ch != out_ch else nn.Identity()
        self.skip_gate = nn.Sequential(nn.Conv1d(out_ch, out_ch, 1), nn.Sigmoid())
        self.cond_skip = nn.Linear(cond_dim, out_ch)
        nn.init.zeros_(self.cond_skip.weight)
        nn.init.zeros_(self.cond_skip.bias)
        block_in = out_ch if additive else 2 * out_ch
        self.block = CondAwareShortLongLowRankBlock1D(block_in, out_ch, cond_dim, kernel_sizes=kernel_sizes)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = self.x_proj(x)
        s = self.s_proj(skip)
        cond_gain = 1.0 + 0.20 * torch.tanh(self.cond_skip(cond.float())).unsqueeze(-1).to(s.dtype)
        s = s * self.skip_gate(s) * cond_gain
        fused = x + s if self.additive else torch.cat([x, s], dim=1)
        return self.block(fused, cond)


class CondAwareNoConcatAdditiveUNet(nn.Module):
    """Original four-level condition-aware additive U-Net used by m80."""

    def __init__(self, in_channels: int, out_channels: int, base: int, cond_dim: int, additive: bool = True, kernel_sizes: tuple[int, ...] = (3, 17)) -> None:
        super().__init__()
        self.enc0 = CondAwareShortLongLowRankBlock1D(in_channels, base, cond_dim, kernel_sizes=kernel_sizes)
        self.f0 = CondFiLM1D(base, cond_dim)
        self.down1 = CondAwareDownBlock(base, base * 2, cond_dim, kernel_sizes); self.f1 = CondFiLM1D(base * 2, cond_dim)
        self.down2 = CondAwareDownBlock(base * 2, base * 4, cond_dim, kernel_sizes); self.f2 = CondFiLM1D(base * 4, cond_dim)
        self.down3 = CondAwareDownBlock(base * 4, base * 6, cond_dim, kernel_sizes); self.f3 = CondFiLM1D(base * 6, cond_dim)
        self.down4 = CondAwareDownBlock(base * 6, base * 8, cond_dim, kernel_sizes); self.f4 = CondFiLM1D(base * 8, cond_dim)
        self.bottleneck = CondAwareShortLongLowRankBlock1D(base * 8, base * 8, cond_dim, kernel_sizes=kernel_sizes)
        self.fb = CondFiLM1D(base * 8, cond_dim)
        self.up4 = CondAwareUpBlock(base * 8, base * 6, base * 6, cond_dim, additive, kernel_sizes); self.fu4 = CondFiLM1D(base * 6, cond_dim)
        self.up3 = CondAwareUpBlock(base * 6, base * 4, base * 4, cond_dim, additive, kernel_sizes); self.fu3 = CondFiLM1D(base * 4, cond_dim)
        self.up2 = CondAwareUpBlock(base * 4, base * 2, base * 2, cond_dim, additive, kernel_sizes); self.fu2 = CondFiLM1D(base * 2, cond_dim)
        self.up1 = CondAwareUpBlock(base * 2, base, base, cond_dim, additive, kernel_sizes); self.fu1 = CondFiLM1D(base, cond_dim)
        self.out = nn.Sequential(
            nn.Conv1d(base, base, 3, padding=1),
            nn.PReLU(base),
            nn.Conv1d(base, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        e0 = self.f0(self.enc0(x, cond), cond)
        e1 = self.f1(self.down1(e0, cond), cond)
        e2 = self.f2(self.down2(e1, cond), cond)
        e3 = self.f3(self.down3(e2, cond), cond)
        e4 = self.f4(self.down4(e3, cond), cond)
        z = self.fb(self.bottleneck(e4, cond), cond)
        d3 = self.fu4(self.up4(z, e3, cond), cond)
        d2 = self.fu3(self.up3(d3, e2, cond), cond)
        d1 = self.fu2(self.up2(d2, e1, cond), cond)
        d0 = self.fu1(self.up1(d1, e0, cond), cond)
        return self.out(d0)


class PhaseRotationRefineHead(nn.Module):
    """Original bounded I/Q gain-rotation refinement head."""

    def __init__(self, hidden: int = 24, max_gain: float = 0.12, max_phase: float = 0.15) -> None:
        super().__init__()
        hidden = max(8, int(hidden))
        self.net = nn.Sequential(
            nn.Conv1d(2, hidden, 9, padding=4, bias=False),
            nn.GroupNorm(groups(hidden), hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2, groups=hidden, bias=False),
            nn.SiLU(),
            nn.Conv1d(hidden, 2, 1),
        )
        self.max_gain = float(max_gain)
        self.max_phase = float(max_phase)
        self.scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        delta = self.net(y)
        gain_delta, phase_delta = delta[:, 0:1], delta[:, 1:2]
        gain = 1.0 + self.max_gain * torch.tanh(self.scale).to(y.dtype) * torch.tanh(gain_delta)
        phase = self.max_phase * torch.tanh(self.scale).to(y.dtype) * torch.tanh(phase_delta)
        cos_phase, sin_phase = torch.cos(phase), torch.sin(phase)
        yr, yi = y[:, 0:1], y[:, 1:2]
        rr = gain * (cos_phase * yr - sin_phase * yi)
        ri = gain * (sin_phase * yr + cos_phase * yi)
        return torch.cat([rr, ri], dim=1)


class M80Receiver(nn.Module):
    """Direction-conditioned, stream-preserving neural waveform extractor.

    No target channel vector and no analytic beam residual are supplied to the
    proposed receiver. Target information enters only through [sin(theta),
    cos(theta)], the learned complex spatial weights, FiLM, condition-aware
    short/long kernel selection, and condition-aware additive skip fusion.
    """

    kind = "target"

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.num_antennas = cfg.num_antennas
        self.weight_net = LearnedSpatialWeightNet(cfg.num_antennas, cfg.weight_hidden)
        stem_pairs = max(4, cfg.base_ch // 2)
        self.complex_stem = ComplexConv1d(cfg.num_antennas, stem_pairs, kernel_size=9)
        cond_dim = 2 + 2 * cfg.num_antennas
        self.backbone = CondAwareNoConcatAdditiveUNet(
            2 * stem_pairs, 2, cfg.base_ch, cond_dim, additive=True, kernel_sizes=(3, 17)
        )
        self.refine = PhaseRotationRefineHead(
            max(8, cfg.base_ch // 2),
            max_gain=cfg.refine_max_gain,
            max_phase=cfg.refine_max_phase,
        )

    def forward(self, obs: torch.Tensor, direction: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        direction_used = direction
        weights = self.weight_net(direction_used).to(torch.complex64)
        y = obs_to_complex(obs, self.num_antennas)
        rotated = torch.conj(weights).unsqueeze(-1) * y
        x = self.complex_stem(flatten_complex_channels(rotated))
        cond = torch.cat([
            direction_used.float(),
            weights.real.float(),
            weights.imag.float(),
        ], dim=1)
        return self.refine(self.backbone(x, cond))



# Public interface; preserve the original class/state-dict names for compatibility.
DirectionConditionedReceiver = M80Receiver
