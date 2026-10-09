# DC-PANR：方向条件化导频辅助神经接收机（公开代码）

论文：M. Li, Y. Li, J. Xie, and Q. Feng, “Direction-Conditioned Pilot-Aided Neural Receiver for Cochannel Signal Separation,” *IEEE Wireless Communications Letters*, 2026，DOI：[10.1109/LWC.2026.3741686](https://doi.org/10.1109/LWC.2026.3741686)。

本仓库只包含**本文提出的方法**，删除了所有对比算法、消融实验、参数扫描、审稿补充实验、绘图脚本和已有结果。它是**单个共享模型训练 + 单帧接收推理**的独立工程。

## 保留的完整方法

1. 已知目标方向 `DOA`，输入为多天线复数 IQ 波形；DOA 映射为 `[sinθ, cosθ]`。
2. 可学习复数空间权重；对每个天线独立加权并保留各天线数据流。
3. 复数卷积前端、双尺度条件化时序 U-Net、FiLM、门控加性跳跃连接。
4. 有界复增益/相位修正输出连续时间波形。
5. 导频复数最小二乘校准的**训练损失**。
6. **推理期间**利用接收波形、已知 DOA、已知导频进行相关定时，再用导频 LS 消除复增益不确定性、抽样、硬判决及 Gray 比特映射。

**重要：** 这是多天线、已知目标 DOA 和导频的辅助分离方法，不是无导频、无方向信息的单通道盲源分离。推理不使用真实信道增益或仿真真实符号定时。训练阶段允许使用模拟器产生的真值波形及符号位置作为监督信息。

## 一、解压后首次安装

在 PyCharm 打开工程根目录，新建 Python 3.10+ 环境（推荐 3.11/3.12），在 Terminal 执行：

```bash
python -m pip install -e .
```

推荐采用配套 GPU 的 PyTorch 官方安装方法；无 GPU 可先做快速软件自检。

Windows 用户也可双击 `RUN_QUICK_WINDOWS.bat`：它会创建 `.venv`、安装依赖并执行一次快速测试。若已在 PyCharm 配置好环境，直接运行 `run_once.py --quick` 即可。

## 二、最快运行：软件自检（非正式实验）

```bash
python run_once.py --quick
```

使用**缩小的网络、缩短的帧、很少的训练数据和 1 个 epoch**，只用于检验数据生成、训练、保存、实际导频定时、推理、输出文件是否通畅。**不能用于文章结果或作为论文方法的性能指标。**

快速运行后产生：

- `outputs/quick_smoke/example_input.npz`：生成的单帧多天线接收 IQ、用户 DOA 和已知导频；
- `outputs/quick_smoke/one_frame_output.npz`：提取波形、估计时偏、校准符号及解调比特；
- `outputs/quick_smoke/run_summary.json`：实际运行摘要；
- `outputs/checkpoints/dcpanr_qpsk_SMOKE_ONLY_seed20260711.pt`：仅用于测试的模型权重。

## 三、一次正式训练 + 单帧推理

```bash
python run_once.py --modulation QPSK --snr-db 0
```

或 16QAM：

```bash
python run_once.py --modulation 16QAM --snr-db 12
```

默认参数：3 用户、3 接收天线、长度 1024、每符号 8 个采样、20 个导频；训练 16000 帧、验证 2400 帧，最大 100 epoch、batch 32，主网络 `base_ch=48`，训练学习率 8e-4；训练接收机参数量 **3,363,283**。这需要较长训练时间，建议 CUDA GPU。

模型默认保存至：

```text
outputs/checkpoints/dcpanr_qpsk_full_seed20260711.pt
```

以后相同配置再次运行会自动加载已训练模型，添加 `--force` 则重新训练。**这里单独训练一个随机种子，不提供论文三次独立训练曲线的统计复现。**

## 四、只进行正式训练

```bash
python train.py --modulation QPSK
```

## 五、使用自己真实的混叠接收信号

你应准备 `your_frame.npz`，有三个键：

| 键名 | 数据形状 | 含义 |
|---|---|---|
| `iq` | `[M,N]`，complex64 | 每个天线一行的复数 IQ 接收波形 |
| `doa_deg` | `[K]`，float | 外部提供的 K 个目标用户 DOA（单位：度） |
| `pilot_indices` | `[K,P]`，int | K 个用户各自已知的导频**星座索引**，与模型的调制方案相同 |

例如默认 QPSK 模型的 `M=3, N=1024, K=3, P=20`。索引对应的调制星座定义位于 `dc_panr/data.py`。用户 DOA 的排列顺序应与导频序列顺序一致。默认模型适用矩形脉冲、整数定时、窄带帧常信道；真实设备数据若不满足这些条件，需要相应的数据处理与重新训练。

执行：

```bash
python infer.py --checkpoint outputs/checkpoints/dcpanr_qpsk_full_seed20260711.pt --input your_frame.npz --output outputs/your_result.npz
```

输出 `your_result.npz` 包含：

- `raw_waveforms`：K 个用户的神经网络输出复数波形；
- `estimated_offsets`：由已知导频相关估计的实际时偏；
- `timing_scores`：时偏候选的相关分数；
- `estimated_complex_gain`：导频 LS 复数校准系数；
- `symbol_positions`：估计的采样位置；
- `calibrated_symbols`：校准后的符号；
- `payload_symbol_indices` / `payload_constellation_symbols` / `payload_bits`：硬判决输出。

### 注意

本压缩包**不提供论文最终预训练权重**，只能从代码重新训练；也不含完整论文实验数据。不应将 `--quick` 的结果理解为真实分离性能。RRC 定时函数也有保留，但它需要匹配的 RRC 训练模型，不代表矩形脉冲训练权重可直接在 RRC 信号上达到论文效果。

## 六、软件基础单元测试

```bash
python -m unittest discover -s tests -v
```

## 七、开源前最后需要确认

项目内包含 `LICENSE`（MIT）、`CITATION.cff`、`.gitignore`、GitHub 仓库上传指导及英文 `README.md`，可作为 GitHub 仓库源文件使用。**正式公开前，请与共同作者及可能拥有知识产权的单位核实软件开源权限**，并检查是否存在第三方代码授权、专利或保密限制。

GitHub/Zenodo 具体步骤在 `docs/OPEN_SOURCE_CHECKLIST.md`。
