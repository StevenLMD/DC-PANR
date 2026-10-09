"""Direction-Conditioned Pilot-Aided Neural Receiver for cochannel separation."""
from .config import ExperimentConfig, config_for_modulation, small_smoke_config
from .model import DirectionConditionedReceiver
from .inference import infer_frame, load_receiver

__all__ = [
    'ExperimentConfig', 'config_for_modulation', 'small_smoke_config',
    'DirectionConditionedReceiver', 'infer_frame', 'load_receiver',
]
__version__ = '1.0.0'
