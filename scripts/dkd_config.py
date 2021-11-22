import torch
import dataclasses
from typing import Tuple


@dataclasses.dataclass
class TaskConfig:
    keyword: str = 'sheila'  # We will use 1 key word -- 'sheila'
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    num_epochs: int = 25
    n_mels: int = 40
    cnn_out_channels: int = 3
    kernel_size: Tuple[int, int] = (3, 20)
    stride: Tuple[int, int] = (2, 8)
    hidden_size: int = 20
    gru_num_layers: int = 1
    bidirectional: bool = False
    num_classes: int = 2
    sample_rate: int = 16000
    device: torch.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    wandb_api: str = "99f2c4dae0db3099861ebd92a63e1194f42d16d9"
    dkd_alpha: float = 0.3
    dkd_temperature: float = 7
    train_type: str = "distill"
    teacher_model_path: str = "model_weight"
    quantize_model: bool = False
