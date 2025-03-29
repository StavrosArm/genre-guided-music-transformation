import torch
import torch.nn as nn
from transformers import AutoModel

class MusicModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = AutoModel.from_pretrained(config.model.model_name)

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return self.model(*x)

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return self.forward(input)
