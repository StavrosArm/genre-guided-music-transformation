import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from types import SimpleNamespace


class MusicModel(nn.Module):
    def __init__(self, config, freeze_encoder = True):
        """
        Init function. Loads, the wav2vec model, and initializes a linear layer targeting the genre classes.
        By default, the encoder's weights are frozen.
        """
        super().__init__()
        self.config = config
        self.encoder = Wav2Vec2Model.from_pretrained(self.config.model.model_name)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(self.config.model.hidden_size, self.config.model.num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        :param x: Input tensor of the waveform
        :return: Output tensor of the class logits

        """
        outputs = self.encoder(x)
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        logits = self.classifier(pooled)

        return logits


if __name__ == "__main__":
    config = SimpleNamespace()
    config.model = SimpleNamespace(
        model_name="facebook/wav2vec2-base",
        hidden_size=768,
        num_labels=8
    )

    model = MusicModel(config)
    print(model)