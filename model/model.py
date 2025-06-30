import torch
import torch.nn as nn
from transformers import AutoModel
from types import SimpleNamespace
import os


class MusicModel(nn.Module):
    def __init__(self, config, freeze_encoder=True):
        """
        Init function. Loads, the music2vec model, and initializes a linear layer targeting the genre classes.
        By default, the encoder's weights are frozen.
        """
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(self.config.model.model_name)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(nn.GELU(), nn.Dropout(self.config.training.dropout_prob),
                                        nn.Linear(self.config.model.hidden_size, self.config.model.num_labels))

    def load_from_checkpoint(self, filename="best_model.pt"):
        checkpoint_path = os.path.join(self.config.training.checkpoint_dir, filename)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(state_dict)

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
