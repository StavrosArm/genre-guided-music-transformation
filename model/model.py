import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from types import SimpleNamespace
import os


class MusicModel(nn.Module):
    def __init__(self, config, freeze_encoder=True):
        """
        Init function. Loads the music2vec model, and initializes a linear layer targeting the genre classes.
        Only uses encoder layers up to the 7th layer.
        """
        super().__init__()
        self.config = config
        full_model = AutoModel.from_pretrained(self.config.model.model_name)

        # Keep only the first 7 transformer layers
        full_model.encoder.layers = full_model.encoder.layers[:7]
        self.encoder = full_model

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.GELU(),
            nn.Dropout(self.config.training.dropout_prob),
            nn.Linear(self.config.model.hidden_size, self.config.model.classification_dim),
            nn.GELU(),
            nn.Dropout(self.config.training.dropout_prob),
            nn.Linear(self.config.model.classification_dim, self.config.model.num_labels)
        )

    def load_from_checkpoint(self, filename="best_model.pt"):
        checkpoint_path = os.path.join(self.config.training.checkpoint_dir, filename)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(x)
        hidden_states = outputs.last_hidden_state
        max_pooled, _ = hidden_states.max(dim=1)
        mean_pooled = hidden_states.mean(dim=1)
        pooled = (max_pooled + mean_pooled)/2.0
        logits = self.classifier(pooled)
        return logits


if __name__ == "__main__":
    config = SimpleNamespace()
    config.model = SimpleNamespace(
        model_name="facebook/wav2vec2-base",
        hidden_size=768,
        classification_dim=512,
        num_labels=8
    )
    config.training = SimpleNamespace(
        dropout_prob=0.1
    )

    model = MusicModel(config)
    print(model)
