import yaml

class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        self.model = {}
        self.optimizer = {}
        self.loss = None
        self.dataset = {}
        self.feature_extractor = {}
        self.training = {}
        self.experiment = {}

        self.load_config()

    def load_config(self):
        with open(self.config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        self.model = cfg.get("model", {})
        self.optimizer = cfg.get("optimizer", {})
        self.loss = cfg.get("loss", "")
        self.dataset = cfg.get("dataset", {})
        self.feature_extractor = cfg.get("feature_extractor", {})
        self.training = cfg.get("training", {})
        self.experiment = cfg.get("experiment", {})

    def __repr__(self):
        return f"<Config from {self.config_path}>"

