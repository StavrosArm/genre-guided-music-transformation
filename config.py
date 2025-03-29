import yaml

class Namespace:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, Namespace(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return str(self.__dict__)


class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        self.model = None
        self.optimizer = None
        self.loss = None
        self.dataset = None
        self.feature_extractor = None
        self.training = None
        self.experiment = None

        self.load_config()

    def load_config(self):
        with open(self.config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        self.model = Namespace(cfg.get("model", {}))
        self.optimizer = Namespace(cfg.get("optimizer", {}))
        self.loss = cfg.get("loss", "")
        self.dataset = Namespace(cfg.get("dataset", {}))
        self.feature_extractor = Namespace(cfg.get("feature_extractor", {}))
        self.training = Namespace(cfg.get("training", {}))
        self.experiment = Namespace(cfg.get("experiment", {}))

    def __repr__(self):
        return f"<Config from {self.config_path}>"
