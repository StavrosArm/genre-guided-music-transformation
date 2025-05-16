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
        self.load_config()

    def load_config(self):
        with open(self.config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        for section in ["model", "optimizer", "loss", "dataset", "training", "experiment", "distortion"]:
            setattr(self, section, Namespace(cfg.get(section, {})))

    def __repr__(self):
        return f"<Config from {self.config_path}>"


if __name__ == "__main__":
    config = Config("config.yaml")
    print(config)