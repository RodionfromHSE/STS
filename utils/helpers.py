import json
from omegaconf import OmegaConf, DictConfig


def read_config(path: str) -> DictConfig:
    """Reads a config file from a given path and resolves it.

    Args:
        path (str): Path to the config file.

    Returns:
        DictConfig: Config file in DictConfig format.
    """
    config = OmegaConf.load(path)
    config = OmegaConf.to_container(config, resolve=True)
    config = DictConfig(config)
    return config


def print_config(config: DictConfig) -> None:
    """Prints the config file.

    Args:
        config (DictConfig): Config file.
    """
    print(json.dumps(OmegaConf.to_container(config, resolve=True), indent=2))
