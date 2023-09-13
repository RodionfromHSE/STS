import json
from omegaconf import OmegaConf, DictConfig


def read_config(path: str, set_args: dict = None) -> DictConfig:
    """Reads a config file from a given path and resolves it.

    Args:
        path (str): Path to the config file.
        set_args (dict, optional): Arguments to set in the config file. Defaults to None.

    Returns:
        DictConfig: Config file in DictConfig format.
    """
    config = OmegaConf.load(path)
    if set_args is not None:
        config = OmegaConf.merge(config, set_args)
    config = OmegaConf.to_container(config, resolve=True)
    config = DictConfig(config)
    return config


def print_config(config: DictConfig) -> None:
    """Prints the config file.

    Args:
        config (DictConfig): Config file.
    """
    print(json.dumps(OmegaConf.to_container(config, resolve=True), indent=2))
