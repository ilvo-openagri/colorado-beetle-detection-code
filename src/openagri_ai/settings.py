import os
import importlib


def load_config(config_name: str):
    try:
        config_module = importlib.import_module(f"config.{config_name}")
        return {k: v for k, v in vars(config_module).items() if not k.startswith("__")}
    except ModuleNotFoundError as ex:
        print(ex)
        raise Exception(f"Configuration '{config_name}' not found.")


CONFIG_NAME = os.getenv("CONFIG_NAME", "default")
if CONFIG_NAME == "raspi":
    from .config.raspi import config
else:
    from .config.default import config
