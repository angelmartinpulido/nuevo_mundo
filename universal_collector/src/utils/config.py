import json
import os


def load_config(config_path="config.json"):
    """
    Carga la configuración del sistema desde un archivo JSON.

    Args:
        config_path (str): Ruta al archivo de configuración

    Returns:
        dict: Configuración cargada
    """
    default_config = {
        "model_path": "models/",
        "data_path": "data/",
        "log_level": "INFO",
    }

    if not os.path.exists(config_path):
        return default_config

    with open(config_path, "r") as f:
        config = json.load(f)

    return {**default_config, **config}
