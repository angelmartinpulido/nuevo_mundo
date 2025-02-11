"""
Módulo de configuración central del sistema.
Maneja todas las configuraciones y validaciones necesarias.
"""
import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigurationManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.config_path = os.getenv("CONFIG_PATH", "config/config.json")
            self.config: Dict[str, Any] = {}
            self.load_config()
            self.initialized = True

    def load_config(self) -> None:
        """Carga la configuración desde el archivo JSON y valida su estructura."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    self.config = json.load(f)
            else:
                self.config = self._get_default_config()
            self._validate_config()
        except Exception as e:
            logging.error(f"Error loading configuration: {str(e)}")
            self.config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Retorna la configuración por defecto del sistema."""
        return {
            "system": {
                "name": "adaptive_ai_system",
                "version": "1.0.0",
                "log_level": "INFO",
                "max_threads": 4,
                "timeout": 30,
            },
            "security": {
                "encryption_enabled": True,
                "key_rotation_days": 30,
                "min_password_length": 12,
            },
            "ai": {
                "model_path": "models/",
                "training_data_path": "data/training/",
                "batch_size": 32,
                "learning_rate": 0.001,
            },
            "monitoring": {
                "enabled": True,
                "interval": 60,
                "metrics": ["cpu", "memory", "network"],
            },
            "adapters": {"retry_attempts": 3, "timeout": 5, "buffer_size": 1024},
        }

    def _validate_config(self) -> None:
        """Valida la estructura y valores de la configuración."""
        required_sections = ["system", "security", "ai", "monitoring", "adapters"]
        for section in required_sections:
            if section not in self.config:
                self.config[section] = self._get_default_config()[section]

    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtiene un valor de configuración por su clave.

        Args:
            key: La clave de configuración (puede usar notación punto, ej: 'system.name')
            default: Valor por defecto si no se encuentra la clave

        Returns:
            El valor de configuración o el valor por defecto
        """
        try:
            parts = key.split(".")
            value = self.config
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Establece un valor de configuración.

        Args:
            key: La clave de configuración (puede usar notación punto)
            value: El valor a establecer
        """
        parts = key.split(".")
        config = self.config
        for part in parts[:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]
        config[parts[-1]] = value

    def save(self) -> None:
        """Guarda la configuración actual en el archivo."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving configuration: {str(e)}")


# Singleton instance
config = ConfigurationManager()
