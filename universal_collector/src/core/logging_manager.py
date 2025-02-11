"""
Sistema de logging centralizado con capacidades avanzadas.
"""
import logging
import sys
import os
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler
from .config import config


class LoggingManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.log_dir = "logs"
            self.max_size = 10 * 1024 * 1024  # 10MB
            self.backup_count = 5
            self.setup_logging()
            self.initialized = True

    def setup_logging(self) -> None:
        """Configura el sistema de logging con rotación de archivos y formato personalizado."""
        os.makedirs(self.log_dir, exist_ok=True)

        # Crear el formateador
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Configurar el logger raíz
        root_logger = logging.getLogger()
        root_logger.setLevel(config.get("system.log_level", "INFO"))

        # Limpiar handlers existentes
        root_logger.handlers = []

        # Handler para archivo
        log_file = os.path.join(
            self.log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log"
        )
        file_handler = RotatingFileHandler(
            log_file, maxBytes=self.max_size, backupCount=self.backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """
        Obtiene un logger configurado para un módulo específico.

        Args:
            name: Nombre del módulo/componente

        Returns:
            Logger configurado
        """
        return logging.getLogger(name)

    def set_level(self, level: str) -> None:
        """
        Cambia el nivel de logging en tiempo de ejecución.

        Args:
            level: Nivel de logging ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        numeric_level = getattr(logging, level.upper(), None)
        if isinstance(numeric_level, int):
            logging.getLogger().setLevel(numeric_level)

    def add_file_handler(self, filename: str, level: Optional[str] = None) -> None:
        """
        Añade un handler adicional para un archivo específico.

        Args:
            filename: Nombre del archivo de log
            level: Nivel de logging opcional para este handler
        """
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        file_path = os.path.join(self.log_dir, filename)
        handler = RotatingFileHandler(
            file_path, maxBytes=self.max_size, backupCount=self.backup_count
        )

        if level:
            numeric_level = getattr(logging, level.upper(), None)
            if isinstance(numeric_level, int):
                handler.setLevel(numeric_level)

        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)


# Singleton instance
logger_manager = LoggingManager()
