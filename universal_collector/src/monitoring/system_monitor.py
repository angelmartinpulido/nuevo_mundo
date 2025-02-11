"""
Sistema de monitoreo que supervisa recursos y rendimiento del sistema.
"""
import psutil
import time
import threading
from typing import Dict, List, Optional, Callable
from datetime import datetime
from ..core.config import config
from ..core.logging_manager import logger_manager
from ..core.error_handler import handle_errors, SystemError

logger = logger_manager.get_logger(__name__)


class MetricCollector:
    """Recolector de métricas del sistema."""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {
            "cpu": [],
            "memory": [],
            "disk": [],
            "network": [],
        }
        self.max_history = 1000

    def collect_cpu(self) -> float:
        """Recolecta uso de CPU."""
        return psutil.cpu_percent(interval=1)

    def collect_memory(self) -> Dict[str, float]:
        """Recolecta uso de memoria."""
        mem = psutil.virtual_memory()
        return {
            "total": mem.total / (1024 * 1024 * 1024),  # GB
            "used": mem.used / (1024 * 1024 * 1024),  # GB
            "percent": mem.percent,
        }

    def collect_disk(self) -> Dict[str, float]:
        """Recolecta uso de disco."""
        disk = psutil.disk_usage("/")
        return {
            "total": disk.total / (1024 * 1024 * 1024),  # GB
            "used": disk.used / (1024 * 1024 * 1024),  # GB
            "percent": disk.percent,
        }

    def collect_network(self) -> Dict[str, float]:
        """Recolecta estadísticas de red."""
        net = psutil.net_io_counters()
        return {
            "bytes_sent": net.bytes_sent / (1024 * 1024),  # MB
            "bytes_recv": net.bytes_recv / (1024 * 1024),  # MB
            "packets_sent": net.packets_sent,
            "packets_recv": net.packets_recv,
        }

    def collect_all(self) -> Dict[str, Any]:
        """Recolecta todas las métricas."""
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu": self.collect_cpu(),
            "memory": self.collect_memory(),
            "disk": self.collect_disk(),
            "network": self.collect_network(),
        }


class SystemMonitor:
    """Monitor principal del sistema."""

    def __init__(self):
        self.collector = MetricCollector()
        self.monitoring = False
        self.interval = config.get("monitoring.interval", 60)
        self.callbacks: List[Callable] = []
        self._monitor_thread: Optional[threading.Thread] = None

    @handle_errors
    def start(self) -> None:
        """Inicia el monitoreo en un hilo separado."""
        if self.monitoring:
            logger.warning("Monitoring already started")
            return

        self.monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        logger.info("System monitoring started")

    def stop(self) -> None:
        """Detiene el monitoreo."""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("System monitoring stopped")

    def register_callback(self, callback: Callable) -> None:
        """
        Registra una función callback para notificaciones.

        Args:
            callback: Función a llamar con las métricas recolectadas
        """
        self.callbacks.append(callback)

    def _monitor_loop(self) -> None:
        """Loop principal de monitoreo."""
        while self.monitoring:
            try:
                metrics = self.collector.collect_all()
                self._process_metrics(metrics)
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")

    def _process_metrics(self, metrics: Dict) -> None:
        """
        Procesa las métricas recolectadas.

        Args:
            metrics: Diccionario con las métricas
        """
        # Verificar umbrales
        self._check_thresholds(metrics)

        # Notificar callbacks
        for callback in self.callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"Error in monitoring callback: {str(e)}")

    def _check_thresholds(self, metrics: Dict) -> None:
        """
        Verifica si las métricas superan los umbrales configurados.

        Args:
            metrics: Diccionario con las métricas
        """
        # CPU
        if metrics["cpu"] > 90:
            logger.warning(f"High CPU usage: {metrics['cpu']}%")

        # Memoria
        if metrics["memory"]["percent"] > 90:
            logger.warning(f"High memory usage: {metrics['memory']['percent']}%")

        # Disco
        if metrics["disk"]["percent"] > 90:
            logger.warning(f"High disk usage: {metrics['disk']['percent']}%")


# Singleton instance
system_monitor = SystemMonitor()
