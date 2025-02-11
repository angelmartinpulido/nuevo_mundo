import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import time
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from datetime import datetime


@dataclass
class MonitoringMetrics:
    system_load: float
    memory_usage: float
    network_traffic: float
    security_status: str
    error_rate: float
    response_time: float
    timestamp: float


class AdvancedMonitoring:
    def __init__(self):
        self._setup_logging()
        self.metrics_history = []
        self.alert_history = []
        self.monitoring_rules = {}
        self.alert_thresholds = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._initialize_monitoring()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename="advanced_monitoring.log",
        )
        self.logger = logging.getLogger("AdvancedMonitoring")

    def _initialize_monitoring(self):
        """Inicializa el sistema de monitoreo"""
        self._load_monitoring_rules()
        self._setup_alert_thresholds()
        self._initialize_metrics_collection()

    async def start_monitoring(self):
        """Inicia el monitoreo continuo"""
        try:
            while True:
                # Recolección de métricas
                metrics = await self._collect_metrics()

                # Análisis de métricas
                analysis = await self._analyze_metrics(metrics)

                # Verificación de alertas
                alerts = await self._check_alerts(analysis)

                # Almacenamiento de datos
                await self._store_monitoring_data(metrics, analysis, alerts)

                # Espera antes de la siguiente recolección
                await asyncio.sleep(60)  # Intervalo de monitoreo
        except Exception as e:
            self.logger.error(f"Error en monitoreo continuo: {str(e)}")
            raise

    async def _collect_metrics(self) -> MonitoringMetrics:
        """Recolecta métricas del sistema"""
        try:
            return MonitoringMetrics(
                system_load=await self._measure_system_load(),
                memory_usage=await self._measure_memory_usage(),
                network_traffic=await self._measure_network_traffic(),
                security_status=await self._check_security_status(),
                error_rate=await self._calculate_error_rate(),
                response_time=await self._measure_response_time(),
                timestamp=time.time(),
            )
        except Exception as e:
            self.logger.error(f"Error en recolección de métricas: {str(e)}")
            raise

    async def _analyze_metrics(self, metrics: MonitoringMetrics) -> Dict[str, Any]:
        """Analiza las métricas recolectadas"""
        try:
            return {
                "performance_score": self._calculate_performance_score(metrics),
                "health_status": self._determine_health_status(metrics),
                "trend_analysis": await self._analyze_trends(metrics),
                "anomaly_detection": await self._detect_anomalies(metrics),
                "resource_prediction": await self._predict_resource_usage(metrics),
            }
        except Exception as e:
            self.logger.error(f"Error en análisis de métricas: {str(e)}")
            raise

    def _calculate_performance_score(self, metrics: MonitoringMetrics) -> float:
        """Calcula puntuación de rendimiento"""
        try:
            weights = {
                "system_load": 0.3,
                "memory_usage": 0.2,
                "network_traffic": 0.2,
                "error_rate": 0.15,
                "response_time": 0.15,
            }

            score = (
                (1 - metrics.system_load) * weights["system_load"]
                + (1 - metrics.memory_usage) * weights["memory_usage"]
                + (1 - metrics.network_traffic) * weights["network_traffic"]
                + (1 - metrics.error_rate) * weights["error_rate"]
                + (1 / metrics.response_time) * weights["response_time"]
            )

            return min(max(score, 0), 1)  # Normalizar entre 0 y 1
        except Exception as e:
            self.logger.error(f"Error en cálculo de rendimiento: {str(e)}")
            return 0.0

    def _determine_health_status(self, metrics: MonitoringMetrics) -> str:
        """Determina el estado de salud del sistema"""
        try:
            if metrics.error_rate > 0.1 or metrics.system_load > 0.9:
                return "CRITICAL"
            elif metrics.error_rate > 0.05 or metrics.system_load > 0.8:
                return "WARNING"
            elif metrics.error_rate > 0.01 or metrics.system_load > 0.7:
                return "ATTENTION"
            else:
                return "HEALTHY"
        except Exception as e:
            self.logger.error(f"Error en determinación de salud: {str(e)}")
            return "UNKNOWN"

    async def _analyze_trends(self, metrics: MonitoringMetrics) -> Dict[str, Any]:
        """Analiza tendencias en las métricas"""
        try:
            # Obtener historial reciente
            recent_metrics = self.metrics_history[-100:]

            # Calcular tendencias
            trends = {
                "system_load": self._calculate_trend(
                    [m.system_load for m in recent_metrics]
                ),
                "memory_usage": self._calculate_trend(
                    [m.memory_usage for m in recent_metrics]
                ),
                "error_rate": self._calculate_trend(
                    [m.error_rate for m in recent_metrics]
                ),
            }

            return trends
        except Exception as e:
            self.logger.error(f"Error en análisis de tendencias: {str(e)}")
            return {}

    def _calculate_trend(self, values: List[float]) -> str:
        """Calcula la tendencia de una serie de valores"""
        try:
            if len(values) < 2:
                return "STABLE"

            slope = np.polyfit(range(len(values)), values, 1)[0]

            if slope > 0.01:
                return "INCREASING"
            elif slope < -0.01:
                return "DECREASING"
            else:
                return "STABLE"
        except Exception as e:
            self.logger.error(f"Error en cálculo de tendencia: {str(e)}")
            return "UNKNOWN"

    async def _detect_anomalies(
        self, metrics: MonitoringMetrics
    ) -> List[Dict[str, Any]]:
        """Detecta anomalías en las métricas"""
        try:
            anomalies = []

            # Verificar umbrales
            if metrics.system_load > self.alert_thresholds.get("system_load", 0.9):
                anomalies.append(
                    {
                        "type": "high_system_load",
                        "value": metrics.system_load,
                        "threshold": self.alert_thresholds["system_load"],
                    }
                )

            if metrics.error_rate > self.alert_thresholds.get("error_rate", 0.1):
                anomalies.append(
                    {
                        "type": "high_error_rate",
                        "value": metrics.error_rate,
                        "threshold": self.alert_thresholds["error_rate"],
                    }
                )

            return anomalies
        except Exception as e:
            self.logger.error(f"Error en detección de anomalías: {str(e)}")
            return []

    async def _predict_resource_usage(
        self, metrics: MonitoringMetrics
    ) -> Dict[str, float]:
        """Predice uso futuro de recursos"""
        try:
            # Obtener historial reciente
            recent_metrics = self.metrics_history[-100:]

            # Realizar predicciones
            predictions = {
                "system_load_1h": self._predict_value(
                    [m.system_load for m in recent_metrics]
                ),
                "memory_usage_1h": self._predict_value(
                    [m.memory_usage for m in recent_metrics]
                ),
                "network_traffic_1h": self._predict_value(
                    [m.network_traffic for m in recent_metrics]
                ),
            }

            return predictions
        except Exception as e:
            self.logger.error(f"Error en predicción de recursos: {str(e)}")
            return {}

    def _predict_value(self, values: List[float]) -> float:
        """Predice el próximo valor en una serie"""
        try:
            if len(values) < 2:
                return values[-1] if values else 0.0

            # Usar regresión lineal simple para predicción
            x = np.array(range(len(values)))
            y = np.array(values)
            coefficients = np.polyfit(x, y, 1)

            # Predecir próximo valor
            next_x = len(values)
            prediction = coefficients[0] * next_x + coefficients[1]

            return max(min(prediction, 1.0), 0.0)  # Normalizar entre 0 y 1
        except Exception as e:
            self.logger.error(f"Error en predicción de valor: {str(e)}")
            return 0.0

    async def _check_alerts(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Verifica y genera alertas basadas en el análisis"""
        try:
            alerts = []

            # Verificar estado de salud
            if analysis["health_status"] in ["CRITICAL", "WARNING"]:
                alerts.append(
                    {
                        "type": "health_alert",
                        "severity": analysis["health_status"],
                        "message": f"Sistema en estado {analysis['health_status']}",
                        "timestamp": time.time(),
                    }
                )

            # Verificar anomalías
            for anomaly in analysis["anomaly_detection"]:
                alerts.append(
                    {
                        "type": "anomaly_alert",
                        "severity": "WARNING",
                        "message": f"Anomalía detectada: {anomaly['type']}",
                        "details": anomaly,
                        "timestamp": time.time(),
                    }
                )

            return alerts
        except Exception as e:
            self.logger.error(f"Error en verificación de alertas: {str(e)}")
            return []

    async def _store_monitoring_data(
        self,
        metrics: MonitoringMetrics,
        analysis: Dict[str, Any],
        alerts: List[Dict[str, Any]],
    ):
        """Almacena datos de monitoreo"""
        try:
            # Almacenar métricas
            self.metrics_history.append(metrics)

            # Almacenar alertas
            self.alert_history.extend(alerts)

            # Limpiar históricos antiguos
            self._cleanup_old_data()
        except Exception as e:
            self.logger.error(f"Error en almacenamiento de datos: {str(e)}")

    def _cleanup_old_data(self):
        """Limpia datos antiguos"""
        try:
            # Mantener solo últimas 24 horas de métricas
            cutoff_time = time.time() - (24 * 60 * 60)

            self.metrics_history = [
                m for m in self.metrics_history if m.timestamp > cutoff_time
            ]

            self.alert_history = [
                a for a in self.alert_history if a["timestamp"] > cutoff_time
            ]
        except Exception as e:
            self.logger.error(f"Error en limpieza de datos: {str(e)}")

    def get_monitoring_report(self) -> Dict[str, Any]:
        """Genera reporte de monitoreo"""
        try:
            return {
                "current_metrics": self.metrics_history[-1]
                if self.metrics_history
                else None,
                "alerts_last_24h": len(self.alert_history),
                "system_health": self._determine_health_status(self.metrics_history[-1])
                if self.metrics_history
                else "UNKNOWN",
                "generated_at": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error generando reporte: {str(e)}")
            return {}
