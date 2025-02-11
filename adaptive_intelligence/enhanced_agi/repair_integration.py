"""
Integration Module for Self-Repair System
"""

from typing import Dict, List, Any, Optional
import logging
import threading
import time
from .self_repair_system import (
    SelfRepairSystem,
    VulnerabilityReport,
    RepairReport,
    VulnerabilityLevel,
)


class RepairIntegration:
    """Integrador del sistema de auto-reparación con el sistema AGI principal"""

    def __init__(self):
        self.repair_system = SelfRepairSystem()
        self.active = False
        self.monitor_thread = None
        self.repair_callbacks = []
        self.vulnerability_callbacks = []

    def initialize(self):
        """Inicializa el sistema de reparación"""
        try:
            # Configurar logging
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

            # Iniciar sistema de reparación
            self.repair_system.start()

            # Iniciar monitoreo
            self._start_monitoring()

            self.active = True
            logging.info("Sistema de reparación integrado iniciado correctamente")

        except Exception as e:
            logging.error(f"Error al inicializar sistema de reparación: {str(e)}")
            raise

    def shutdown(self):
        """Detiene el sistema de reparación"""
        try:
            self.active = False
            self.repair_system.stop()

            if self.monitor_thread:
                self.monitor_thread.join()

            logging.info("Sistema de reparación detenido correctamente")

        except Exception as e:
            logging.error(f"Error al detener sistema de reparación: {str(e)}")
            raise

    def _start_monitoring(self):
        """Inicia el monitoreo del sistema"""

        def monitor_loop():
            while self.active:
                try:
                    # Obtener estado actual
                    status = self.repair_system.get_status()

                    # Verificar vulnerabilidades críticas
                    if status["vulnerabilities_detected"] > 0:
                        self._handle_vulnerabilities()

                    # Verificar reparaciones pendientes
                    if status["repairs_pending"] > 0:
                        self._handle_repairs()

                    # Notificar callbacks
                    self._notify_status(status)

                    time.sleep(5)  # Intervalo de monitoreo

                except Exception as e:
                    logging.error(f"Error en monitoreo: {str(e)}")
                    time.sleep(1)  # Esperar antes de reintentar

        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def _handle_vulnerabilities(self):
        """Maneja las vulnerabilidades detectadas"""
        try:
            # Forzar escaneo
            vulnerabilities = self.repair_system.force_scan()

            # Notificar vulnerabilidades
            for callback in self.vulnerability_callbacks:
                callback(vulnerabilities)

            # Priorizar vulnerabilidades críticas
            critical_vulns = [
                v for v in vulnerabilities if v.level == VulnerabilityLevel.CRITICAL
            ]

            # Reparar vulnerabilidades críticas inmediatamente
            for vuln in critical_vulns:
                self.repair_system.force_repair(vuln)

        except Exception as e:
            logging.error(f"Error al manejar vulnerabilidades: {str(e)}")

    def _handle_repairs(self):
        """Maneja las reparaciones pendientes"""
        try:
            # Obtener historial de reparaciones
            repair_history = self.repair_system.get_repair_history()

            # Notificar reparaciones
            for callback in self.repair_callbacks:
                callback(repair_history)

            # Analizar resultados
            self._analyze_repair_results(repair_history)

        except Exception as e:
            logging.error(f"Error al manejar reparaciones: {str(e)}")

    def _analyze_repair_results(self, repair_history: List[RepairReport]):
        """Analiza los resultados de las reparaciones"""
        try:
            # Calcular estadísticas
            total_repairs = len(repair_history)
            successful_repairs = len(
                [r for r in repair_history if r.success_rate > 0.8]
            )
            failed_repairs = total_repairs - successful_repairs

            # Registrar métricas
            logging.info(
                f"""
                Análisis de reparaciones:
                - Total: {total_repairs}
                - Exitosas: {successful_repairs}
                - Fallidas: {failed_repairs}
                - Tasa de éxito: {(successful_repairs/total_repairs)*100 if total_repairs > 0 else 0}%
            """
            )

            # Identificar patrones de fallo
            if failed_repairs > 0:
                self._analyze_failure_patterns(repair_history)

        except Exception as e:
            logging.error(f"Error al analizar resultados: {str(e)}")

    def _analyze_failure_patterns(self, repair_history: List[RepairReport]):
        """Analiza patrones en reparaciones fallidas"""
        try:
            # Agrupar por tipo de vulnerabilidad
            failure_patterns = {}

            for report in repair_history:
                if report.success_rate <= 0.8:
                    vuln_type = report.vulnerability.component
                    if vuln_type not in failure_patterns:
                        failure_patterns[vuln_type] = 0
                    failure_patterns[vuln_type] += 1

            # Identificar componentes problemáticos
            problematic_components = {
                comp: count
                for comp, count in failure_patterns.items()
                if count > 3  # Umbral de fallos
            }

            if problematic_components:
                logging.warning(
                    f"Componentes con fallos recurrentes: {problematic_components}"
                )

        except Exception as e:
            logging.error(f"Error al analizar patrones de fallo: {str(e)}")

    def _notify_status(self, status: Dict[str, Any]):
        """Notifica el estado actual del sistema"""
        try:
            logging.info(f"Estado actual del sistema de reparación: {status}")

            # Verificar métricas críticas
            if status["vulnerabilities_detected"] > 10:
                logging.warning("Alto número de vulnerabilidades detectadas")

            if status["repairs_pending"] > 5:
                logging.warning("Alto número de reparaciones pendientes")

        except Exception as e:
            logging.error(f"Error al notificar estado: {str(e)}")

    def register_repair_callback(self, callback: callable):
        """Registra un callback para notificaciones de reparación"""
        self.repair_callbacks.append(callback)

    def register_vulnerability_callback(self, callback: callable):
        """Registra un callback para notificaciones de vulnerabilidades"""
        self.vulnerability_callbacks.append(callback)

    def get_system_health(self) -> Dict[str, Any]:
        """Obtiene el estado de salud del sistema"""
        try:
            status = self.repair_system.get_status()
            repair_history = self.repair_system.get_repair_history()

            return {
                "status": status,
                "total_repairs": len(repair_history),
                "active": self.active,
                "last_vulnerabilities": len(self.repair_system.force_scan()),
                "health_score": self._calculate_health_score(status, repair_history),
            }
        except Exception as e:
            logging.error(f"Error al obtener salud del sistema: {str(e)}")
            return {"error": str(e)}

    def _calculate_health_score(
        self, status: Dict[str, Any], repair_history: List[RepairReport]
    ) -> float:
        """Calcula una puntuación de salud del sistema"""
        try:
            # Factores de puntuación
            vulnerability_factor = 1.0 - (status["vulnerabilities_detected"] * 0.1)
            repair_factor = 1.0 - (status["repairs_pending"] * 0.05)

            # Historial de reparaciones
            if repair_history:
                success_rate = sum(r.success_rate for r in repair_history) / len(
                    repair_history
                )
            else:
                success_rate = 1.0

            # Calcular puntuación final
            health_score = (
                vulnerability_factor * 0.4 + repair_factor * 0.3 + success_rate * 0.3
            )

            return max(0.0, min(1.0, health_score))  # Normalizar entre 0 y 1

        except Exception as e:
            logging.error(f"Error al calcular puntuación de salud: {str(e)}")
            return 0.0
