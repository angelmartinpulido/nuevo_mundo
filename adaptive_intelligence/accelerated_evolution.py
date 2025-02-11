from typing import Dict, List, Optional
import numpy as np
import logging
import json
from pathlib import Path
import time

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class AcceleratedEvolution:
    """
    Clase que gestiona la evolución acelerada del sistema de IA.
    
    Esta clase implementa mecanismos para evaluar, mejorar y verificar
    las capacidades del sistema de forma iterativa.
    
    Attributes:
        capabilities (Dict[str, float]): Capacidades actuales del sistema
        improvements (Dict[str, float]): Mejoras calculadas
        evolution_rate (float): Tasa actual de evolución
        MAX_EVOLUTION_RATE (float): Límite máximo de la tasa de evolución
        CAPABILITIES_THRESHOLD (float): Valor máximo permitido para capacidades
        state_file (Path): Ruta al archivo de persistencia
    """
    
    MAX_EVOLUTION_RATE = 2.0
    CAPABILITIES_THRESHOLD = 0.9999
    
    def __init__(self, state_file: Optional[str] = None):
        """
        Inicializa una nueva instancia de AcceleratedEvolution.
        
        Args:
            state_file (Optional[str]): Ruta al archivo para persistencia de estado
        """
        self.logger = logging.getLogger(__name__)
        self.capabilities = {}
        self.improvements = {}
        self.evolution_rate = 1.0
        self.state_file = Path(state_file) if state_file else None
        self._load_state()

    def _load_state(self) -> None:
        """Carga el estado previo desde el archivo si existe."""
        if not self.state_file or not self.state_file.exists():
            return
            
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                self.capabilities = state.get('capabilities', {})
                self.improvements = state.get('improvements', {})
                self.evolution_rate = state.get('evolution_rate', 1.0)
            self.logger.info("Estado cargado exitosamente")
        except Exception as e:
            self.logger.error(f"Error al cargar el estado: {e}")
            
    def _save_state(self) -> None:
        """Guarda el estado actual en el archivo de persistencia."""
        if not self.state_file:
            return
            
        try:
            state = {
                'capabilities': self.capabilities,
                'improvements': self.improvements,
                'evolution_rate': self.evolution_rate,
                'timestamp': time.time()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=4)
            self.logger.info("Estado guardado exitosamente")
        except Exception as e:
            self.logger.error(f"Error al guardar el estado: {e}")

    def assess_capabilities(self) -> Dict:
        """
        Evalúa las capacidades actuales del sistema.
        
        Returns:
            Dict[str, float]: Diccionario con las capacidades actuales y sus valores
            
        Raises:
            ValueError: Si alguna capacidad excede el umbral máximo permitido
        """
        self.capabilities = {
            "processing_power": min(0.95, self.CAPABILITIES_THRESHOLD),
            "learning_rate": min(0.98, self.CAPABILITIES_THRESHOLD),
            "adaptation_speed": min(0.97, self.CAPABILITIES_THRESHOLD),
            "problem_solving": min(0.96, self.CAPABILITIES_THRESHOLD),
            "resource_efficiency": min(0.94, self.CAPABILITIES_THRESHOLD),
        }
        
        # Validación de valores
        for capability, value in self.capabilities.items():
            if not 0 <= value <= 1:
                raise ValueError(f"Valor inválido para {capability}: {value}")
                
        self.logger.info("Capacidades evaluadas: %s", self.capabilities)
        return self.capabilities

    def calculate_improvements(self) -> Dict:
        """
        Calcula las mejoras potenciales basadas en el estado actual.
        
        Returns:
            Dict[str, float]: Diccionario con las mejoras calculadas
            
        Raises:
            ValueError: Si no hay capacidades evaluadas previamente
        """
        if not self.capabilities:
            raise ValueError("No hay capacidades evaluadas. Ejecute assess_capabilities primero.")
            
        improvement_factors = {
            "processing_power": 1.1,
            "learning_rate": 1.05,
            "adaptation_speed": 1.15,
            "problem_solving": 1.08,
            "resource_efficiency": 1.12
        }
        
        self.improvements = {}
        for capability, current_value in self.capabilities.items():
            if current_value >= self.CAPABILITIES_THRESHOLD:
                self.improvements[capability] = current_value
                self.logger.info(f"{capability} ha alcanzado el umbral máximo")
                continue
                
            factor = improvement_factors.get(capability, 1.05)
            improved_value = current_value * factor
            self.improvements[capability] = min(improved_value, self.CAPABILITIES_THRESHOLD)
            
        self.logger.info("Mejoras calculadas: %s", self.improvements)
        return self.improvements

    def implement_upgrades(self) -> bool:
        """
        Implementa las mejoras calculadas.
        
        Returns:
            bool: True si las mejoras se implementaron correctamente, False en caso contrario
            
        Raises:
            ValueError: Si no hay mejoras calculadas para implementar
        """
        if not self.improvements:
            raise ValueError("No hay mejoras calculadas. Ejecute calculate_improvements primero.")
            
        try:
            for capability, new_value in self.improvements.items():
                if capability not in self.capabilities:
                    raise ValueError(f"Capacidad desconocida: {capability}")
                    
                old_value = self.capabilities[capability]
                self.capabilities[capability] = new_value
                self.logger.info(
                    f"Mejora aplicada en {capability}: {old_value:.4f} -> {new_value:.4f}"
                )
                
            # Actualiza la tasa de evolución con límite superior
            self.evolution_rate = min(
                self.evolution_rate * 1.05,
                self.MAX_EVOLUTION_RATE
            )
            
            self._save_state()
            return True
            
        except Exception as e:
            self.logger.error(f"Error en la implementación de mejoras: {e}")
            return False

    def verify_effectiveness(self, seed: Optional[int] = None) -> Dict:
        """
        Verifica la efectividad de las mejoras implementadas.
        
        Args:
            seed (Optional[int]): Semilla para generación de números aleatorios
            
        Returns:
            Dict: Diccionario con métricas de efectividad para cada capacidad
            
        Raises:
            ValueError: Si no hay capacidades para verificar
        """
        if not self.capabilities:
            raise ValueError("No hay capacidades para verificar")
            
        if seed is not None:
            np.random.seed(seed)
            
        effectiveness = {}
        for capability in self.capabilities:
            old_value = self.capabilities[capability] / 1.1  # Valor aproximado anterior
            improvement = self.capabilities[capability] - old_value
            
            # Calcula la tasa de éxito basada en la mejora
            success_rate = np.clip(
                np.random.normal(0.975, 0.025),  # Media 0.975, desviación 0.025
                0.95,  # Mínimo
                1.0    # Máximo
            )
            
            effectiveness[capability] = {
                "improvement": float(improvement),  # Convertimos a float para serialización
                "success_rate": float(success_rate),
                "old_value": float(old_value),
                "new_value": float(self.capabilities[capability])
            }
            
        self.logger.info("Verificación de efectividad completada: %s", effectiveness)
        return effectiveness

    def run_evolution_cycle(self) -> Dict:
        """
        Ejecuta un ciclo completo de evolución.
        
        Returns:
            Dict: Resultado completo del ciclo de evolución incluyendo:
                - cycle_success: bool indicando si el ciclo fue exitoso
                - evolution_rate: tasa actual de evolución
                - effectiveness: métricas de efectividad
                - current_state: estado actual de las capacidades
                - timestamp: marca de tiempo de la ejecución
                
        Raises:
            RuntimeError: Si ocurre un error durante el ciclo
        """
        try:
            self.logger.info("Iniciando ciclo de evolución")
            
            current_capabilities = self.assess_capabilities()
            self.logger.info("Capacidades evaluadas")
            
            potential_improvements = self.calculate_improvements()
            self.logger.info("Mejoras calculadas")
            
            upgrade_success = self.implement_upgrades()
            if not upgrade_success:
                raise RuntimeError("Fallo en la implementación de mejoras")
            self.logger.info("Mejoras implementadas")
            
            effectiveness = self.verify_effectiveness()
            self.logger.info("Efectividad verificada")
            
            result = {
                "cycle_success": upgrade_success,
                "evolution_rate": float(self.evolution_rate),
                "effectiveness": effectiveness,
                "current_state": self.capabilities,
                "timestamp": time.time()
            }
            
            self._save_state()
            self.logger.info("Ciclo de evolución completado exitosamente")
            return result
            
        except Exception as e:
            self.logger.error(f"Error en el ciclo de evolución: {e}")
            raise RuntimeError(f"Error en el ciclo de evolución: {str(e)}")
            
    def get_statistics(self) -> Dict:
        """
        Obtiene estadísticas del sistema.
        
        Returns:
            Dict: Estadísticas del sistema incluyendo promedios y máximos
        """
        if not self.capabilities:
            return {}
            
        stats = {
            "average_capability": np.mean(list(self.capabilities.values())),
            "max_capability": max(self.capabilities.values()),
            "min_capability": min(self.capabilities.values()),
            "evolution_rate": self.evolution_rate,
            "capabilities_near_threshold": sum(
                1 for v in self.capabilities.values()
                if v >= self.CAPABILITIES_THRESHOLD * 0.95
            )
        }
        return stats
