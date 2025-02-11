"""
Feedback System for Improved Learning Accuracy
"""
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import numpy as np
import logging
from dataclasses import dataclass, field
from functools import lru_cache

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class FeedbackConfig:
    """Configuración global para el sistema de retroalimentación"""
    MAX_HISTORY_LENGTH: int = 1000
    ADAPTATION_WINDOW_SIZE: int = 50
    IMPROVEMENT_THRESHOLD: float = 0.01
    TREND_WINDOW_SIZE: int = 10

@dataclass
class FeedbackEntry:
    """Entrada de retroalimentación estructurada"""
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ''
    prediction: Any = None
    actual: Any = None
    context: Dict[str, Any] = field(default_factory=dict)
    accuracy: float = 0.0
    adaptation_score: float = 0.0

class FeedbackSystem:
    """
    Sistema avanzado de retroalimentación con métricas detalladas
    
    Características:
    - Registro de retroalimentación con múltiples tipos de datos
    - Cálculo de precisión adaptativo
    - Seguimiento de métricas de aprendizaje
    - Análisis de tendencias
    """
    
    def __init__(self, config: Optional[FeedbackConfig] = None):
        """
        Inicializa el sistema de retroalimentación
        
        Args:
            config (Optional[FeedbackConfig]): Configuración personalizada
        """
        self.config = config or FeedbackConfig()
        self.feedback_history: List[FeedbackEntry] = []
        self.learning_metrics: Dict[str, Dict[str, Any]] = {}
        
    @lru_cache(maxsize=128)
    def _calculate_accuracy(
        self, 
        prediction: Union[int, float, List, np.ndarray, bool], 
        actual: Union[int, float, List, np.ndarray, bool]
    ) -> float:
        """
        Calcula la precisión de manera robusta para diferentes tipos de datos
        
        Args:
            prediction: Valor predicho
            actual: Valor real
        
        Returns:
            Precisión como valor flotante entre 0 y 1
        """
        try:
            # Manejo de tipos numéricos
            if isinstance(prediction, (int, float)) and isinstance(actual, (int, float)):
                max_val = max(abs(prediction), abs(actual))
                return 1.0 - min(1.0, abs(prediction - actual) / max_val) if max_val > 0 else 1.0
            
            # Manejo de listas y arrays
            if isinstance(prediction, (list, np.ndarray)) and isinstance(actual, (list, np.ndarray)):
                pred_array = np.array(prediction)
                act_array = np.array(actual)
                return float(np.mean(pred_array == act_array))
            
            # Comparación básica para otros tipos
            return float(prediction == actual)
        
        except Exception as e:
            logger.warning(f"Error calculando precisión: {e}")
            return 0.0
        
    def record_feedback(
        self, 
        source: str, 
        prediction: Any, 
        actual: Any, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Registra una entrada de retroalimentación
        
        Args:
            source: Fuente de la retroalimentación
            prediction: Valor predicho
            actual: Valor real
            context: Contexto adicional
        
        Returns:
            Diccionario con detalles de la retroalimentación
        """
        context = context or {}
        
        # Calcular precisión
        accuracy = self._calculate_accuracy(prediction, actual)
        
        # Crear entrada de retroalimentación
        feedback_entry = FeedbackEntry(
            source=source,
            prediction=prediction,
            actual=actual,
            context=context,
            accuracy=accuracy
        )
        
        # Calcular puntuación de adaptación
        feedback_entry.adaptation_score = self._calculate_adaptation_score(feedback_entry)
        
        # Almacenar entrada
        self.feedback_history.append(feedback_entry)
        
        # Limitar longitud del historial
        if len(self.feedback_history) > self.config.MAX_HISTORY_LENGTH:
            self.feedback_history.pop(0)
        
        # Actualizar métricas de aprendizaje
        self._update_learning_metrics(feedback_entry)
        
        return {
            "timestamp": feedback_entry.timestamp,
            "source": feedback_entry.source,
            "prediction": feedback_entry.prediction,
            "actual": feedback_entry.actual,
            "context": feedback_entry.context,
            "accuracy": feedback_entry.accuracy,
            "adaptation_score": feedback_entry.adaptation_score
        }
        
    def _calculate_adaptation_score(self, feedback: FeedbackEntry) -> float:
        """
        Calcula la puntuación de adaptación basada en el historial reciente
        
        Args:
            feedback: Entrada de retroalimentación
        
        Returns:
            Puntuación de adaptación
        """
        # Obtener retroalimentaciones recientes de la misma fuente
        recent_feedbacks = [
            f for f in self.feedback_history[-self.config.ADAPTATION_WINDOW_SIZE:] 
            if f.source == feedback.source
        ]
        
        if not recent_feedbacks:
            return 0.5
        
        # Calcular promedio ponderado de precisión
        accuracies = [f.accuracy for f in recent_feedbacks]
        weights = np.exp(np.linspace(-1, 0, len(accuracies)))
        weights /= weights.sum()
        
        return float(np.average(accuracies, weights=weights))
        
    def _update_learning_metrics(self, feedback: FeedbackEntry):
        """
        Actualiza las métricas de aprendizaje para una fuente específica
        
        Args:
            feedback: Entrada de retroalimentación
        """
        source = feedback.source
        
        if source not in self.learning_metrics:
            self.learning_metrics[source] = {
                "total_samples": 0,
                "accuracy_history": [],
                "adaptation_history": [],
                "improvement_rate": 0.0
            }
        
        metrics = self.learning_metrics[source]
        metrics["total_samples"] += 1
        metrics["accuracy_history"].append(feedback.accuracy)
        metrics["adaptation_history"].append(feedback.adaptation_score)
        
        # Limitar longitud del historial
        metrics["accuracy_history"] = metrics["accuracy_history"][-self.config.MAX_HISTORY_LENGTH:]
        metrics["adaptation_history"] = metrics["adaptation_history"][-self.config.MAX_HISTORY_LENGTH:]
        
        # Calcular tasa de mejora
        if len(metrics["accuracy_history"]) >= 2:
            recent_window = metrics["accuracy_history"][-self.config.TREND_WINDOW_SIZE:]
            metrics["improvement_rate"] = np.mean(
                recent_window[len(recent_window) // 2 :]
            ) - np.mean(recent_window[: len(recent_window) // 2])
        
    def get_learning_status(self, source: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene el estado de aprendizaje
        
        Args:
            source: Fuente específica (opcional)
        
        Returns:
            Métricas de aprendizaje
        """
        if source and source in self.learning_metrics:
            return self._get_source_metrics(source)
        
        # Métricas generales
        overall_metrics = {
            "total_sources": len(self.learning_metrics),
            "total_feedback": len(self.feedback_history),
            "average_accuracy": 0.0,
            "average_adaptation": 0.0,
            "sources": {}
        }
        
        for src, metrics in self.learning_metrics.items():
            source_metrics = self._get_source_metrics(src)
            overall_metrics["sources"][src] = source_metrics
            overall_metrics["average_accuracy"] += source_metrics["current_accuracy"]
            overall_metrics["average_adaptation"] += source_metrics["current_adaptation"]
        
        if overall_metrics["total_sources"] > 0:
            overall_metrics["average_accuracy"] /= overall_metrics["total_sources"]
            overall_metrics["average_adaptation"] /= overall_metrics["total_sources"]
        
        return overall_metrics
        
    def _get_source_metrics(self, source: str) -> Dict[str, Any]:
        """
        Obtiene métricas detalladas para una fuente específica
        
        Args:
            source: Fuente de retroalimentación
        
        Returns:
            Métricas detalladas
        """
        metrics = self.learning_metrics[source]
        return {
            "total_samples": metrics["total_samples"],
            "current_accuracy": metrics["accuracy_history"][-1] 
                if metrics["accuracy_history"] else 0.0,
            "current_adaptation": metrics["adaptation_history"][-1] 
                if metrics["adaptation_history"] else 0.0,
            "improvement_rate": metrics["improvement_rate"],
            "trend": self._calculate_trend(metrics["accuracy_history"])
        }
        
    def _calculate_trend(
        self, 
        history: List[float], 
        window_size: Optional[int] = None
    ) -> str:
        """
        Calcula la tendencia basada en el historial
        
        Args:
            history: Historial de precisión
            window_size: Tamaño de ventana para análisis
        
        Returns:
            Cadena representando la tendencia
        """
        window_size = window_size or self.config.TREND_WINDOW_SIZE
        
        if len(history) < window_size:
            return "insufficient_data"
        
        recent = history[-window_size:]
        slope = np.polyfit(range(window_size), recent, 1)[0]
        
        if slope > self.config.IMPROVEMENT_THRESHOLD:
            return "improving"
        elif slope < -self.config.IMPROVEMENT_THRESHOLD:
            return "declining"
        else:
            return "stable"

# Ejemplo de uso
def main():
    # Crear sistema de retroalimentación
    feedback_system = FeedbackSystem()
    
    # Simular algunas entradas de retroalimentación
    sources = ["model_a", "model_b", "model_c"]
    
    for source in sources:
        for i in range(100):
            prediction = i + np.random.normal(0, 0.1)
            actual = i
            
            feedback_system.record_feedback(
                source=source,
                prediction=prediction,
                actual=actual,
                context={"iteration": i}
            )
    
    # Obtener estado de aprendizaje
    overall_status = feedback_system.get_learning_status()
    print("Estado general de aprendizaje:", overall_status)
    
    # Obtener estado de una fuente específica
    source_status = feedback_system.get_learning_status("model_a")
    print("Estado de aprendizaje de model_a:", source_status)

if __name__ == "__main__":
    main()