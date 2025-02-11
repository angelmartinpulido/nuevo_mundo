import psutil
import time
import threading
from typing import List, Dict, Any, Optional
import logging

class LearningBalancer:
    def __init__(self):
        """Inicializa el balanceador de carga con aprendizaje"""
        self.lock = threading.Lock()
        self.learning_weights = {}
        self.operation_history = []
        self._setup_logging()
        
    def _setup_logging(self):
        """Configura el sistema de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('LearningBalancer')
        
    def get_system_resources(self) -> Dict[str, float]:
        """
        Obtiene los recursos actuales del sistema.
        
        Returns:
            Dict con información sobre CPU y memoria.
        """
        try:
            return {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent
            }
        except Exception as e:
            self.logger.error(f"Error al obtener recursos del sistema: {e}")
            return {'cpu_usage': 0.0, 'memory_usage': 0.0}
            
    def assign_task(self, nodes: List[Dict], task_load: int) -> Optional[Dict]:
        """
        Asigna una tarea al nodo más apropiado.
        
        Args:
            nodes: Lista de nodos disponibles
            task_load: Carga de la tarea a asignar
            
        Returns:
            Nodo seleccionado o None si no hay nodos disponibles
        """
        if not nodes or task_load < 0:
            raise ValueError("Nodos inválidos o carga negativa")
            
        with self.lock:
            # Encuentra el nodo menos cargado
            best_node = min(nodes, key=lambda x: x.get('current_load', float('inf')))
            
            if best_node['current_load'] + task_load <= best_node['capacity']:
                return best_node
                
        return None
        
    def balance_load(self, nodes: List[Dict]) -> Optional[int]:
        """
        Balancea la carga entre los nodos disponibles.
        
        Args:
            nodes: Lista de nodos para balancear
            
        Returns:
            Índice del nodo seleccionado o None
        """
        if not isinstance(nodes, list):
            raise TypeError("Se espera una lista de nodos")
            
        if not nodes:
            return None
            
        # Encuentra el nodo con menor carga
        min_load_index = 0
        min_load = float('inf')
        
        for i, node in enumerate(nodes):
            if node['load'] < min_load:
                min_load = node['load']
                min_load_index = i
                
        return min_load_index if min_load < 100 else None
        
    def calculate_metrics(self, operations: List[Dict]) -> Dict[str, float]:
        """
        Calcula métricas de rendimiento basadas en operaciones previas.
        
        Args:
            operations: Lista de operaciones realizadas
            
        Returns:
            Dict con métricas calculadas
        """
        if not operations:
            return {
                'success_rate': 0.0,
                'average_latency': 0.0,
                'total_operations': 0
            }
            
        success_count = sum(1 for op in operations if op['success'])
        total_latency = sum(op['latency'] for op in operations)
        
        return {
            'success_rate': success_count / len(operations),
            'average_latency': total_latency / len(operations),
            'total_operations': len(operations)
        }
        
    def check_nodes_health(self, nodes: List[Dict]) -> Dict[str, List]:
        """
        Verifica el estado de salud de los nodos.
        
        Args:
            nodes: Lista de nodos a verificar
            
        Returns:
            Dict con clasificación de nodos por estado
        """
        current_time = time.time()
        health_status = {
            'healthy_nodes': [],
            'degraded_nodes': [],
            'offline_nodes': []
        }
        
        for node in nodes:
            if node['status'] == 'healthy':
                health_status['healthy_nodes'].append(node)
            elif node['status'] == 'degraded':
                health_status['degraded_nodes'].append(node)
            else:
                health_status['offline_nodes'].append(node)
                
        return health_status
        
    def update_learning_weights(self, history: List[Dict]) -> Dict[str, float]:
        """
        Actualiza los pesos de aprendizaje basados en el historial.
        
        Args:
            history: Lista de decisiones previas y sus resultados
            
        Returns:
            Dict con pesos actualizados
        """
        weights = {}
        
        for entry in history:
            node = entry['decision']
            outcome = entry['outcome']
            
            if node not in weights:
                weights[node] = 1.0
                
            # Ajusta los pesos según el resultado
            if outcome == 'success':
                weights[node] *= 1.1  # Incrementa el peso en caso de éxito
            else:
                weights[node] *= 0.9  # Reduce el peso en caso de fallo
                
        self.learning_weights = weights
        return weights
        
    def is_thread_safe(self) -> bool:
        """
        Verifica si las operaciones son thread-safe.
        
        Returns:
            bool indicando si las operaciones son seguras
        """
        return hasattr(self, 'lock') and isinstance(self.lock, threading._RLock)