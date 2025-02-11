"""
Sistema de Distribución Adaptativa de Recursos
Optimización dinámica de distribución de recursos a medida que el sistema crece
"""

import asyncio
import math
import random
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
import numpy as np

@dataclass
class ResourceFragment:
    """Fragmento de recurso con información de distribución"""
    id: str
    type: str
    priority: int
    size: float
    dependencies: Set[str] = field(default_factory=set)
    distribution_metadata: Dict[str, Any] = field(default_factory=dict)

class AdaptiveResourceDistributor:
    """
    Distribuidor de recursos con escalabilidad adaptativa
    
    Principios de diseño:
    1. A más nodos, menor % de distribución
    2. Mantener acceso rápido y redundancia
    3. Minimizar sobrecarga de red
    4. Optimizar latencia
    """
    
    def __init__(self, initial_nodes: int = 10):
        # Registro de recursos
        self.resources: Dict[str, ResourceFragment] = {}
        
        # Registro de nodos
        self.nodes: Set[str] = set()
        
        # Métricas de distribución
        self.distribution_metrics = {
            'total_nodes': initial_nodes,
            'resource_distribution': {},
            'access_latency': {},
            'network_overhead': {}
        }
        
        # Parámetros de distribución adaptativa
        self.distribution_strategy = {
            'base_percentage': 0.3,  # Porcentaje base de distribución
            'min_percentage': 0.05,  # Porcentaje mínimo
            'scaling_factor': 0.95   # Factor de reducción al escalar
        }
    
    def add_node(self, node_id: str):
        """Añadir un nuevo nodo al sistema"""
        self.nodes.add(node_id)
        self.distribution_metrics['total_nodes'] += 1
        
        # Recalcular distribución de recursos
        self._recalculate_resource_distribution()
    
    def _recalculate_resource_distribution(self):
        """
        Recalcula la distribución de recursos al añadir nodos
        
        Estrategia:
        - Reducir porcentaje de distribución
        - Mantener redundancia crítica
        - Optimizar acceso
        """
        total_nodes = len(self.nodes)
        
        # Calcular nuevo porcentaje de distribución
        base_percentage = self.distribution_strategy['base_percentage']
        min_percentage = self.distribution_strategy['min_percentage']
        scaling_factor = self.distribution_strategy['scaling_factor']
        
        # Fórmula de reducción exponencial
        new_percentage = max(
            base_percentage * (scaling_factor ** math.log(total_nodes)),
            min_percentage
        )
        
        # Redistribuir cada recurso
        for resource_id, resource in self.resources.items():
            self._distribute_resource(resource, new_percentage)
    
    def _distribute_resource(self, resource: ResourceFragment, percentage: float):
        """
        Distribuir un recurso en un porcentaje específico de nodos
        
        Estrategias de distribución:
        1. Priorizar nodos con menor carga
        2. Mantener redundancia según prioridad
        3. Distribuir de manera uniforme
        """
        # Calcular número de nodos para distribuir
        total_nodes = len(self.nodes)
        nodes_to_distribute = max(
            int(total_nodes * percentage),
            1  # Siempre al menos un nodo
        )
        
        # Ajustar por prioridad del recurso
        priority_multiplier = self._get_priority_multiplier(resource.priority)
        nodes_to_distribute = max(
            int(nodes_to_distribute * priority_multiplier),
            1
        )
        
        # Seleccionar nodos para distribución
        selected_nodes = self._select_optimal_nodes(nodes_to_distribute)
        
        # Actualizar metadatos de distribución
        resource.distribution_metadata = {
            'nodes': selected_nodes,
            'distribution_percentage': percentage,
            'timestamp': asyncio.get_event_loop().time()
        }
    
    def _get_priority_multiplier(self, priority: int) -> float:
        """
        Obtener multiplicador de distribución según prioridad
        
        Prioridades:
        0 (Crítico) -> 1.5
        1 (Alto) -> 1.2
        2 (Medio) -> 1.0
        3 (Bajo) -> 0.8
        """
        multipliers = {
            0: 1.5,  # Crítico
            1: 1.2,  # Alto
            2: 1.0,  # Medio
            3: 0.8   # Bajo
        }
        return multipliers.get(priority, 1.0)
    
    def _select_optimal_nodes(self, num_nodes: int) -> List[str]:
        """
        Seleccionar nodos óptimos para distribución
        
        Criterios de selección:
        1. Carga actual del nodo
        2. Proximidad de recursos relacionados
        3. Latencia de red
        4. Capacidad de almacenamiento
        """
        # Convertir lista de nodos a lista para selección
        node_list = list(self.nodes)
        
        # Selección aleatoria balanceada
        selected = random.sample(node_list, min(num_nodes, len(node_list)))
        
        return selected
    
    async def add_resource(
        self, 
        resource_id: str, 
        resource_type: str, 
        priority: int, 
        size: float,
        dependencies: Optional[Set[str]] = None
    ) -> bool:
        """
        Añadir un nuevo recurso al sistema
        
        Parámetros:
        - resource_id: Identificador único
        - resource_type: Tipo de recurso
        - priority: Prioridad del recurso
        - size: Tamaño del recurso
        - dependencies: Dependencias del recurso
        """
        try:
            # Crear fragmento de recurso
            resource = ResourceFragment(
                id=resource_id,
                type=resource_type,
                priority=priority,
                size=size,
                dependencies=dependencies or set()
            )
            
            # Distribuir recurso
            self._distribute_resource(
                resource, 
                self.distribution_strategy['base_percentage']
            )
            
            # Registrar recurso
            self.resources[resource_id] = resource
            
            return True
        
        except Exception as e:
            logging.error(f"Error añadiendo recurso: {e}")
            return False
    
    def get_resource_nodes(self, resource_id: str) -> List[str]:
        """
        Obtener nodos que contienen un recurso específico
        
        Estrategias de acceso:
        1. Devolver nodos de distribución
        2. Buscar en nodos con recursos relacionados
        3. Seleccionar nodos con menor latencia
        """
        resource = self.resources.get(resource_id)
        
        if not resource:
            logging.warning(f"Recurso {resource_id} no encontrado")
            return []
        
        return resource.distribution_metadata.get('nodes', [])
    
    async def access_resource(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """
        Acceder a un recurso de manera optimizada
        
        Estrategias de acceso:
        1. Seleccionar nodo con menor latencia
        2. Balanceo de carga
        3. Tolerancia a fallos
        """
        # Obtener nodos con el recurso
        resource_nodes = self.get_resource_nodes(resource_id)
        
        if not resource_nodes:
            logging.error(f"No hay nodos para el recurso {resource_id}")
            return None
        
        # Estrategia de acceso tolerante a fallos
        for node in resource_nodes:
            try:
                # Simular acceso al recurso
                resource_data = await self._fetch_resource_from_node(node, resource_id)
                
                if resource_data:
                    return resource_data
            
            except Exception as e:
                logging.warning(f"Fallo al acceder al recurso en nodo {node}: {e}")
        
        logging.error(f"Imposible acceder al recurso {resource_id}")
        return None
    
    async def _fetch_resource_from_node(self, node_id: str, resource_id: str) -> Optional[Dict[str, Any]]:
        """
        Simular recuperación de recurso desde un nodo
        En implementación real, sería una llamada de red
        """
        # Simulación de recuperación de recurso
        return {
            'node_id': node_id,
            'resource_id': resource_id,
            'timestamp': asyncio.get_event_loop().time()
        }
    
    def get_distribution_status(self) -> Dict[str, Any]:
        """
        Obtener estado actual de distribución de recursos
        """
        return {
            'total_nodes': len(self.nodes),
            'total_resources': len(self.resources),
            'distribution_strategy': self.distribution_strategy,
            'resource_distribution': {
                resource_id: {
                    'nodes': len(resource.distribution_metadata.get('nodes', [])),
                    'distribution_percentage': resource.distribution_metadata.get('distribution_percentage', 0)
                }
                for resource_id, resource in self.resources.items()
            }
        }

# Ejemplo de uso
async def main():
    # Crear distribuidor de recursos
    distributor = AdaptiveResourceDistributor(initial_nodes=10)
    
    # Añadir nodos
    for i in range(10):
        distributor.add_node(f"node_{i}")
    
    # Añadir recursos
    await distributor.add_resource(
        resource_id="recurso_critico_1",
        resource_type="core_system",
        priority=0,  # Crítico
        size=100.0,
        dependencies={"dependencia_1", "dependencia_2"}
    )
    
    await distributor.add_resource(
        resource_id="recurso_bajo_1",
        resource_type="user_data",
        priority=3,  # Bajo
        size=10.0
    )
    
    # Añadir más nodos para ver cambios en distribución
    for i in range(10, 50):
        distributor.add_node(f"node_{i}")
    
    # Obtener estado de distribución
    status = distributor.get_distribution_status()
    print(json.dumps(status, indent=2))
    
    # Acceder a un recurso
    recurso = await distributor.access_resource("recurso_critico_1")
    print("Recurso accedido:", recurso)

if __name__ == "__main__":
    asyncio.run(main())
```