"""
Calculator for smartphone cluster requirements
"""


class SmartphoneClusterCalculator:
    """Calculates requirements for smartphone-based learning clusters"""

    def __init__(self):
        # Especificaciones típicas de un smartphone gama media 2024
        self.smartphone_specs = {
            "ram": 6,  # GB
            "storage": 128,  # GB
            "processor_speed": 2.4,  # GHz
            "cores": 8,
            "gpu_power": 1.2,  # TFLOPS
            "network_speed": 100,  # Mbps promedio en 4G/5G
            "power_consumption": 4,  # Watts en uso intensivo
            "efficiency_factor": 0.7,  # Factor de eficiencia del cluster
        }

        # Requisitos de Qwen-2.5
        self.model_requirements = {
            "total_size": 14000,  # GB (14TB)
            "processing_power_needed": 2000,  # TFLOPS para procesamiento en un día
            "memory_needed": 28000,  # GB (28TB) para análisis efectivo
            "network_bandwidth_needed": 1000,  # Gbps para distribución rápida
            "target_time": 24,  # horas
        }

    def calculate_nodes_needed(self) -> dict:
        """Calcula el número de nodos (smartphones) necesarios"""

        # 1. Cálculo por memoria
        nodes_by_memory = self._calculate_nodes_by_memory()

        # 2. Cálculo por capacidad de procesamiento
        nodes_by_processing = self._calculate_nodes_by_processing()

        # 3. Cálculo por ancho de banda
        nodes_by_network = self._calculate_nodes_by_network()

        # 4. Cálculo por almacenamiento
        nodes_by_storage = self._calculate_nodes_by_storage()

        # Tomamos el máximo de todos los requisitos
        total_nodes = max(
            nodes_by_memory, nodes_by_processing, nodes_by_network, nodes_by_storage
        )

        # Añadimos un 20% extra para redundancia y tolerancia a fallos
        total_nodes = int(total_nodes * 1.2)

        return {
            "total_nodes_needed": total_nodes,
            "breakdown": {
                "by_memory": nodes_by_memory,
                "by_processing": nodes_by_processing,
                "by_network": nodes_by_network,
                "by_storage": nodes_by_storage,
            },
            "cluster_specs": self._calculate_cluster_specs(total_nodes),
            "limitations": self._identify_limitations(total_nodes),
            "recommendations": self._generate_recommendations(total_nodes),
        }

    def _calculate_nodes_by_memory(self) -> int:
        """Calcula nodos necesarios basado en requisitos de memoria"""
        memory_per_node = (
            self.smartphone_specs["ram"] * self.smartphone_specs["efficiency_factor"]
        )
        return int(self.model_requirements["memory_needed"] / memory_per_node)

    def _calculate_nodes_by_processing(self) -> int:
        """Calcula nodos necesarios basado en capacidad de procesamiento"""
        processing_per_node = (
            self.smartphone_specs["gpu_power"]
            * self.smartphone_specs["efficiency_factor"]
        )
        return int(
            self.model_requirements["processing_power_needed"] / processing_per_node
        )

    def _calculate_nodes_by_network(self) -> int:
        """Calcula nodos necesarios basado en requisitos de red"""
        network_per_node = (
            self.smartphone_specs["network_speed"] / 1000
        ) * self.smartphone_specs[
            "efficiency_factor"
        ]  # Convertido a Gbps
        return int(
            self.model_requirements["network_bandwidth_needed"] / network_per_node
        )

    def _calculate_nodes_by_storage(self) -> int:
        """Calcula nodos necesarios basado en almacenamiento"""
        storage_per_node = (
            self.smartphone_specs["storage"]
            * self.smartphone_specs["efficiency_factor"]
        )
        return int(self.model_requirements["total_size"] / storage_per_node)

    def _calculate_cluster_specs(self, total_nodes: int) -> dict:
        """Calcula las especificaciones totales del cluster"""
        return {
            "total_ram": total_nodes * self.smartphone_specs["ram"],
            "total_storage": total_nodes * self.smartphone_specs["storage"],
            "total_processing": total_nodes * self.smartphone_specs["gpu_power"],
            "total_network": total_nodes
            * self.smartphone_specs["network_speed"]
            / 1000,
            "power_consumption_kw": (
                total_nodes * self.smartphone_specs["power_consumption"]
            )
            / 1000,
            "effective_efficiency": self.smartphone_specs["efficiency_factor"] * 100,
        }

    def _identify_limitations(self, total_nodes: int) -> list:
        """Identifica las limitaciones del cluster"""
        limitations = []

        if total_nodes > 100000:
            limitations.append(
                "Número extremadamente alto de nodos - difícil de coordinar"
            )

        power_consumption = (
            total_nodes * self.smartphone_specs["power_consumption"]
        ) / 1000
        if power_consumption > 1000:  # más de 1MW
            limitations.append("Consumo de energía excesivo")

        if (
            total_nodes * self.smartphone_specs["network_speed"]
        ) < self.model_requirements["network_bandwidth_needed"] * 1000:
            limitations.append(
                "Ancho de banda insuficiente incluso con todos los nodos"
            )

        return limitations

    def _generate_recommendations(self, total_nodes: int) -> list:
        """Genera recomendaciones para la implementación"""
        recommendations = []

        if total_nodes > 50000:
            recommendations.append(
                "Considerar dividir el cluster en sub-clusters geográficos"
            )

        if (
            total_nodes * self.smartphone_specs["network_speed"]
        ) < self.model_requirements["network_bandwidth_needed"] * 1000:
            recommendations.append("Implementar sistema de caché distribuido")

        recommendations.append(f"Asegurar conexión estable para {total_nodes} nodos")
        recommendations.append("Implementar sistema de recuperación ante fallos")
        recommendations.append(
            "Establecer sistema de rotación de nodos para distribución de carga"
        )

        return recommendations


# Calcular requisitos
calculator = SmartphoneClusterCalculator()
requirements = calculator.calculate_nodes_needed()


# Formatear resultados para mejor legibilidad
def format_number(num):
    if num >= 1000000:
        return f"{num/1000000:.1f}M"
    elif num >= 1000:
        return f"{num/1000:.1f}K"
    return str(num)


formatted_requirements = {
    "total_nodes": format_number(requirements["total_nodes_needed"]),
    "breakdown": {
        "por_memoria": format_number(requirements["breakdown"]["by_memory"]),
        "por_procesamiento": format_number(requirements["breakdown"]["by_processing"]),
        "por_red": format_number(requirements["breakdown"]["by_network"]),
        "por_almacenamiento": format_number(requirements["breakdown"]["by_storage"]),
    },
    "especificaciones_cluster": {
        "ram_total_tb": f"{requirements['cluster_specs']['total_ram']/1000:.1f}TB",
        "almacenamiento_total_tb": f"{requirements['cluster_specs']['total_storage']/1000:.1f}TB",
        "procesamiento_total_tflops": f"{requirements['cluster_specs']['total_processing']:.1f}TFLOPS",
        "red_total_tbps": f"{requirements['cluster_specs']['total_network']/1000:.2f}Tbps",
        "consumo_energia_mw": f"{requirements['cluster_specs']['power_consumption_kw']/1000:.1f}MW",
    },
    "limitaciones": requirements["limitations"],
    "recomendaciones": requirements["recommendations"],
}
