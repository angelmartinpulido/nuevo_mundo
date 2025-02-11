"""
Gestor de Objetivos del Sistema
Maneja la definición, seguimiento y cumplimiento de objetivos
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import json


class ObjectivePriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class ObjectiveDefinition:
    id: str
    name: str
    description: str
    priority: ObjectivePriority
    success_criteria: Dict[str, Any]
    required_functions: List[str]
    dependencies: List[str]
    resources: Dict[str, Any]
    timeline: Dict[str, Any]


class ObjectiveManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.objectives: Dict[str, ObjectiveDefinition] = {}
        self.active_objectives: Dict[str, Dict[str, Any]] = {}
        self.completed_objectives: Dict[str, Dict[str, Any]] = {}
        self.objective_progress: Dict[str, float] = {}

    def define_objective(self, objective_data: Dict[str, Any]) -> str:
        """
        Define un nuevo objetivo en el sistema.

        Args:
            objective_data: Datos completos del objetivo

        Returns:
            ID del objetivo creado
        """
        objective = ObjectiveDefinition(
            id=objective_data["id"],
            name=objective_data["name"],
            description=objective_data["description"],
            priority=ObjectivePriority[objective_data["priority"]],
            success_criteria=objective_data["success_criteria"],
            required_functions=objective_data["required_functions"],
            dependencies=objective_data["dependencies"],
            resources=objective_data["resources"],
            timeline=objective_data["timeline"],
        )

        self._validate_objective(objective)
        self.objectives[objective.id] = objective
        return objective.id

    def _validate_objective(self, objective: ObjectiveDefinition) -> None:
        """
        Valida que un objetivo esté correctamente definido y sea alcanzable.
        """
        if not self._are_resources_available(objective.resources):
            raise ValueError(f"Recursos insuficientes para el objetivo {objective.id}")

        if not self._are_functions_available(objective.required_functions):
            raise ValueError(f"Funciones requeridas no disponibles para {objective.id}")

        if not self._are_dependencies_valid(objective.dependencies):
            raise ValueError(f"Dependencias inválidas para el objetivo {objective.id}")

    def activate_objective(self, objective_id: str) -> None:
        """
        Activa un objetivo para su ejecución.
        """
        if objective_id not in self.objectives:
            raise ValueError(f"Objetivo {objective_id} no encontrado")

        objective = self.objectives[objective_id]
        if self._can_activate_objective(objective):
            self.active_objectives[objective_id] = {
                "start_time": self._get_current_time(),
                "status": "active",
                "progress": 0.0,
            }
            self._initialize_objective_monitoring(objective_id)

    def _can_activate_objective(self, objective: ObjectiveDefinition) -> bool:
        """
        Verifica si un objetivo puede ser activado.
        """
        return (
            self._are_dependencies_met(objective.dependencies)
            and self._are_resources_available(objective.resources)
            and self._are_functions_ready(objective.required_functions)
        )

    def update_objective_progress(
        self, objective_id: str, progress_data: Dict[str, Any]
    ) -> None:
        """
        Actualiza el progreso de un objetivo activo.
        """
        if objective_id not in self.active_objectives:
            raise ValueError(f"Objetivo {objective_id} no está activo")

        current_progress = self._calculate_progress(objective_id, progress_data)
        self.objective_progress[objective_id] = current_progress

        if self._is_objective_completed(objective_id):
            self._complete_objective(objective_id)

    def _calculate_progress(
        self, objective_id: str, progress_data: Dict[str, Any]
    ) -> float:
        """
        Calcula el progreso actual de un objetivo.
        """
        objective = self.objectives[objective_id]
        total_criteria = len(objective.success_criteria)
        met_criteria = sum(
            1
            for criterion in objective.success_criteria
            if self._is_criterion_met(criterion, progress_data)
        )
        return (met_criteria / total_criteria) * 100

    def _is_criterion_met(self, criterion: str, progress_data: Dict[str, Any]) -> bool:
        """
        Verifica si un criterio específico se ha cumplido.
        """
        criterion_type = self._get_criterion_type(criterion)
        criterion_value = progress_data.get(criterion)

        return self._evaluate_criterion(criterion_type, criterion_value)

    def get_objective_status(self, objective_id: str) -> Dict[str, Any]:
        """
        Obtiene el estado actual de un objetivo.
        """
        if objective_id not in self.objectives:
            raise ValueError(f"Objetivo {objective_id} no encontrado")

        objective = self.objectives[objective_id]
        status = {
            "definition": objective.__dict__,
            "active": objective_id in self.active_objectives,
            "completed": objective_id in self.completed_objectives,
            "progress": self.objective_progress.get(objective_id, 0.0),
        }

        if objective_id in self.active_objectives:
            status.update(self.active_objectives[objective_id])

        if objective_id in self.completed_objectives:
            status.update(self.completed_objectives[objective_id])

        return status

    def get_all_objectives_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene el estado de todos los objetivos en el sistema.
        """
        return {
            objective_id: self.get_objective_status(objective_id)
            for objective_id in self.objectives
        }

    def _complete_objective(self, objective_id: str) -> None:
        """
        Marca un objetivo como completado y realiza las acciones necesarias.
        """
        objective = self.objectives[objective_id]
        completion_data = {
            "completion_time": self._get_current_time(),
            "final_state": self._get_objective_final_state(objective_id),
            "metrics": self._collect_objective_metrics(objective_id),
        }

        self.completed_objectives[objective_id] = completion_data
        del self.active_objectives[objective_id]

        self._notify_objective_completion(objective_id, completion_data)
        self._check_dependent_objectives(objective_id)
