"""
Learning Integrator - Coordinates all learning enhancement components
"""
from typing import Dict, Any, Optional
from datetime import datetime

from .feedback_system import FeedbackSystem
from .learning_balancer import LearningBalancer
from ..data_validation.data_validator import DataValidator, DataPriorityManager


class LearningIntegrator:
    def __init__(self):
        self.data_validator = DataValidator()
        self.priority_manager = DataPriorityManager()
        self.feedback_system = FeedbackSystem()
        self.learning_balancer = LearningBalancer()

        # Initialize default validation rules
        self._setup_default_validation_rules()

    def process_input(
        self, data: Any, source: str, data_type: str, is_real_time: bool = True
    ) -> Dict[str, Any]:
        """
        Process new input data through the enhanced learning pipeline
        """
        result = {
            "timestamp": datetime.now(),
            "source": source,
            "data_type": data_type,
            "processing_status": "initiated",
        }

        # 1. Validate data
        validation_result = self.data_validator.validate_data(data, data_type)
        result["validation"] = validation_result

        if not validation_result["is_valid"]:
            result["processing_status"] = "failed_validation"
            return result

        # 2. Calculate priority
        priority = self.priority_manager.calculate_priority(
            source, validation_result["quality_score"]
        )
        result["priority"] = priority

        # 3. Balance learning resources
        learning_allocation = self.learning_balancer.optimize_learning_allocation(
            real_time_data={"priority": priority} if is_real_time else {},
            historical_data={} if is_real_time else {"priority": priority},
        )
        result["resource_allocation"] = learning_allocation

        # 4. Process the learning
        learning_result = self._process_learning(
            data, source, data_type, is_real_time, learning_allocation
        )
        result.update(learning_result)

        # 5. Collect feedback
        feedback = self.feedback_system.record_feedback(
            source=source,
            prediction=learning_result.get("prediction"),
            actual=learning_result.get("actual"),
            context={
                "priority": priority,
                "validation_score": validation_result["quality_score"],
                "is_real_time": is_real_time,
            },
        )
        result["feedback"] = feedback

        # 6. Update priority based on feedback
        self.priority_manager.update_priority_from_feedback(
            source, feedback["accuracy"]
        )

        result["processing_status"] = "completed"
        return result

    def get_learning_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the learning system
        """
        return {
            "validation_stats": self.data_validator.get_validation_statistics(),
            "feedback_status": self.feedback_system.get_learning_status(),
            "priority_weights": self.priority_manager.source_weights,
            "learning_metrics": self._get_learning_metrics(),
        }

    def _setup_default_validation_rules(self):
        """Setup default validation rules for common data types"""

        # Numeric data validation
        def validate_numeric(data):
            if not isinstance(data, (int, float)):
                return {"valid": False, "issue": "not_numeric"}
            return {"valid": True}

        # Text data validation
        def validate_text(data):
            if not isinstance(data, str):
                return {"valid": False, "issue": "not_text"}
            if not data.strip():
                return {"valid": False, "issue": "empty_text"}
            return {"valid": True}

        # Array data validation
        def validate_array(data):
            if not isinstance(data, (list, tuple)):
                return {"valid": False, "issue": "not_array"}
            if not data:
                return {"valid": False, "issue": "empty_array"}
            return {"valid": True}

        self.data_validator.add_validation_rule("numeric", validate_numeric)
        self.data_validator.add_validation_rule("text", validate_text)
        self.data_validator.add_validation_rule("array", validate_array)

    def _process_learning(
        self,
        data: Any,
        source: str,
        data_type: str,
        is_real_time: bool,
        allocation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process the actual learning based on allocated resources
        """
        # This would connect to the actual learning implementation
        # For now, return a placeholder result
        return {
            "learning_completed": True,
            "prediction": None,
            "actual": None,
            "processing_time": 0.0,
        }

    def _get_learning_metrics(self) -> Dict[str, Any]:
        """
        Get detailed metrics about the learning system performance
        """
        return {
            "total_processed": 0,  # Would be implemented
            "success_rate": 0.0,  # Would be implemented
            "average_quality": 0.0,  # Would be implemented
            "resource_efficiency": 0.0,  # Would be implemented
        }
