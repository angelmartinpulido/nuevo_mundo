"""
Data Validation System for Enhanced Learning Quality
"""
from typing import Any, Dict, List, Optional
import numpy as np
from datetime import datetime


class DataValidator:
    def __init__(self):
        self.validation_rules = {}
        self.quality_metrics = {}
        self.validation_history = []

    def add_validation_rule(self, data_type: str, rule: callable):
        """Add a validation rule for a specific data type"""
        if data_type not in self.validation_rules:
            self.validation_rules[data_type] = []
        self.validation_rules[data_type].append(rule)

    def validate_data(self, data: Any, data_type: str) -> Dict[str, Any]:
        """
        Validate data against defined rules
        Returns: Dict with validation results and quality metrics
        """
        results = {
            "timestamp": datetime.now(),
            "data_type": data_type,
            "is_valid": True,
            "quality_score": 0.0,
            "issues": [],
        }

        if data_type in self.validation_rules:
            for rule in self.validation_rules[data_type]:
                try:
                    rule_result = rule(data)
                    if not rule_result["valid"]:
                        results["is_valid"] = False
                        results["issues"].append(rule_result["issue"])
                except Exception as e:
                    results["is_valid"] = False
                    results["issues"].append(str(e))

        # Calculate quality score
        results["quality_score"] = self._calculate_quality_score(
            data, results["issues"]
        )
        self.validation_history.append(results)
        return results

    def _calculate_quality_score(self, data: Any, issues: List[str]) -> float:
        """Calculate a quality score between 0 and 1"""
        base_score = 1.0

        # Reduce score based on issues
        issue_penalty = len(issues) * 0.1
        base_score -= min(issue_penalty, 0.8)  # Keep minimum score of 0.2

        # Additional quality metrics
        if hasattr(data, "__len__"):
            # Check for missing values
            if hasattr(data, "isnull"):  # pandas DataFrame/Series
                missing_ratio = data.isnull().sum() / len(data)
                base_score *= 1 - missing_ratio
            elif isinstance(data, (list, np.ndarray)):
                missing_count = sum(1 for x in data if x is None)
                base_score *= 1 - missing_count / len(data)

        return max(0.0, min(1.0, base_score))

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get statistics about data validation"""
        if not self.validation_history:
            return {}

        stats = {
            "total_validations": len(self.validation_history),
            "valid_ratio": sum(1 for x in self.validation_history if x["is_valid"])
            / len(self.validation_history),
            "average_quality": sum(x["quality_score"] for x in self.validation_history)
            / len(self.validation_history),
            "common_issues": self._get_common_issues(),
        }
        return stats

    def _get_common_issues(self) -> Dict[str, int]:
        """Analyze common validation issues"""
        issues_count = {}
        for validation in self.validation_history:
            for issue in validation["issues"]:
                issues_count[issue] = issues_count.get(issue, 0) + 1
        return dict(sorted(issues_count.items(), key=lambda x: x[1], reverse=True))


class DataPriorityManager:
    def __init__(self):
        self.priority_rules = {}
        self.source_weights = {}
        self.learning_feedback = {}

    def set_source_priority(self, source: str, priority: float):
        """Set priority weight for a data source"""
        self.source_weights[source] = max(0.0, min(1.0, priority))

    def calculate_priority(self, data_source: str, quality_score: float) -> float:
        """Calculate final priority based on source weight and quality"""
        base_weight = self.source_weights.get(data_source, 0.5)
        return base_weight * quality_score

    def update_priority_from_feedback(self, data_source: str, feedback_score: float):
        """Update source priority based on learning feedback"""
        current_weight = self.source_weights.get(data_source, 0.5)
        # Adaptive weight adjustment
        self.source_weights[data_source] = current_weight * 0.8 + feedback_score * 0.2
