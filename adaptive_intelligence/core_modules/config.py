from typing import Dict, Any
import json
from pathlib import Path


class SystemConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.config = {
            "system": {
                "version": "2.0.0",
                "max_threads": 1024,
                "quantum_enabled": True,
                "hpc_enabled": True,
                "auto_scaling": True,
                "error_tolerance": 0.00001,
                "backup_interval_ms": 100,
            },
            "security": {
                "encryption_level": "quantum_resistant",
                "key_rotation_interval_ms": 3600000,
                "audit_level": "full",
                "secure_boot": True,
            },
            "performance": {
                "max_latency_ms": 1,
                "target_throughput_gbs": 100,
                "memory_buffer_gb": 256,
                "cache_policy": "adaptive",
            },
            "modules": {
                "m1": {
                    "compression_ratio": 0.95,
                    "batch_size": 10000,
                    "network_allocation": 0.4,
                },
                "m2": {
                    "shard_count": 1000,
                    "replication_factor": 3,
                    "consistency_level": "strong",
                },
                "m3": {"dram_size_tb": 1, "ssd_size_pb": 1, "hdd_size_pb": 10},
                "m4": {
                    "quantum_shots": 10000,
                    "reasoning_timeout_ms": 10,
                    "cache_size_gb": 512,
                },
                "m5": {
                    "federation_nodes": 10000,
                    "learning_rate": "adaptive",
                    "validation_ratio": 0.2,
                },
                "m6": {
                    "rule_check_interval_ms": 1,
                    "rollback_points": 100,
                    "audit_depth": "full",
                },
                "m7": {
                    "gpu_allocation": 0.8,
                    "real_time_priority": "critical",
                    "response_timeout_ms": 10,
                },
                "m8": {
                    "improvement_cycles": 1000,
                    "validation_rounds": 100,
                    "confidence_threshold": 0.99999,
                },
            },
        }

    def get(self, path: str, default: Any = None) -> Any:
        """Get config value using dot notation path"""
        current = self.config
        for key in path.split("."):
            if isinstance(current, dict):
                current = current.get(key, default)
            else:
                return default
        return current

    def update(self, path: str, value: Any) -> None:
        """Update config value using dot notation path"""
        keys = path.split(".")
        current = self.config
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value

    def save(self, path: str = "config.json") -> None:
        """Save config to file"""
        with open(path, "w") as f:
            json.dump(self.config, f, indent=2)

    def load(self, path: str = "config.json") -> None:
        """Load config from file"""
        if Path(path).exists():
            with open(path, "r") as f:
                self.config.update(json.load(f))


# Global config instance
CONFIG = SystemConfig()
