from typing import Dict, Any, List, Type
import asyncio
from .base_module import BaseModule
from .m1_hyperoptimized_ingestion import M1HyperoptimizedIngestion
from .m2_global_modeling import M2GlobalModeling
from .m3_distributed_memory import M3DistributedMemory
from .m4_combined_reasoning import M4CombinedReasoning
from .m5_federated_learning import M5FederatedLearning
from .m6_metacognition import M6Metacognition
from .m7_interaction import M7Interaction
from .m8_self_improvement import M8SelfImprovement
from .config import CONFIG
from .logging_system import LOGGER, LogLevel
from .metrics_system import METRICS
from .error_handling import ERROR_MANAGER
from .quantum_security import SECURITY_MANAGER


class SystemOrchestrator:
    def __init__(self):
        self.modules: Dict[str, BaseModule] = {}
        self.module_dependencies = {
            "M1HyperoptimizedIngestion": [],
            "M2GlobalModeling": ["M1HyperoptimizedIngestion"],
            "M3DistributedMemory": ["M1HyperoptimizedIngestion"],
            "M4CombinedReasoning": ["M2GlobalModeling", "M3DistributedMemory"],
            "M5FederatedLearning": ["M4CombinedReasoning"],
            "M6Metacognition": [],  # Can start independently
            "M7Interaction": ["M4CombinedReasoning", "M5FederatedLearning"],
            "M8SelfImprovement": ["M6Metacognition"],
        }

    async def initialize(self) -> None:
        # Initialize all core systems
        await LOGGER.log(
            LogLevel.INFO, "Orchestrator", "Starting core systems initialization"
        )

        # Add handlers
        await LOGGER.add_handler(FileHandler("system.log"))
        await LOGGER.add_handler(ConsoleHandler())
        await LOGGER.add_handler(MetricsHandler())
        await LOGGER.add_handler(AlertHandler())

        # Initialize modules in dependency order
        module_classes = {
            "M1HyperoptimizedIngestion": M1HyperoptimizedIngestion,
            "M2GlobalModeling": M2GlobalModeling,
            "M3DistributedMemory": M3DistributedMemory,
            "M4CombinedReasoning": M4CombinedReasoning,
            "M5FederatedLearning": M5FederatedLearning,
            "M6Metacognition": M6Metacognition,
            "M7Interaction": M7Interaction,
            "M8SelfImprovement": M8SelfImprovement,
        }

        initialized = set()
        while len(initialized) < len(module_classes):
            for module_name, deps in self.module_dependencies.items():
                if module_name not in initialized and all(
                    d in initialized for d in deps
                ):
                    await self._initialize_module(module_classes[module_name])
                    initialized.add(module_name)

        await LOGGER.log(
            LogLevel.INFO, "Orchestrator", "All modules initialized successfully"
        )

    async def _initialize_module(self, module_class: Type[BaseModule]) -> None:
        module_name = module_class.__name__
        try:
            module = module_class()
            await module.initialize()
            self.modules[module_name] = module

            await LOGGER.log(
                LogLevel.INFO,
                "Orchestrator",
                f"Module {module_name} initialized successfully",
            )

        except Exception as e:
            await LOGGER.log(
                LogLevel.ERROR,
                "Orchestrator",
                f"Failed to initialize module {module_name}: {str(e)}",
            )
            raise

    async def process(self, input_data: Any) -> Any:
        # Process data through the module pipeline
        current_data = input_data

        # M1: Ingestion
        current_data = await self.modules["M1HyperoptimizedIngestion"]._secure_process(
            current_data
        )

        # M2 & M3: Parallel processing
        m2_task = asyncio.create_task(
            self.modules["M2GlobalModeling"]._secure_process(current_data)
        )
        m3_task = asyncio.create_task(
            self.modules["M3DistributedMemory"]._secure_process(current_data)
        )
        m2_result, m3_result = await asyncio.gather(m2_task, m3_task)

        # M4: Reasoning
        reasoning_input = {"model_data": m2_result, "memory_data": m3_result}
        current_data = await self.modules["M4CombinedReasoning"]._secure_process(
            reasoning_input
        )

        # M5: Learning
        current_data = await self.modules["M5FederatedLearning"]._secure_process(
            current_data
        )

        # M6: Metacognition (runs continuously in background)
        metacognition_task = asyncio.create_task(
            self.modules["M6Metacognition"]._secure_process(current_data)
        )

        # M7: Interaction
        current_data = await self.modules["M7Interaction"]._secure_process(current_data)

        # M8: Self-improvement (runs continuously in background)
        improvement_task = asyncio.create_task(
            self.modules["M8SelfImprovement"]._secure_process(current_data)
        )

        return current_data

    async def shutdown(self) -> None:
        # Shutdown all modules in reverse dependency order
        shutdown_order = list(self.modules.keys())
        shutdown_order.reverse()

        for module_name in shutdown_order:
            try:
                await self.modules[module_name].shutdown()
                await LOGGER.log(
                    LogLevel.INFO,
                    "Orchestrator",
                    f"Module {module_name} shutdown successfully",
                )
            except Exception as e:
                await LOGGER.log(
                    LogLevel.ERROR,
                    "Orchestrator",
                    f"Error shutting down module {module_name}: {str(e)}",
                )

    async def health_check(self) -> Dict[str, Any]:
        # Collect health metrics from all modules
        health_data = {"system_status": "healthy", "modules": {}}

        for module_name, module in self.modules.items():
            try:
                module_health = await module.health_check()
                health_data["modules"][module_name] = module_health
            except Exception as e:
                health_data["modules"][module_name] = {
                    "status": "error",
                    "error": str(e),
                }
                health_data["system_status"] = "degraded"

        # Add system-wide metrics
        health_data.update(
            {
                "metrics": METRICS.get_all_metrics(),
                "errors": ERROR_MANAGER.get_error_statistics(),
                "security": SECURITY_MANAGER.get_security_statistics(),
            }
        )

        return health_data


# Handlers (referenced in initialize)
class FileHandler:
    def __init__(self, filename: str):
        self.filename = filename

    async def __call__(self, entry: Any):
        with open(self.filename, "a") as f:
            f.write(f"{entry}\n")


class ConsoleHandler:
    async def __call__(self, entry: Any):
        print(entry)


class MetricsHandler:
    async def __call__(self, entry: Any):
        # Update metrics based on log entry
        pass


class AlertHandler:
    async def __call__(self, entry: Any):
        # Send alerts for critical log entries
        pass


# Global orchestrator instance
ORCHESTRATOR = SystemOrchestrator()
