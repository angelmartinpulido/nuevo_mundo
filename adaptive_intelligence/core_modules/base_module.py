from abc import ABC, abstractmethod
import asyncio
from typing import Any, Dict, Optional
from .config import CONFIG
from .logging_system import LOGGER, LogLevel
from .metrics_system import METRICS, MetricType
from .error_handling import ERROR_MANAGER, ErrorSeverity, ErrorCategory
from .quantum_security import SECURITY_MANAGER, SecurityLevel, EncryptionMode


class BaseModule(ABC):
    def __init__(self):
        self.is_running = False
        self._event_loop = None
        self._config = {}
        self._security_context = None
        self._initialize_metrics()

    def _initialize_metrics(self):
        module_name = self.__class__.__name__

        # Register standard metrics
        METRICS.register_metric(
            f"{module_name}_processing_time",
            MetricType.HISTOGRAM,
            f"Processing time for {module_name}",
            {"module": module_name},
        )

        METRICS.register_metric(
            f"{module_name}_error_count",
            MetricType.COUNTER,
            f"Error count for {module_name}",
            {"module": module_name},
        )

        METRICS.register_metric(
            f"{module_name}_throughput",
            MetricType.GAUGE,
            f"Throughput for {module_name}",
            {"module": module_name},
        )

    async def _setup_security(self):
        self._security_context = await SECURITY_MANAGER.create_security_context(
            level=SecurityLevel.QUANTUM,
            encryption_mode=EncryptionMode.HYBRID,
            quantum_keys=True,
            audit_enabled=True,
        )

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize module resources and connections"""
        await self._setup_security()
        await LOGGER.log(LogLevel.INFO, self.__class__.__name__, "Module initialized")

    async def _secure_process(self, input_data: Any) -> Any:
        try:
            # Encrypt input if needed
            if isinstance(input_data, bytes):
                input_data = await SECURITY_MANAGER.decrypt_data(
                    input_data, self._security_context
                )

            # Process data
            start_time = asyncio.get_event_loop().time()
            result = await self.process(input_data)
            processing_time = asyncio.get_event_loop().time() - start_time

            # Update metrics
            module_name = self.__class__.__name__
            METRICS.get_metric(f"{module_name}_processing_time").add_value(
                processing_time
            )
            METRICS.get_metric(f"{module_name}_throughput").add_value(1)

            # Encrypt output if needed
            if isinstance(result, bytes):
                result = await SECURITY_MANAGER.encrypt_data(
                    result, self._security_context
                )

            return result

        except Exception as e:
            # Handle error
            module_name = self.__class__.__name__
            METRICS.get_metric(f"{module_name}_error_count").add_value(1)

            await ERROR_MANAGER.handle_error(
                error=e,
                module=module_name,
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.PROCESSING,
            )

            raise

    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Process input data according to module's specific logic"""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources and connections"""
        await LOGGER.log(LogLevel.INFO, self.__class__.__name__, "Module shutdown")

    def configure(self, config: Dict) -> None:
        """Update module configuration"""
        self._config.update(config)

    async def health_check(self) -> Dict:
        """Return module health metrics"""
        module_name = self.__class__.__name__

        metrics = {
            "status": "running" if self.is_running else "stopped",
            "module_type": module_name,
            "metrics": {
                name: metric.get_statistics()
                for name, metric in METRICS.metrics.items()
                if name.startswith(module_name)
            },
            "security": self._security_context.to_dict()
            if self._security_context
            else None,
        }

        return metrics

    def get_config(self) -> Dict:
        """Return current module configuration"""
        return self._config.copy()
