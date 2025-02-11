"""
Sistema centralizado de gestión de errores con capacidades de recuperación.
"""
from typing import Optional, Callable, Any, Dict, Type
import traceback
import sys
from functools import wraps
from .logging_manager import logger_manager

logger = logger_manager.get_logger(__name__)


class SystemError(Exception):
    """Clase base para errores del sistema."""

    def __init__(self, message: str, error_code: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class ValidationError(SystemError):
    """Error de validación de datos o configuración."""

    pass


class SecurityError(SystemError):
    """Error relacionado con la seguridad."""

    pass


class NetworkError(SystemError):
    """Error de red o comunicación."""

    pass


class ErrorHandler:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.error_handlers: Dict[Type[Exception], Callable] = {}
            self.recovery_strategies: Dict[str, Callable] = {}
            self.register_default_handlers()
            self.initialized = True

    def register_default_handlers(self) -> None:
        """Registra los manejadores de errores por defecto."""
        self.register_handler(ValidationError, self._handle_validation_error)
        self.register_handler(SecurityError, self._handle_security_error)
        self.register_handler(NetworkError, self._handle_network_error)

    def register_handler(
        self, exception_type: Type[Exception], handler: Callable
    ) -> None:
        """
        Registra un manejador para un tipo específico de excepción.

        Args:
            exception_type: Tipo de excepción a manejar
            handler: Función que manejará la excepción
        """
        self.error_handlers[exception_type] = handler

    def register_recovery_strategy(self, error_code: str, strategy: Callable) -> None:
        """
        Registra una estrategia de recuperación para un código de error.

        Args:
            error_code: Código de error
            strategy: Función de recuperación
        """
        self.recovery_strategies[error_code] = strategy

    def handle_error(self, error: Exception) -> Any:
        """
        Maneja una excepción usando el manejador apropiado.

        Args:
            error: La excepción a manejar

        Returns:
            El resultado del manejador de errores
        """
        handler = self._get_handler(error)
        return handler(error)

    def _get_handler(self, error: Exception) -> Callable:
        """
        Obtiene el manejador apropiado para una excepción.

        Args:
            error: La excepción

        Returns:
            El manejador de errores
        """
        for error_type, handler in self.error_handlers.items():
            if isinstance(error, error_type):
                return handler
        return self._handle_unknown_error

    def _handle_validation_error(self, error: ValidationError) -> None:
        """Maneja errores de validación."""
        logger.error(
            f"Validation error: {str(error)}",
            extra={"error_code": error.error_code, "details": error.details},
        )

    def _handle_security_error(self, error: SecurityError) -> None:
        """Maneja errores de seguridad."""
        logger.critical(
            f"Security error: {str(error)}",
            extra={"error_code": error.error_code, "details": error.details},
        )

    def _handle_network_error(self, error: NetworkError) -> None:
        """Maneja errores de red."""
        logger.error(
            f"Network error: {str(error)}",
            extra={"error_code": error.error_code, "details": error.details},
        )

    def _handle_unknown_error(self, error: Exception) -> None:
        """Maneja errores desconocidos."""
        logger.error(f"Unknown error: {str(error)}\n{traceback.format_exc()}")

    def try_recover(self, error_code: str, **kwargs) -> Optional[Any]:
        """
        Intenta recuperarse de un error usando la estrategia registrada.

        Args:
            error_code: Código del error
            **kwargs: Argumentos adicionales para la estrategia de recuperación

        Returns:
            El resultado de la estrategia de recuperación si existe
        """
        strategy = self.recovery_strategies.get(error_code)
        if strategy:
            try:
                return strategy(**kwargs)
            except Exception as e:
                logger.error(f"Recovery strategy failed: {str(e)}")
        return None


def handle_errors(func: Callable) -> Callable:
    """
    Decorador para manejar errores automáticamente.

    Args:
        func: La función a decorar

    Returns:
        Función decorada con manejo de errores
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_handler = ErrorHandler()
            return error_handler.handle_error(e)

    return wrapper


# Singleton instance
error_handler = ErrorHandler()
