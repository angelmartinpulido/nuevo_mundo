"""
Sistema de adaptadores universal para integración con diferentes sistemas y protocolos.
"""
from typing import Any, Dict, Optional, Callable, List
import json
import requests
from abc import ABC, abstractmethod
from ..core.config import config
from ..core.logging_manager import logger_manager
from ..core.error_handler import handle_errors, NetworkError

logger = logger_manager.get_logger(__name__)


class BaseAdapter(ABC):
    """Clase base para todos los adaptadores."""

    def __init__(self):
        self.retry_attempts = config.get("adapters.retry_attempts", 3)
        self.timeout = config.get("adapters.timeout", 5)
        self.callbacks: List[Callable] = []

    @abstractmethod
    def connect(self) -> bool:
        """Establece conexión con el sistema externo."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Cierra la conexión con el sistema externo."""
        pass

    @abstractmethod
    def send(self, data: Any) -> bool:
        """Envía datos al sistema externo."""
        pass

    @abstractmethod
    def receive(self) -> Optional[Any]:
        """Recibe datos del sistema externo."""
        pass

    def register_callback(self, callback: Callable) -> None:
        """Registra un callback para eventos."""
        self.callbacks.append(callback)

    def notify_callbacks(self, event: str, data: Any) -> None:
        """Notifica a todos los callbacks registrados."""
        for callback in self.callbacks:
            try:
                callback(event, data)
            except Exception as e:
                logger.error(f"Error in adapter callback: {str(e)}")


class HTTPAdapter(BaseAdapter):
    """Adaptador para comunicación HTTP/HTTPS."""

    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.session = requests.Session()
        self.headers: Dict[str, str] = {}

    @handle_errors
    def connect(self) -> bool:
        """Verifica la conexión al servidor."""
        try:
            response = self.session.get(self.base_url, timeout=self.timeout)
            return response.status_code == 200
        except requests.RequestException as e:
            raise NetworkError(f"Connection failed: {str(e)}", "NET_001")

    def disconnect(self) -> None:
        """Cierra la sesión HTTP."""
        self.session.close()

    @handle_errors
    def send(self, data: Any) -> bool:
        """
        Envía datos via HTTP POST.

        Args:
            data: Datos a enviar

        Returns:
            True si el envío fue exitoso
        """
        for attempt in range(self.retry_attempts):
            try:
                response = self.session.post(
                    self.base_url, json=data, headers=self.headers, timeout=self.timeout
                )
                if response.status_code == 200:
                    self.notify_callbacks("send_success", data)
                    return True

                logger.warning(
                    f"HTTP send failed (attempt {attempt + 1}/{self.retry_attempts}): "
                    f"Status {response.status_code}"
                )
            except requests.RequestException as e:
                logger.error(f"HTTP send error: {str(e)}")

        raise NetworkError("Failed to send data after retries", "NET_002")

    @handle_errors
    def receive(self) -> Optional[Dict]:
        """
        Recibe datos via HTTP GET.

        Returns:
            Datos recibidos o None si hay error
        """
        try:
            response = self.session.get(
                self.base_url, headers=self.headers, timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                self.notify_callbacks("receive_success", data)
                return data

            logger.warning(f"HTTP receive failed: Status {response.status_code}")
            return None

        except requests.RequestException as e:
            raise NetworkError(f"Failed to receive data: {str(e)}", "NET_003")

    def set_header(self, key: str, value: str) -> None:
        """
        Establece un header HTTP.

        Args:
            key: Nombre del header
            value: Valor del header
        """
        self.headers[key] = value


class WebSocketAdapter(BaseAdapter):
    """Adaptador para comunicación WebSocket."""

    def __init__(self, url: str):
        super().__init__()
        self.url = url
        self.ws = None
        self.connected = False

    @handle_errors
    def connect(self) -> bool:
        """Establece conexión WebSocket."""
        try:
            import websocket

            self.ws = websocket.WebSocketApp(
                self.url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            self.connected = True
            return True
        except Exception as e:
            raise NetworkError(f"WebSocket connection failed: {str(e)}", "NET_004")

    def disconnect(self) -> None:
        """Cierra la conexión WebSocket."""
        if self.ws:
            self.ws.close()
        self.connected = False

    @handle_errors
    def send(self, data: Any) -> bool:
        """
        Envía datos via WebSocket.

        Args:
            data: Datos a enviar

        Returns:
            True si el envío fue exitoso
        """
        if not self.connected:
            raise NetworkError("WebSocket not connected", "NET_005")

        try:
            self.ws.send(json.dumps(data))
            self.notify_callbacks("send_success", data)
            return True
        except Exception as e:
            raise NetworkError(f"WebSocket send failed: {str(e)}", "NET_006")

    def receive(self) -> Optional[Any]:
        """
        Recibe datos via WebSocket.
        Este método no se usa directamente ya que WebSocket es asíncrono.
        Los datos se reciben en el callback _on_message.
        """
        return None

    def _on_message(self, ws: Any, message: str) -> None:
        """Callback para mensajes WebSocket."""
        try:
            data = json.loads(message)
            self.notify_callbacks("receive_success", data)
        except json.JSONDecodeError:
            logger.error("Failed to parse WebSocket message")

    def _on_error(self, ws: Any, error: str) -> None:
        """Callback para errores WebSocket."""
        logger.error(f"WebSocket error: {error}")
        self.notify_callbacks("error", error)

    def _on_close(self, ws: Any) -> None:
        """Callback para cierre de conexión WebSocket."""
        self.connected = False
        logger.info("WebSocket connection closed")
        self.notify_callbacks("close", None)


class AdapterFactory:
    """Fábrica para crear adaptadores."""

    @staticmethod
    def create_adapter(adapter_type: str, **kwargs) -> BaseAdapter:
        """
        Crea un adaptador del tipo especificado.

        Args:
            adapter_type: Tipo de adaptador ('http' o 'websocket')
            **kwargs: Argumentos para el adaptador

        Returns:
            Instancia del adaptador
        """
        if adapter_type == "http":
            return HTTPAdapter(kwargs.get("base_url", ""))
        elif adapter_type == "websocket":
            return WebSocketAdapter(kwargs.get("url", ""))
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")


# Factory instance
adapter_factory = AdapterFactory()
