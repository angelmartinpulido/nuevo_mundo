"""
Sistema de seguridad centralizado con múltiples capas de protección.
"""
import os
import hashlib
import hmac
import base64
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from ..core.config import config
from ..core.logging_manager import logger_manager
from ..core.error_handler import handle_errors, SecurityError

logger = logger_manager.get_logger(__name__)


class SecurityManager:
    """Gestor central de seguridad."""

    def __init__(self):
        self.key_file = ".security/key.key"
        self.fernet: Optional[Fernet] = None
        self.session_tokens: Dict[str, datetime] = {}
        self.max_failed_attempts = 5
        self.failed_attempts: Dict[str, int] = {}
        self.initialize_security()

    @handle_errors
    def initialize_security(self) -> None:
        """Inicializa el sistema de seguridad."""
        if not os.path.exists(os.path.dirname(self.key_file)):
            os.makedirs(os.path.dirname(self.key_file))

        if not os.path.exists(self.key_file):
            self._generate_key()
        else:
            self._load_key()

    def _generate_key(self) -> None:
        """Genera una nueva clave de encriptación."""
        key = Fernet.generate_key()
        with open(self.key_file, "wb") as f:
            f.write(key)
        self.fernet = Fernet(key)

    def _load_key(self) -> None:
        """Carga la clave de encriptación existente."""
        with open(self.key_file, "rb") as f:
            key = f.read()
        self.fernet = Fernet(key)

    def encrypt(self, data: str) -> str:
        """
        Encripta datos.

        Args:
            data: Datos a encriptar

        Returns:
            Datos encriptados en formato base64
        """
        if not self.fernet:
            raise SecurityError("Security system not initialized", "SEC_001")
        return self.fernet.encrypt(data.encode()).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """
        Desencripta datos.

        Args:
            encrypted_data: Datos encriptados en formato base64

        Returns:
            Datos desencriptados
        """
        if not self.fernet:
            raise SecurityError("Security system not initialized", "SEC_001")
        return self.fernet.decrypt(encrypted_data.encode()).decode()

    def generate_token(self, user_id: str, expiration_hours: int = 24) -> str:
        """
        Genera un token de sesión.

        Args:
            user_id: ID del usuario
            expiration_hours: Horas de validez del token

        Returns:
            Token de sesión
        """
        expiration = datetime.now() + timedelta(hours=expiration_hours)
        token = base64.b64encode(os.urandom(32)).decode()
        self.session_tokens[token] = expiration
        return token

    def validate_token(self, token: str) -> bool:
        """
        Valida un token de sesión.

        Args:
            token: Token a validar

        Returns:
            True si el token es válido
        """
        if token not in self.session_tokens:
            return False

        expiration = self.session_tokens[token]
        if datetime.now() > expiration:
            del self.session_tokens[token]
            return False

        return True

    def hash_password(
        self, password: str, salt: Optional[str] = None
    ) -> tuple[str, str]:
        """
        Genera un hash seguro de una contraseña.

        Args:
            password: Contraseña a hashear
            salt: Salt opcional

        Returns:
            Tupla con (hash, salt)
        """
        if not salt:
            salt = base64.b64encode(os.urandom(16)).decode()

        hash_obj = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt.encode(), 100000
        )
        hash_str = base64.b64encode(hash_obj).decode()
        return hash_str, salt

    def verify_password(self, password: str, hash_str: str, salt: str) -> bool:
        """
        Verifica una contraseña contra su hash.

        Args:
            password: Contraseña a verificar
            hash_str: Hash almacenado
            salt: Salt usado

        Returns:
            True si la contraseña es correcta
        """
        new_hash, _ = self.hash_password(password, salt)
        return hmac.compare_digest(new_hash.encode(), hash_str.encode())

    def check_failed_attempts(self, user_id: str) -> bool:
        """
        Verifica si un usuario ha excedido los intentos fallidos.

        Args:
            user_id: ID del usuario

        Returns:
            True si el usuario está bloqueado
        """
        return self.failed_attempts.get(user_id, 0) >= self.max_failed_attempts

    def record_failed_attempt(self, user_id: str) -> None:
        """
        Registra un intento fallido de autenticación.

        Args:
            user_id: ID del usuario
        """
        self.failed_attempts[user_id] = self.failed_attempts.get(user_id, 0) + 1
        if self.check_failed_attempts(user_id):
            logger.warning(f"User {user_id} blocked due to too many failed attempts")

    def reset_failed_attempts(self, user_id: str) -> None:
        """
        Reinicia el contador de intentos fallidos.

        Args:
            user_id: ID del usuario
        """
        if user_id in self.failed_attempts:
            del self.failed_attempts[user_id]

    def validate_password_strength(self, password: str) -> bool:
        """
        Valida la fortaleza de una contraseña.

        Args:
            password: Contraseña a validar

        Returns:
            True si la contraseña cumple los requisitos
        """
        if len(password) < config.get("security.min_password_length", 12):
            return False

        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(not c.isalnum() for c in password)

        return all([has_upper, has_lower, has_digit, has_special])


# Singleton instance
security_manager = SecurityManager()
