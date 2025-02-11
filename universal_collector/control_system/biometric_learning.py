"""
Sistema de Aprendizaje Biométrico Adaptativo
Aprende y se adapta a los cambios naturales del operador
"""

from typing import Dict, Any, Optional, List
import numpy as np
from dataclasses import dataclass
import time
import logging
from datetime import datetime


@dataclass
class BiometricProfile:
    facial_data: np.ndarray  # Datos faciales actuales
    voice_pattern: np.ndarray  # Patrón de voz actual
    facial_history: List[np.ndarray]  # Historial de cambios faciales
    voice_history: List[np.ndarray]  # Historial de cambios de voz
    last_update: float
    creation_date: datetime


class BiometricLearning:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._profile: Optional[BiometricProfile] = None
        self._learning_active = False
        self._change_threshold = 0.15  # Umbral de cambio aceptable
        self._min_confidence = 0.95  # Confianza mínima para actualización

    def initialize_profile(
        self, facial_data: np.ndarray, voice_pattern: np.ndarray
    ) -> None:
        """
        Inicializa el perfil biométrico por primera vez.
        Solo se ejecuta en la primera instalación.
        """
        self._profile = BiometricProfile(
            facial_data=facial_data,
            voice_pattern=voice_pattern,
            facial_history=[facial_data],
            voice_history=[voice_pattern],
            last_update=time.time(),
            creation_date=datetime.now(),
        )
        self._learning_active = True
        self.logger.info("Perfil biométrico inicial creado")

    def verify_and_learn(
        self, facial_data: np.ndarray, voice_pattern: np.ndarray
    ) -> bool:
        """
        Verifica los datos biométricos y aprende de los cambios graduales.
        """
        if not self._profile:
            return False

        # Verificación inicial
        facial_match = self._verify_facial(facial_data)
        voice_match = self._verify_voice(voice_pattern)

        if facial_match and voice_match:
            # Analizar cambios graduales
            self._analyze_and_update(facial_data, voice_pattern)
            return True
        return False

    def _verify_facial(self, facial_data: np.ndarray) -> bool:
        """
        Verifica los datos faciales considerando cambios graduales.
        """
        if not self._profile:
            return False

        # Comparación con el perfil actual
        main_similarity = self._calculate_facial_similarity(
            facial_data, self._profile.facial_data
        )

        # Comparación con el historial reciente
        historical_match = any(
            self._calculate_facial_similarity(facial_data, historical)
            > self._min_confidence
            for historical in self._profile.facial_history[-5:]  # Últimos 5 registros
        )

        return main_similarity > self._min_confidence or historical_match

    def _verify_voice(self, voice_pattern: np.ndarray) -> bool:
        """
        Verifica el patrón de voz considerando cambios naturales.
        """
        if not self._profile:
            return False

        # Comparación con el perfil actual
        main_similarity = self._calculate_voice_similarity(
            voice_pattern, self._profile.voice_pattern
        )

        # Comparación con el historial reciente
        historical_match = any(
            self._calculate_voice_similarity(voice_pattern, historical)
            > self._min_confidence
            for historical in self._profile.voice_history[-5:]
        )

        return main_similarity > self._min_confidence or historical_match

    def _analyze_and_update(
        self, facial_data: np.ndarray, voice_pattern: np.ndarray
    ) -> None:
        """
        Analiza y actualiza el perfil con cambios graduales detectados.
        """
        if not self._profile or not self._learning_active:
            return

        current_time = time.time()
        time_since_update = current_time - self._profile.last_update

        # Analizar cambios faciales
        facial_change = self._calculate_facial_change(facial_data)
        if facial_change < self._change_threshold:
            self._update_facial_profile(facial_data)

        # Analizar cambios de voz
        voice_change = self._calculate_voice_change(voice_pattern)
        if voice_change < self._change_threshold:
            self._update_voice_profile(voice_pattern)

        # Actualizar timestamp
        self._profile.last_update = current_time

    def _update_facial_profile(self, facial_data: np.ndarray) -> None:
        """
        Actualiza el perfil facial con nuevos datos.
        """
        if not self._profile:
            return

        # Actualizar datos principales
        self._profile.facial_data = facial_data

        # Mantener historial
        self._profile.facial_history.append(facial_data)
        if len(self._profile.facial_history) > 10:  # Mantener últimos 10
            self._profile.facial_history.pop(0)

    def _update_voice_profile(self, voice_pattern: np.ndarray) -> None:
        """
        Actualiza el perfil de voz con nuevos datos.
        """
        if not self._profile:
            return

        # Actualizar datos principales
        self._profile.voice_pattern = voice_pattern

        # Mantener historial
        self._profile.voice_history.append(voice_pattern)
        if len(self._profile.voice_history) > 10:  # Mantener últimos 10
            self._profile.voice_history.pop(0)

    def _calculate_facial_change(self, facial_data: np.ndarray) -> float:
        """
        Calcula el grado de cambio facial respecto al perfil actual.
        """
        if not self._profile:
            return 1.0

        return 1.0 - self._calculate_facial_similarity(
            facial_data, self._profile.facial_data
        )

    def _calculate_voice_change(self, voice_pattern: np.ndarray) -> float:
        """
        Calcula el grado de cambio en la voz respecto al perfil actual.
        """
        if not self._profile:
            return 1.0

        return 1.0 - self._calculate_voice_similarity(
            voice_pattern, self._profile.voice_pattern
        )

    def _calculate_facial_similarity(
        self, data1: np.ndarray, data2: np.ndarray
    ) -> float:
        """
        Calcula la similitud entre dos conjuntos de datos faciales.
        """
        # Implementar comparación avanzada de características faciales
        return np.mean(np.abs(data1 - data2))

    def _calculate_voice_similarity(
        self, pattern1: np.ndarray, pattern2: np.ndarray
    ) -> float:
        """
        Calcula la similitud entre dos patrones de voz.
        """
        # Implementar comparación avanzada de características de voz
        return np.mean(np.abs(pattern1 - pattern2))

    def get_profile_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del perfil biométrico.
        """
        if not self._profile:
            return {"status": "no_profile"}

        return {
            "status": "active",
            "creation_date": self._profile.creation_date,
            "last_update": datetime.fromtimestamp(self._profile.last_update),
            "facial_updates": len(self._profile.facial_history),
            "voice_updates": len(self._profile.voice_history),
            "learning_active": self._learning_active,
        }
