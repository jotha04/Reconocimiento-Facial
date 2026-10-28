"""
Normalizador de Texto para Síntesis de Voz (TTSTextNormalizer).
Transforma exclusivamente la representación superficial del texto para que
los analizadores G2P de TTS (como espeak-ng / phonemizer / Kokoro) pronuncien correctamente
los nombres y términos propios del proyecto sin alterar la lógica de negocio ni la identidad.
"""

import re
from typing import Dict, Optional, Tuple, List, Pattern


class TTSTextNormalizer:
    """
    Normalizador ortográfico y fonético superficial para el subsistema de audio.
    Soporta excepciones de pronunciación explícitas basadas en diccionario/configuración
    con respeto de mayúsculas/minúsculas y preservación de límites de palabra.
    """

    VERSION = "v1"

    # Excepciones base validadas para Kokoro
    DEFAULT_OVERRIDES: Dict[str, str] = {
        "AudacIA": "Audacia",
        "Jonathan": "Yónatan",
        "Joshua": "Yóshua",
    }

    # Para retrocompatibilidad
    DEFAULT_REPLACEMENTS = DEFAULT_OVERRIDES

    def __init__(
        self,
        overrides: Optional[Dict[str, str]] = None,
        custom_replacements: Optional[Dict[str, str]] = None
    ):
        self.version = self.VERSION
        self._overrides = dict(self.DEFAULT_OVERRIDES)
        if overrides:
            self._overrides.update(overrides)
        if custom_replacements:
            self._overrides.update(custom_replacements)

        # Pre-compilar patrones regex para máxima eficiencia
        self._compiled_rules: List[Tuple[Pattern, str, str]] = []
        for word_pattern, target in self._overrides.items():
            # Si ya incluye límites \b explícitos, usarlos; de lo contrario envolver con \b
            if word_pattern.startswith(r"\b") and word_pattern.endswith(r"\b"):
                clean_key = word_pattern[2:-2]
                pattern = re.compile(word_pattern, re.IGNORECASE)
            else:
                clean_key = word_pattern
                pattern = re.compile(rf"\b{re.escape(word_pattern)}\b", re.IGNORECASE)
            self._compiled_rules.append((pattern, target, clean_key))

    @staticmethod
    def _replace_match(match: re.Match, target: str, orig_key: str) -> str:
        matched_str = match.group(0)
        # Si es coincidencia exacta con la clave original, retornar target tal cual
        if matched_str == orig_key:
            return target
        # Todo en mayúsculas
        if matched_str.isupper():
            return target.upper()
        # Todo en minúsculas
        if matched_str.islower():
            return target.lower()
        # Titlecase (primera mayúscula, resto minúsculas)
        if matched_str[0].isupper() and (len(matched_str) == 1 or matched_str[1:].islower()):
            return target[0].upper() + target[1:] if target else ""
        return target

    def normalize(self, text: str) -> str:
        """
        Normaliza el texto lógico de entrada produciendo la versión fonética para TTS.
        Si la entrada está vacía o no contiene términos especiales, retorna el texto inalterado.
        """
        if not text:
            return ""

        normalized = text
        for pattern, target, orig_key in self._compiled_rules:
            normalized = pattern.sub(
                lambda m, t=target, k=orig_key: self._replace_match(m, t, k),
                normalized
            )

        return normalized
