"""
Motor de Decisión de Saludos (GreetingDecisionEngine).
Aplica las reglas fundamentales de saludo de AudacIA:
1. Una persona conocida -> Saludo personalizado con su nombre.
2. N personas conocidas (todas conocidas) -> Saludo personalizado con lista natural en español.
3. Grupo mixto (conocidas + desconocidas) -> Saludo plural genérico (sin mencionar solo a los conocidos).
4. Todas desconocidas (1 o N) -> Saludo genérico (singular o plural).
"""

import re
from typing import List, Tuple
from audacia.app.interaction.session import Participant, GroupType
from audacia.app.attention.config import GreetingsConfig


def get_spanish_conjunction(next_word: str) -> str:
    """
    Determina si corresponde usar 'y' o 'e' según la fonética del español.
    Se usa 'e' cuando la palabra siguiente comienza fonéticamente con el sonido /i/
    (i- o hi- seguido de consonante), excepto diptongos como 'hie-' o 'ia-'.
    """
    clean = next_word.strip().lower()
    if not clean:
        return "y"
    # Excepciones con diptongo inicial (ej. Hierro, Hielo, Iatrogenia) usan 'y'
    if clean.startswith("hie") or clean.startswith("hia") or clean.startswith("hio"):
        return "y"
    if clean.startswith("ie") or clean.startswith("ia") or clean.startswith("io"):
        return "y"
    # Comienza con 'i' o 'hi'
    if clean.startswith("i") or clean.startswith("hi"):
        return "e"
    return "y"


def format_names_spanish(names: List[str]) -> str:
    """
    Formatea una lista de N nombres en una cadena natural en español:
    - ["Jonathan"] -> "Jonathan"
    - ["Jonathan", "María"] -> "Jonathan y María"
    - ["Jonathan", "Ignacio"] -> "Jonathan e Ignacio"
    - ["Jonathan", "María", "Carlos"] -> "Jonathan, María y Carlos"
    """
    cleaned = [n.strip() for n in names if n.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        conj = get_spanish_conjunction(cleaned[1])
        return f"{cleaned[0]} {conj} {cleaned[1]}"

    # 3 o más nombres
    conj = get_spanish_conjunction(cleaned[-1])
    prefix = ", ".join(cleaned[:-1])
    return f"{prefix} {conj} {cleaned[-1]}"


class GreetingDecisionEngine:
    """
    Clasifica la composición del grupo confirmado y produce el saludo exacto.
    """

    def __init__(self, config: GreetingsConfig):
        self.config = config

    def classify_group(self, participants: List[Participant]) -> GroupType:
        """Determina la categoría del grupo según las identidades de sus participantes."""
        if not participants:
            return GroupType.SINGLE_UNKNOWN

        knowns = [p for p in participants if p.is_known]
        unknowns = [p for p in participants if not p.is_known]

        total = len(participants)

        if len(knowns) == total:
            return GroupType.SINGLE_KNOWN if total == 1 else GroupType.MULTIPLE_KNOWN
        elif len(unknowns) == total:
            return GroupType.SINGLE_UNKNOWN if total == 1 else GroupType.MULTIPLE_UNKNOWN
        else:
            return GroupType.MIXED

    def generate_greeting(self, participants: List[Participant]) -> Tuple[GroupType, str]:
        """
        Produce el tipo de grupo y el texto exacto del saludo basado en los participantes.
        """
        group_type = self.classify_group(participants)

        if group_type == GroupType.SINGLE_KNOWN:
            name = participants[0].identity_label
            text = self.config.single_known.format(name=name)

        elif group_type == GroupType.MULTIPLE_KNOWN:
            names_list = [p.identity_label for p in participants]
            formatted_names = format_names_spanish(names_list)
            text = self.config.multiple_known.format(names=formatted_names)

        elif group_type == GroupType.SINGLE_UNKNOWN:
            text = self.config.single_unknown

        elif group_type == GroupType.MULTIPLE_UNKNOWN:
            text = self.config.multiple_unknown

        elif group_type == GroupType.MIXED:
            # Regla estricta: NO mencionar solamente al conocido si hay desconocidos en el grupo
            text = self.config.mixed_group

        else:
            text = self.config.single_unknown

        return group_type, text
