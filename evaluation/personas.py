"""Map saved K-Means cluster labels to reusable customer personas."""

from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path
from typing import Iterable, TypedDict


class Persona(TypedDict, total=False):
    name: str
    description: str
    behavior: str
    spending: str
    loyalty: str
    churn_risk: str
    marketing_action: str
    assignment_rule: str


PERSONAS_PATH = Path(__file__).resolve().parent / "results" / "personas.json"
with PERSONAS_PATH.open(encoding="utf-8") as persona_file:
    _PERSONA_DATA = json.load(persona_file)

_CLUSTER_PERSONAS: dict[str, Persona] = _PERSONA_DATA["personas"]
_NOISE_PERSONA: Persona = _PERSONA_DATA["noise"]


def get_persona(cluster_label: int) -> Persona:
    """Return persona details for a saved K-Means label or DBSCAN noise label.

    Cluster IDs are model-specific and must come from the corresponding saved
    four-cluster K-Means model. The DBSCAN label -1 maps to the noise profile.
    """
    if isinstance(cluster_label, bool) or not isinstance(cluster_label, Integral):
        raise TypeError("cluster_label must be an integer.")
    normalized_label = int(cluster_label)
    if normalized_label == -1:
        return _NOISE_PERSONA.copy()

    persona = _CLUSTER_PERSONAS.get(str(normalized_label))
    if persona is None:
        valid_labels = ", ".join(sorted(_CLUSTER_PERSONAS))
        raise ValueError(
            f"Unknown K-Means cluster label {normalized_label}; expected one of {valid_labels}."
        )
    return persona.copy()


def assign_personas(cluster_labels: Iterable[int]) -> list[str]:
    """Map a sequence of cluster labels to persona names."""
    return [get_persona(label)["name"] for label in cluster_labels]
