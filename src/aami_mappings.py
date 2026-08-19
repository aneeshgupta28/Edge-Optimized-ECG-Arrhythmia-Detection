"""
AAMI Beat-Class Mappings
========================
Maps MIT-BIH annotation symbols → 5 AAMI superclasses per:
  de Chazal et al., IEEE TBME 2004
  ANSI/AAMI EC57:2012

AAMI Classes
------------
  N  — Normal and bundle-branch-block beats
  S  — Supraventricular ectopic beats (SVEB)
  V  — Ventricular ectopic beats (VEB)
  F  — Fusion of ventricular and normal beats
  Q  — Unknown / paced / unclassifiable

Symbols that do NOT appear in this map are discarded (e.g., rhythm annotations,
noise markers, non-beat entries that wfdb returns with symbols like '+', '~', etc.)
"""

from __future__ import annotations

# Mapping: MIT-BIH symbol to AAMI class

SYMBOL_TO_AAMI: dict[str, str] = {
    # N — Normal & bundle branch block
    "N":  "N",   # Normal beat
    "L":  "N",   # Left bundle branch block beat
    "R":  "N",   # Right bundle branch block beat
    "e":  "N",   # Atrial escape beat
    "j":  "N",   # Nodal (junctional) escape beat

    # S — Supraventricular ectopic beats
    "A":  "S",   # Atrial premature beat
    "a":  "S",   # Aberrated atrial premature beat
    "J":  "S",   # Nodal (junctional) premature beat
    "S":  "S",   # Supraventricular premature beat

    # V — Ventricular ectopic beats
    "V":  "V",   # Premature ventricular contraction
    "E":  "V",   # Ventricular escape beat

    # F — Fusion beats
    "F":  "F",   # Fusion of ventricular and normal beat

    # Q — Unknown / paced / unclassifiable
    "/":  "Q",   # Paced beat
    "f":  "Q",   # Fusion of paced and normal beat
    "Q":  "Q",   # Unclassifiable beat
}

# Integer label encoding (for sklearn / XGBoost)
AAMI_CLASSES: list[str] = ["N", "S", "V", "F", "Q"]

AAMI_TO_INT: dict[str, int] = {cls: i for i, cls in enumerate(AAMI_CLASSES)}
INT_TO_AAMI: dict[int, str] = {i: cls for cls, i in AAMI_TO_INT.items()}

# Clinically important classes (used for emphasis in reporting)
CLINICAL_FOCUS_CLASSES: list[str] = ["S", "V"]


def symbol_to_aami(symbol: str) -> str | None:
    """Convert a MIT-BIH annotation symbol to its AAMI class.

    Returns None if the symbol is not a beat annotation (rhythm markers,
    noise markers, etc. should be ignored during feature extraction).
    """
    return SYMBOL_TO_AAMI.get(symbol, None)


def symbol_to_int(symbol: str) -> int | None:
    """Convert a MIT-BIH annotation symbol directly to integer label.

    Returns None if symbol is not a recognized beat annotation.
    """
    aami = symbol_to_aami(symbol)
    if aami is None:
        return None
    return AAMI_TO_INT[aami]


def aami_to_int(aami_class: str) -> int:
    return AAMI_TO_INT[aami_class]


def int_to_aami(label: int) -> str:
    return INT_TO_AAMI[label]
