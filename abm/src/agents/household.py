"""Household agent implementation for MacroEconVue ABM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import src.config as config


@dataclass
class HouseholdAgent:
    """Represents a single household participating in the ABM."""
