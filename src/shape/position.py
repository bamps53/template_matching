from dataclasses import dataclass
import numpy as np

@dataclass
class Position:
    cx: float = 0.0
    cy: float = 0.0
    angle: float = 0.0