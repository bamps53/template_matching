from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Position:
    cx: float = 0.0
    cy: float = 0.0
    angle: float = 0.0

@dataclass
class Size:
    height: float
    width: float

    def as_int(self) -> Size:
        return Size(int(self.height + 0.5), int(self.width + 0.5))

@dataclass
class Coords:
    cx: int
    cy: int

    def __post_init__(self):
        self.cx = int(self.cx + 0.5)
        self.cy = int(self.cy + 0.5)

@dataclass
class Point:
    cx: float
    cy: float