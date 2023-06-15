from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class Window:
    cx: float = 0.0
    cy: float = 0.0
    height: float = 0.0
    width: float = 0.0
    angle: float = 0.0

    @classmethod
    def from_array(cls, array: np.ndarray) -> 'Window':
        assert array.shape == (5, )
        return cls(
            cx=array[0],
            cy=array[1],
            height=array[2],
            width=array[3],
            angle=array[4],
        )

    def as_boxPoints(self):
        rotated_rect = ((self.cx, self.cy), (self.width, self.height), self.angle)
        # Compute the four vertices of the rectangle
        box = cv2.boxPoints(rotated_rect)
        box = np.int0(box)  # Convert the coordinates to integers
        return box
