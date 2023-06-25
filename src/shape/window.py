from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Tuple, Union
import numpy as np
import cv2
import torch

from shape.basic import Coords, Point, Size


@dataclass
class Window:
    cx: float = 0.0
    cy: float = 0.0
    width: float = 0.0
    height: float = 0.0
    angle: float = 0.0

    @classmethod
    def from_array(cls, array: Union[torch.Tensor, np.ndarray]) -> Window:
        if isinstance(array, torch.Tensor):
            array = array.numpy()
        assert array.shape == (5, )
        return cls(
            cx=array[0],
            cy=array[1],
            width=array[2],
            height=array[3],
            angle=array[4],
        )

    @classmethod
    def from_ltrba(cls, ltrba: np.ndarray) -> Window:
        cx = (ltrba[0] + ltrba[2]) / 2
        cy = (ltrba[1] + ltrba[3]) / 2
        w = ltrba[2] - ltrba[0]
        h = ltrba[3] - ltrba[1]
        a = ltrba[4]
        return cls(cx, cy, w, h, a)

    def as_boxPoints(self) -> np.ndarray:
        rotated_rect = ((self.cx, self.cy), (self.width, self.height), self.angle)
        # Compute the four vertices of the rectangle
        box = cv2.boxPoints(rotated_rect)
        box = box.astype(np.int64)
        return box

    def as_roi(self) -> torch.Tensor:
        batch_index = 0
        roi = torch.tensor([batch_index, self.cx, self.cy, self.width, self.height, self.angle]).float()
        return roi[None]

    def get_center(self, spatial_scale: float = 1.0) -> np.ndarray:
        return Coords(self.cx * spatial_scale, self.cy * spatial_scale)

    def get_size(self) -> Size:
        return Size(self.height, self.width)

    def rotate(self, rotation_center: Union[Coords, Point], theta: float) -> Window:
        """
        Rotate window(rectangle) around rotation center by theta degrees.
        Args:
            window: Window to rotate
            rotation_center: Center of rotation
            theta: Angle in degrees
        """
        # Convert theta to radians
        theta *= -1
        rad_theta = math.radians(theta)

        # Shift window center to origin
        shifted_x = self.cx - rotation_center.cx
        shifted_y = self.cy - rotation_center.cy

        # Apply rotation
        new_x = shifted_x * math.cos(rad_theta) - shifted_y * math.sin(rad_theta)
        new_y = shifted_x * math.sin(rad_theta) + shifted_y * math.cos(rad_theta)

        # Shift window center back
        new_x += rotation_center.cx
        new_y += rotation_center.cy

        # Add theta to the window's current rotation angle
        new_angle = self.angle + theta

        return Window(cx=new_x, cy=new_y, width=self.width, height=self.height, angle=new_angle)
