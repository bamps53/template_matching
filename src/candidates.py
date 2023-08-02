from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

import torch

from shape.basic import Size


@dataclass
class NumSteps:
    """
    Number of steps for each dimension.
    """
    angle: int
    x: int
    y: int


class Candidates:
    """
    Generate candidate rois for a feature map.
    It contains all posible rois for a feature map.
    """

    def __init__(self, template_size: Size, image_size: Size, angle_step: int, x_step: int = 1, y_step: int = 1):
        self.template_size = template_size
        self.image_size = image_size
        self.angle_step = angle_step
        self.x_step = x_step
        self.y_step = y_step
        self.rois = self._generate_rois()

    def _generate_rois(self) -> torch.Tensor:
        # TODO make it faster
        # angles = torch.arange(0, 360, self.angle_step)
        # h_coords = torch.arange(0, self.height)
        # w_coords = torch.arange(0, self.width)

        # grid = torch.meshgrid(angles, h_coords, w_coords, indexing='ij')
        # rois = torch.stack(grid + [torch.tensor(self.height), torch.tensor(self.width)], dim=-1)
        # batch_dim = torch.zeros((len(rois, 1)))
        # rois = torch.cat([batch_dim, rois], dim=-1)
        # return rois

        rois = []
        batch_index = 0
        for cy in range(0, self.image_size.height, self.y_step):
            for cx in range(0, self.image_size.width, self.x_step):
                for angle in range(0, 360, self.angle_step):
                    rois.append([batch_index, cx, cy, self.template_size.width, self.template_size.height, angle])
        return torch.Tensor(rois).float()

    def get_num_steps(self) -> NumSteps:
        num_angle_steps = 360 // self.angle_step
        num_x_steps = self.image_size.width // self.x_step
        num_y_steps = self.image_size.height // self.y_step
        return NumSteps(num_angle_steps, num_x_steps, num_y_steps)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.rois[index, 1:].cpu().numpy()
    
    @classmethod
    def empty(cls) -> Candidates:
        return Candidates(Size(0, 0), Size(0, 0), 0)
