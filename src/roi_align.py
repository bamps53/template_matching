from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple, Union
import torch
import matplotlib.pyplot as plt
from detectron2.layers.roi_align_rotated import roi_align_rotated
from feature_extractor import FeatureMaps

from shape.window import Window


@dataclass
class MultiScaleFeatures:
    features2: torch.Tensor
    features4: torch.Tensor
    features8: torch.Tensor
    features16: torch.Tensor

    def visualize(self) -> None:
        _, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(self.features2[0].mean(0).numpy())
        axes[1].imshow(self.features4[0].mean(0).numpy())
        axes[2].imshow(self.features8[0].mean(0).numpy())
        axes[3].imshow(self.features16[0].mean(0).numpy())

    def __getitem__(self, index: int) -> torch.Tensor:
        if index == 0:
            return self.features2
        elif index == 1:
            return self.features4
        elif index == 2:
            return self.features8
        elif index == 3:
            return self.features16
        else:
            raise IndexError()


class RoIFeatureExtractor(ABC):

    @abstractmethod
    def extract(self, feature_map: torch.Tensor, rois: Union[torch.Tensor, Window], spatial_scale: float) -> torch.Tensor:
        pass

    def multi_extract(self, feature_maps: FeatureMaps, rois: Union[torch.Tensor, Window]) -> MultiScaleFeatures:
        features2 = self.extract(feature_maps.map2, rois, 1.0 / 2)
        features4 = self.extract(feature_maps.map4, rois, 1.0 / 4)
        features8 = self.extract(feature_maps.map8, rois, 1.0 / 8)
        features16 = self.extract(feature_maps.map16, rois, 1.0 / 16)
        return MultiScaleFeatures(features2, features4, features8, features16)


class RoIAlignFeatureExtractor(RoIFeatureExtractor):
    ANGLE_INDEX = 5

    def __init__(self, output_size: Union[int, Tuple[int]], sampling_ratio: int, device='cpu'):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.device = device

    def extract(self, feature_map: torch.Tensor, rois: Union[torch.Tensor, Window], spatial_scale: float) -> torch.Tensor:
        if isinstance(rois, Window):
            rois = rois.as_roi()
        rois = rois.clone()
        rois = rois.to(self.device)
        n = len(rois)
        rois[:, self.ANGLE_INDEX] *= -1  # convert to counter-clockwise
        features: torch.Tensor = roi_align_rotated(
            feature_map, rois, self.output_size, spatial_scale, self.sampling_ratio).reshape(n, -1)
        return features


class CenterFeatureExtractor(RoIFeatureExtractor):
    def extract(self, feature_map: torch.Tensor, rois: Union[torch.Tensor, Window], spatial_scale: float) -> torch.Tensor:
        if isinstance(rois, torch.Tensor):
            raise NotImplementedError()
        center = rois.get_center(spatial_scale)
        features = feature_map[:, :, center.cy, center.cx]
        return features
