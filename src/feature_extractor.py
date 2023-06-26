from __future__ import annotations
from abc import ABC, abstractmethod
import cv2
import numpy as np
import timm
import torch
from torchvision.transforms import Compose, ToTensor, Normalize

from matplotlib import pyplot as plt

from dataclasses import dataclass
from typing import List, Optional, Union
from models.re_resnet import build_re_resnet50

from shape.window import Window


@dataclass
class FeatureMaps:
    map2: torch.Tensor = torch.empty((1, 3, 0, 0))
    map4: torch.Tensor = torch.empty((1, 3, 0, 0))
    map8: torch.Tensor = torch.empty((1, 3, 0, 0))
    map16: torch.Tensor = torch.empty((1, 3, 0, 0))

    def __post_init__(self):
        print(f'map2  mean: {self.map2.mean():.3f} std: {self.map2.std():.3f}')
        print(f'map4  mean: {self.map4.mean():.3f} std: {self.map4.std():.3f}')
        print(f'map8  mean: {self.map8.mean():.3f} std: {self.map8.std():.3f}')
        print(f'map16 mean: {self.map16.mean():.3f} std: {self.map16.std():.3f}')

    @classmethod
    def from_outputs(cls, outputs: List[torch.Tensor]) -> FeatureMaps:
        output0 = outputs[0]
        output1 = outputs[1]
        output2 = outputs[2]
        output3 = outputs[3]

        w0 = output0.shape[-1]
        w1 = output1.shape[-1]
        w2 = output2.shape[-1]
        w3 = output3.shape[-1]
        assert w0 == w1 * 2, (w0, w1, w2, w3)
        assert w1 == w2 * 2, (w0, w1, w2, w3)
        assert w2 == w3 * 2, (w0, w1, w2, w3)

        return cls(output0, output1, output2, output3)

    def visualize(self) -> None:
        _, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(self.map2[0].mean(0).cpu().numpy())
        axes[1].imshow(self.map4[0].mean(0).cpu().numpy())
        axes[2].imshow(self.map8[0].mean(0).cpu().numpy())
        axes[3].imshow(self.map16[0].mean(0).cpu().numpy())
        plt.show()

    def __getitem__(self, index: int) -> torch.Tensor:
        if index == 0:
            return self.map2
        elif index == 1:
            return self.map4
        elif index == 2:
            return self.map8
        elif index == 3:
            return self.map16
        else:
            raise IndexError()


def get_transform() -> Compose:
    return Compose([
        ToTensor(),
        Normalize(
            mean=torch.tensor((0.5, 0.5, 0.5)),
            std=torch.tensor((0.5, 0.5, 0.5)))
    ])


class FeatureExtractor(ABC):
    @abstractmethod
    def __call__(self, image: Union[torch.Tensor, np.ndarray]) -> FeatureMaps:
        pass


class Preprocessor:
    """do image preprocessing"""

    def __init__(self, transform: Optional[Compose] = None, device: torch.device = 'cpu'):
        self.transform = transform or get_transform()
        self.device = device

    def __call__(self, image: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            assert image.ndim == 3
            assert image.dtype == np.uint8
            image = self.transform(image)[None]
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4
            assert image.shape[0] == 1
        return image.to(self.device)


class CNN(FeatureExtractor):
    """
    A wrapper around a CNN model that extracts feature maps from images.
    """

    def __init__(self, model_name: str = 'tf_efficientnet_lite0', transform: Optional[Compose] = None, device: torch.device = 'cpu', pretrained: bool = True):
        self.model = timm.create_model(model_name, features_only=True, pretrained=pretrained)
        self.model = self.model.to(device)
        self.model.eval()

        transform = transform or get_transform()
        self.preprocessor = Preprocessor(transform, device)
        self.device = device

    def __call__(self, image: Union[torch.Tensor, np.ndarray]) -> FeatureMaps:
        image = self.preprocessor(image)
        with torch.no_grad():
            outputs = self.model(image)
        return FeatureMaps.from_outputs(outputs)


class ReCNN(FeatureExtractor):
    """
    A wrapper around a CNN model that extracts feature maps from images.
    """

    def __init__(self, model_path: str = '../models/re_resnet50.pth', transform: Optional[Compose] = None, device: torch.device = 'cpu'):
        self.model = build_re_resnet50(model_path)
        self.model = self.model.to(device)
        self.model.eval()
        assert transform is None
        transform = Compose([
            ToTensor(),
            Normalize(
                mean=torch.tensor((0.485, 0.456, 0.406)),
                std=torch.tensor((0.229, 0.224, 0.225)))
        ])
        self.preprocessor = Preprocessor(transform, device)

        self.device = device

    def __call__(self, image: Union[torch.Tensor, np.ndarray]) -> FeatureMaps:
        image = self.preprocessor(image)
        with torch.no_grad():
            outputs = self.model(image)
        return FeatureMaps.from_outputs(outputs)


class ResizeImageFeatureExtractor(FeatureExtractor):
    """
    Feature extractor that returns resized images as feature maps.
    """

    def __init__(self, model_name: str = "", transform: Optional[Compose] = None, device: torch.device = 'cpu'):
        self.transform = transform or ToTensor()

    def __call__(self, image: Union[torch.Tensor, np.ndarray]) -> FeatureMaps:
        if isinstance(image, torch.Tensor):
            raise NotImplementedError()

        outputs = []
        h, w = image.shape[:2]
        for down_scale in [2, 4, 8, 16]:
            resized_image = cv2.resize(image, (w // down_scale, h // down_scale))
            resized_image = self.transform(resized_image)[None]
            outputs.append(resized_image)

        return FeatureMaps.from_outputs(outputs)


class EdgeImageFeatureExtractor(FeatureExtractor):
    """
    Feature extractor that returns resized images as feature maps.
    """

    def __init__(self, model_name: str = "", transform: Optional[Compose] = None, device: torch.device = 'cpu'):
        self.transform = transform or ToTensor()

    def __call__(self, image: Union[torch.Tensor, np.ndarray]) -> FeatureMaps:
        if isinstance(image, torch.Tensor):
            raise NotImplementedError()

        outputs = []
        h, w = image.shape[:2]
        for down_scale in [2, 4, 8, 16]:
            resized_image = cv2.resize(image, (w // down_scale, h // down_scale))
            resized_image = self.transform(resized_image)[None]
            outputs.append(resized_image)

        return FeatureMaps.from_outputs(outputs)
