import numpy as np
import timm
import torch
from torchvision.transforms import Compose, ToTensor, Normalize

from matplotlib import pyplot as plt

from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class FeatureMap:
    map2: torch.Tensor = torch.empty((1, 3, 0, 0))
    map4: torch.Tensor = torch.empty((1, 3, 0, 0))
    map8: torch.Tensor = torch.empty((1, 3, 0, 0))
    map16: torch.Tensor = torch.empty((1, 3, 0, 0))

    @classmethod
    def from_outputs(cls, outputs: List[torch.Tensor]):
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

    def visualize(self):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(self.map2[0].mean(0).numpy())
        axes[1].imshow(self.map4[0].mean(0).numpy())
        axes[2].imshow(self.map8[0].mean(0).numpy())
        axes[3].imshow(self.map16[0].mean(0).numpy())
        plt.show()


def get_transform() -> Compose:
    return Compose([
        ToTensor(),
        Normalize(
            mean=torch.tensor((0.5, 0.5, 0.5)),
            std=torch.tensor((0.5, 0.5, 0.5)))
    ])


class FeatureExtractor:
    def __init__(self, model_name: str, transform: Optional[Compose] = None, device: torch.device = 'cpu'):
        self.model = timm.create_model(model_name, features_only=True, pretrained=True)
        self.model.eval()
        self.transform = transform or get_transform()

    def _preprocess(self, image: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            assert image.ndim == 3
            assert image.dtype == np.uint8
            image = self.transform(image)[None]
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4
            assert image.shape[0] == 1  # single image only
        return image

    def __call__(self, image: Union[torch.Tensor, np.ndarray]) -> FeatureMap:
        image = self._preprocess(image)
        with torch.no_grad():
            outputs = self.model(image)
        return FeatureMap.from_outputs(outputs)
