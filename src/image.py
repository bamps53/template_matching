import torch
from PIL import Image
import cv2
import numpy as np
from typing import Union
from enum import Enum

class ImageFormat(Enum):
    RGB = 1
    BGR = 2

class ImageEngine(Enum):
    PIL = 1
    CV2 = 2
    NUMPY = 3
    TENSOR = 4

def load_image(image_path: str, format: ImageFormat = ImageFormat.RGB, engine: ImageEngine = ImageEngine.CV2) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    if engine == ImageEngine.PIL:
        image = Image.open(image_path)
        if format == ImageFormat.RGB:
            image = image.convert("RGB")
        elif format == ImageFormat.BGR:
            r, g, b = image.split()
            image = Image.merge("RGB", (b, g, r))
    elif engine == ImageEngine.CV2:
        image = cv2.imread(image_path)
        if format == ImageFormat.RGB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError('Invalid engine')

    return image



def get_image_tensor(img: Image) -> torch.Tensor:
    img = np.array(img)
    img = torch.from_numpy(img)
    img = img.permute(2, 0, 1)
    img = img.float()
    img = img.unsqueeze(0)
    return img

def get_numpy_image(img: torch.Tensor) -> np.ndarray:
    img = img.squeeze(0)
    img = img.permute(1, 2, 0)
    img = img.numpy()
    return img

def rotate_image(
    img: np.ndarray,
    angle: float,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
):
    height, width = img.shape[:2]
    # for images we use additional shifts of (0.5, 0.5) as otherwise
    # we get an ugly black border for 90deg rotations
    matrix = cv2.getRotationMatrix2D((width / 2 - 0.5, height / 2 - 0.5), angle, 1.0)

    return cv2.warpAffine(img, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=0)