from abc import ABC, abstractmethod

import numpy as np
from feature_extractor import FeatureExtractor, FeatureMap
from shape import Position, Window


class Matcher(ABC):
    """
    Class to execute image template matching.
    It takes template image and target image as input.
    It returns the coordinates and rotation angle.
    """
    @abstractmethod
    def set_template(self, template: np.ndarray, window: Window) -> None:
        self.template = template
        self.window = window

    @abstractmethod
    def find(self, target: np.ndarray) -> Position:
        """
        Executes template matching and returns the coordinates and rotation angle.
        """
        return Position()


class CnnMatcher(Matcher):
    """
    Template matching using CNN.
    """

    def __init__(self, feature_extractor: FeatureExtractor) -> None:
        self.feature_extractor = feature_extractor
        self._feature_map: FeatureMap = None

    def set_template(self, template: np.ndarray, window: Window) -> None:
        super().set_template(template, window)
        self._feature_map = self.feature_extractor(template)

    def find(self, target: np.ndarray) -> Position:
        target_feature_map = self.feature_extractor(target)
        return Position()

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import ndimage
# from skimage.transform import pyramid_gaussian

# # テンプレートマッチングの関数
# def template_matching(tmpl, img):
#     h, w = tmpl.shape
#     H, W = img.shape
#     res = np.zeros((H-h+1, W-w+1))
#     for y in range(H-h+1):
#         for x in range(W-w+1):
#             res[y, x] = np.sum((img[y:y+h, x:x+w] - tmpl)**2)
#     return res

# # ピラミッドサーチと回転を組み合わせた探索
# def pyramid_and_rotation_search(img, template, rotation_angles):
#     min_val = float('inf')
#     min_loc = (0, 0)
#     min_scale = 0
#     min_angle = 0
#     for angle in rotation_angles:
#         rotated_template = ndimage.rotate(template, angle)
#         for scale, pimg in enumerate(pyramid_gaussian(img)):
#             result = template_matching(rotated_template, pimg)
#             min_val_tmp = np.min(result)
#             if min_val_tmp < min_val:
#                 min_val = min_val_tmp
#                 min_loc = np.unravel_index(np.argmin(result), result.shape)
#                 min_scale = scale
#                 min_angle = angle
#     return min_val, min_loc, min_scale, min_angle

# # 画像とテンプレートのロード（適切にパスを置き換えてください）
# img = plt.imread("path_to_your_image.png")
# template = plt.imread("path_to_your_template.png")

# # グレースケールに変換
# img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
# template = np.dot(template[...,:3], [0.2989, 0.5870, 0.1140])

# # 回転角度のリスト（ここでは0から360度までの全ての整数値）
# rotation_angles = list(range(360))

# # ピラミッドサーチと回転を組み合わせた探索を実行
# min_val, min_loc, min_scale, min_angle = pyramid_and_rotation_search(img, template, rotation_angles)

# print(f"Min Value: {min_val}")
# print(f"Min Location: {min_loc}")
# print(f"Min Scale: {min_scale}")
# print(f"Min Angle: {min_angle}")
