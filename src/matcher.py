from abc import ABC, abstractmethod

import numpy as np
import torch
from candidates import Candidates
from feature_extractor import CNN, FeatureMaps
from nms import NMS
from roi_align import MultiScaleFeatures, RoIAlignFeatureExtractor
from scorer import CosineSimilarityScorer
from shape import Position, Window


from dataclasses import dataclass
from typing import List, Optional
from shape.basic import Size
from shape.window import DetectedWindow

from utils.timer import timer, timer_decorator


@dataclass
class SearchParams:
    x_step: int
    y_step: int
    angle_step: int
    score_threshold: float
    iou_threshold: float


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
        self.template_size = window.get_size()

    @abstractmethod
    def find(self, target: np.ndarray) -> Position:
        """
        Executes template matching and returns the coordinates and rotation angle.
        """
        return Position()


class NaiveMatcher(Matcher):
    """
    Template matching using brute force search.
    """
    scales = [2, 4, 8, 16]

    def __init__(self,
                 feature_extractor: CNN,
                 roi_feature_extractor: RoIAlignFeatureExtractor,
                 scorer: CosineSimilarityScorer,
                 scale: int = 1,
                 search_params: Optional[SearchParams] = None) -> None:
        self.feature_extractor = feature_extractor
        self.roi_feature_extractor = roi_feature_extractor
        self.scorer = scorer

        self.source_feature_maps: FeatureMaps = None
        self.template_features: MultiScaleFeatures = None

        self.scale = scale
        if search_params is None:
            self.search_params = SearchParams(x_step=1, y_step=1, angle_step=1, score_threshold=0.5, iou_threshold=0.1)
        else:
            self.search_params = search_params
        self.nms = NMS(self.search_params.score_threshold, self.search_params.iou_threshold)

    @timer_decorator
    def set_template(self, template: np.ndarray, window: Window) -> None:
        super().set_template(template, window)
        self.source_feature_maps = self.feature_extractor(template)
        self.template_features = self.roi_feature_extractor.multi_extract(self.source_feature_maps, window)

    @timer_decorator
    def find(self, target_image: np.ndarray) -> List[DetectedWindow]:
        assert self.source_feature_maps is not None
        assert self.template_features is not None

        target_feature_maps = self.feature_extractor(target_image)
        target_image_size = Size(*target_image.shape[:2])

        rois = Candidates(self.template_size,
                          target_image_size,
                          angle_step=self.search_params.angle_step,
                          x_step=self.search_params.x_step,
                          y_step=self.search_params.y_step)
        with timer('calc_roi_features'):
            index = self.scales.index(self.scale)
            roi_features = self.roi_feature_extractor.extract(
                target_feature_maps[index], rois.rois, spatial_scale=1.0 / self.scale)

        with timer('calc_scores'):
            scores = self.scorer.score(self.template_features[index], roi_features)

        nms_rois, nms_scores = self.nms(rois, scores)
        nms_rois = nms_rois.cpu().numpy()
        nms_scores = nms_scores.cpu().numpy()
        return [DetectedWindow.from_array(roi, score) for roi, score in zip(nms_rois, nms_scores)]

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
