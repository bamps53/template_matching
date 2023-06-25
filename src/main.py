from candidates import Candidates
from feature_extractor import CNN
from image import load_image
from matcher import CnnMatcher, Window
from roi_align import RoIAlignFeatureExtractor
from shape import Window, Position
from image import get_image_tensor, get_numpy_image
from drawing import draw_window
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
sys.path.insert(0, '../src')
sys.path.insert(0, '../')


img = load_image('../data/image_1.png')
template_img = load_image('../data/template_1.jpg')
cropped_template = template_img[60:220, 320:500, :].copy()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

window = Window(210, 160, 160, 160, -30)
draw_img = draw_window(img, window)
axes[0].imshow(draw_img)

template_window = Window(410., 140., 180., 160., 0)
draw_img = draw_window(template_img, template_window)
axes[1].imshow(draw_img)


img_tensor = get_image_tensor(img)
template_tensor = get_image_tensor(template_img)

roi_feature_extractor = RoIAlignFeatureExtractor(output_size=100, sampling_ratio=2)

feature_map = template_tensor.float()
roi_features = roi_feature_extractor.extract(feature_map, template_window, spatial_scale=1)

roi_features = get_numpy_image(roi_features.reshape(1, 3, 100, 100)).astype(np.int8)
print(roi_features.shape)
plt.imshow(roi_features)

model_name = 'tf_efficientnet_lite0'
feature_extractor = CNN(model_name)

roi_feature_extractor = RoIAlignFeatureExtractor(output_size=3, sampling_ratio=2)

matcher = CnnMatcher(feature_extractor, roi_feature_extractor)

matcher.set_template(template_img, template_window)
position = matcher.find(template_img)
