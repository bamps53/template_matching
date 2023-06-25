import numpy as np
from PIL import Image
from matcher import CnnMatcher, Position, Window


def test_match_template():
    image = np.asarray(Image.open('./data/image_1.png'))
    template = np.asarray(Image.open('./data/template_1.jpg'))
    window = Window(410, 140, 180, 160, 0)

    matcher = CnnMatcher(template, window)
    position = matcher.find(image)

    expected_position = Position(210, 160, -30)

    assert position.cx == expected_position.cx, f"Expected {expected_position.cx}, but got {position.cx}"
    assert position.cy == expected_position.cy, f"Expected {expected_position.cy}, but got {position.cy}"
    assert position.angle == expected_position.angle, f"Expected {expected_position.angle}, but got {position.angle}"
