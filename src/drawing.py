import cv2
import numpy as np
from typing import Union
import torch
import matplotlib.pyplot as plt

from shape.window import Window

def draw_rotated_rect(image: np.ndarray, input_array: Union[np.ndarray, torch.Tensor]):
    """
    This function draws a rotated rectangle on a blank image.

    :param input_array: 1x5 numpy array or torch tensor containing the rectangle's parameters
                        in the order: center_x, center_y, width, height, angle
    """
    
    # If the input is a torch tensor, convert it to a numpy array
    if isinstance(input_array, torch.Tensor):
        input_array = input_array.numpy()

    # Reshape the input_array for convenience
    input_array = input_array.reshape(-1).astype(np.float32)

    # Extract the rectangle parameters
    center_x, center_y, width, height, angle = input_array

    # Define the rotated rectangle parameters
    rotated_rect = ((center_x, center_y), (width, height), angle)

    # Compute the four vertices of the rectangle
    box = cv2.boxPoints(rotated_rect)
    box = np.int0(box)  # Convert the coordinates to integers

    # Draw the rectangle
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

    return image

def draw_window(image: np.ndarray, window: Window) -> np.ndarray:
    """
    This function draws a rotated rectangle on a image.
    """
    # Draw the rectangle
    cv2.drawContours(image, [window.as_boxPoints()], 0, (0, 255, 0), 2)

    return image

if __name__ == '__main__':
    # Create an empty image
    image = np.zeros((500, 500, 3), dtype='uint8')

    # Test the function with a numpy array
    image = draw_rotated_rect(image, np.array([[250, 250, 200, 100, 45]]))
    plt.imshow(image)