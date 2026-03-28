
import numpy as np

class DualAGNCreator:
    def __init__(self):
        pass

    @staticmethod
    def convolve_images(central_image: np.ndarray, offset_image: np.ndarray):
        return central_image + offset_image
