from abc import abstractmethod

import numpy as np

class Watermark:
    @abstractmethod
    def encode(self, data: np.ndarray,  **kwargs):
        pass

    @abstractmethod
    def decode(self, data: np.ndarray,   **kwargs):
        pass