from dataclasses import dataclass
import numpy as np

@dataclass
class Frame:
    image: np.uint8
    depth : np.uint16
    timestamp_image: int