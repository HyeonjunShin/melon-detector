from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    WIDTH = 1920
    HEIGHT = 1080
    CHANNELS = 3
    TARGET_WIDHT = 448
    TARGET_HEIGHT = 256

    # 카메라 내부 매트릭스
    INTRINSIC = np.array(
        [[907.18255615, 0.0, 962.34729004], [0.0, 906.9387207, 548.12927246], [0.0, 0.0, 1.0]]
    )

    # 왜곡 계수 (D)
    DISTORTION = np.array(
        [
            2.63843089e-01,
            -2.51909852e00,
            -3.29987561e-05,
            -1.81387455e-04,
            1.62435722e00,
            1.49601102e-01,
            -2.34095073e00,
            1.54302502e00,
        ]
    )
