import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthToNormal(nn.Module):
    def __init__(self, focal_length=500.0):
        super(DepthToNormal, self).__init__()
        self.focal_length = focal_length
        
        # Sobel 커널 정의 (x, y 방향의 변화량 계산)
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('weight_x', kernel_x)
        self.register_buffer('weight_y', kernel_y)

    def forward(self, depth):
        """
        Input: depth (B, 1, H, W) - 단위는 미터(m) 권장
        Output: normal (B, 3, H, W) - L2 Normalized된 벡터
        """
        # 1. Depth Gradient 계산 (Padding을 주어 크기 유지)
        dx = F.conv2d(depth, self.weight_x, padding=1)
        dy = F.conv2d(depth, self.weight_y, padding=1)
        
        # 2. Normal Vector의 각 성분 계산
        # dz는 미세한 변화를 감지하기 위한 스케일링 인자 (focal length 기반)
        # 실제 카메라 파라미터가 있다면 더 정확한 계산이 가능합니다.
        dz = torch.ones_like(dx) * (1.0 / self.focal_length)
        
        # 벡터 결합 (nx, ny, nz)
        # 일반적으로 깊이가 깊어질수록 z값이 커지므로 방향을 맞춰줍니다.
        normal = torch.cat((-dx, -dy, dz), dim=1)
        
        # 3. L2 Normalization (단위 벡터화)
        norm = torch.norm(normal, p=2, dim=1, keepdim=True)
        normal = normal / (norm + 1e-6) # 0으로 나누기 방지
        
        return normal

# 사용 예시

import numpy as np
import matplotlib.pyplot as plt
from datasets import clipping_depth, add_noise, compute_normals_from_depth
depth = np.load("/home/hyeonjun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/_melon_dataset/data/depth_16610.npy")
depth_clippied = clipping_depth(depth, 0.2, 2.2)
depth_noised = add_noise(depth, 2.2, 0.005, 0.02)

# depth_batch = torch.randn(1, 1, 480, 640).cuda()
depth_batch = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).cuda()

normal_extractor = DepthToNormal().cuda()
normal_map = normal_extractor(depth_batch)

plt.imshow(normal_map.squeeze().permute(1,2,0).cpu())
plt.show()