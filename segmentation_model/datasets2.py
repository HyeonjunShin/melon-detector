import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.ops import masks_to_boxes
from torchvision.transforms import v2


def resize_with_letterbox(color_img, depth_map, K, target_size=(448, 256)):
    """
    비율을 유지하며 리사이즈하고 부족한 부분을 채웁니다 (Letterbox).
    K 행렬도 Padding된 좌표계에 맞춰 업데이트합니다.
    """
    src_h, src_w = color_img.shape[:2]
    tar_w, tar_h = target_size

    # 1. 비율 유지 스케일 계산 (가장 큰 축 기준)
    scale = min(tar_w / src_w, tar_h / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)

    # 2. 이미지 리사이즈
    resized_color = cv2.resize(color_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_depth = cv2.resize(depth_map, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # 3. Padding 계산 (중앙 정렬)
    pad_w = (tar_w - new_w) // 2
    pad_h = (tar_h - new_h) // 2

    # 4. 최종 캔버스 생성 및 배치
    final_color = np.zeros((tar_h, tar_w, 3), dtype=np.uint8)
    final_depth = np.zeros((tar_h, tar_w), dtype=depth_map.dtype)

    final_color[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized_color
    final_depth[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized_depth

    # 5. K Matrix 업데이트 (Scale 적용 후 Padding만큼 이동)
    K_prime = K.copy().astype(np.float64)
    K_prime[0, 0] *= scale  # fx
    K_prime[1, 1] *= scale  # fy
    K_prime[0, 2] = K_prime[0, 2] * scale + pad_w  # cx
    K_prime[1, 2] = K_prime[1, 2] * scale + pad_h  # cy

    return final_color, final_depth, K_prime


class DepthToNormal(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(
            1, 1, 3, 3
        )
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(
            1, 1, 3, 3
        )
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    def forward(self, depth, fx, fy):
        dz_dx = F.conv2d(depth, self.kernel_x, padding=1)
        dz_dy = F.conv2d(depth, self.kernel_y, padding=1)

        # Normal_x = -dz/dx * fx, Normal_y = -dz/dy * fy, Normal_z = depth
        nx = -dz_dx * fx
        ny = -dz_dy * fy
        nz = depth

        normal = torch.cat([nx, ny, nz], dim=1)
        norm = torch.norm(normal, p=2, dim=1, keepdim=True)
        return normal / (norm + 1e-6)


class ClampDepth(v2.Transform):
    def __init__(self, min_depth=0.2, max_depth=2.2):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, input):
        out = input.clone()
        out[out < self.min_depth] = 0.0
        out[out > self.max_depth] = 0.0
        return out


class AddNoise(v2.Transform):
    def __init__(self, max_depth=2.2, noise_std=0.005, missing_prob=0.02):
        super().__init__()
        self.max_depth = max_depth
        self.noise_std = noise_std
        self.missing_prob = missing_prob

    def forward(self, input):
        out = input.clone()
        noise = torch.randn_like(out) * self.noise_std
        out += noise

        missing_mask = torch.rand_like(out) < self.missing_prob
        out[missing_mask] = 0.0

        out[out > self.max_depth] = 0.0
        return out


class MelonDataset(Dataset):
    def __init__(
        self,
        dir_path,
        txt_path,
        source_img_size=(1080, 1920),
        target_img_size=(256, 448),
        depth_minmax=(0.2, 2.2),
        transform=None,
    ) -> None:
        self.source_img_size = source_img_size
        self.target_img_size = target_img_size
        self.depth_minmax = depth_minmax
        self.transform = transform

        with open(txt_path) as f:
            file_numbers = [line.strip() for line in f.readlines()]

        self.color_filenames = [os.path.join(dir_path, f"rgb_{num}.png") for num in file_numbers]
        self.mask_filenames = [
            os.path.join(dir_path, f"instance_segmentation_{num}.png") for num in file_numbers
        ]
        self.depth_filenames = [os.path.join(dir_path, f"depth_{num}.npy") for num in file_numbers]

        self.color_transforms = v2.Compose(
            [
                v2.Resize(self.target_img_size, interpolation=v2.InterpolationMode.BILINEAR),
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=0.5,  # 0.5 ~ 1.5 사이에서 밝기 결정
                            contrast=0.5,  # 0.5 ~ 1.5 사이에서 대비 결정
                            saturation=0.3,  # 0.7 ~ 1.3 사이에서 채도 결정
                            hue=0.1,  # -0.1 ~ 0.1 사이에서 색상 결정
                        ),
                        v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
                    ],
                    p=0.8,
                ),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        self.depth_transforms = v2.Compose(
            [
                v2.ToImage(),
                ClampDepth(min_depth=0.2, max_depth=2.2),
                AddNoise(max_depth=2.2, noise_std=0.005, missing_prob=0.02),
                v2.Resize(self.target_img_size, interpolation=v2.InterpolationMode.NEAREST),
                v2.ToDtype(torch.float32, scale=False),
            ]
        )

        self.mask_transforms = v2.Compose(
            [
                v2.Resize(self.target_img_size, interpolation=v2.InterpolationMode.NEAREST),
                v2.ToDtype(torch.float32, scale=False),
            ]
        )

        self.color_aug = v2.Compose(
            [
                v2.ColorJitter(brightness=0.2, contrast=0.2),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.color_filenames)

    def __getitem__(self, index: int):
        color = read_image(self.color_filenames[index], ImageReadMode.RGB)
        color = self.color_transforms(color)

        depth = np.load(self.depth_filenames[index])
        depth = self.depth_transforms(depth)

        mask = read_image(self.mask_filenames[index], ImageReadMode.UNCHANGED)
        mask = self.mask_transforms(mask)

        ids = torch.unique(mask)
        ids = ids[ids != 0]
        num_objs = len(ids)

        masks = mask == ids[:, None, None]

        boxes = masks_to_boxes(masks)
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        h, w = self.target_img_size

        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        boxes_cxcywh = torch.stack([cx, cy, bw, bh], dim=1)
        boxes_cxcywh = torch.clamp(boxes_cxcywh, min=0.0, max=1.0)

        # xmin = (cx - bw / 2) * w
        # ymin = (cy - bh / 2) * h
        # xmax = (cx + bw / 2) * w
        # ymax = (cy + bh / 2) * h
        # boxes_xyxy = torch.stack([xmin, ymin, xmax, ymax], dim=1)

        labels = torch.zeros((num_objs,), dtype=torch.int64)
        # labels = torch.ones((num_objs,), dtype=torch.int64)

        target = {"labels": labels, "boxes": boxes_cxcywh, "masks": masks}

        return color, depth, target


def custom_collate_fn(batch):
    images = []
    depth_maps = []
    targets = []

    for img, depth_map, tgt in batch:
        images.append(img)
        depth_maps.append(depth_map)
        targets.append(tgt)

    images = torch.stack(images, dim=0)
    depth_maps = torch.stack(depth_maps, dim=0)

    return images, depth_maps, targets


if __name__ == "__main__":
    K = np.array([[911.2097, 0, 963.1977], [0, 911.2242, 549.0802], [0, 0, 1]])
    train_dataset = MelonDataset(
        "/home/hyeonjun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/_melon_dataset/data",
        "/home/hyeonjun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/_melon_dataset/train.txt",
        source_img_size=(1080, 1920),
        target_img_size=(256, 448),
        depth_minmax=(0.2, 2.2),
        transform=None,
    )
    train_laoder = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # img, target = train_dataset[0]
    for color, depth, batch_targets in train_laoder:
        print(f"배치 이미지 Shape: {color.shape}")  # [8, 3, 640, 640]
        print(f"배치 깊이 맵 Shape: {depth.shape}")  # [8, 1, 640, 640]
        print(f"첫 번째 이미지의 라벨 수: {len(batch_targets[0]['labels'])}")
        print(f"첫 번째 이미지의 마스크 Shape: {batch_targets[0]['masks'].shape}")  # [N, 640, 640]
        # break # 한 배치만 확인하고 종료
        print()
