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


class AddNoise(v2.Transform):
    def __init__(self, max_depth=2.2, noise_std=0.005, missing_prob=0.02):
        super().__init__()
        self.max_depth = max_depth
        self.noise_std = noise_std
        self.missing_prob = missing_prob

    def forward(self, input):
        out = input.clone()
        noise = torch.randn_like(out) * self.noise_std
        noise[noise < 0] = 0.0
        out += noise
        out = torch.clamp(out, 0, self.max_depth)

        missing_mask = torch.rand_like(out) < self.missing_prob
        out[missing_mask] = 0.0
        return out


class Normalize_Normal(v2.Transform):
    def forward(self, input):
        normal_vec = (input / 127.5) - 1.0

        norm = torch.norm(normal_vec, p=2, dim=0, keepdim=True)
        normal_unit = normal_vec / (norm + 1e-6)

        return normal_unit


class Normalize_Depth(v2.Transform):
    def __init__(self, min_depth, max_depth):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, input):
        out = torch.zeros_like(input)
        valid_mask = input > 0.0  # The valid data is greater than 0.0
        invalid_mask = ~valid_mask

        normalized_depth = (input[valid_mask] - self.min_depth) / (self.max_depth - self.min_depth)
        out[valid_mask] = torch.clamp(normalized_depth, 0.0, 1.0) * 0.5

        out[invalid_mask] = 1.0  # The out of data is set to 1.0
        return out


def filtering(depth, normal, mask):
    invalid_mask = (depth < 0.2) | (depth > 2.2)

    out_depth = depth.clone().masked_fill_(invalid_mask, 0.0)
    out_mask = mask.clone().masked_fill_(invalid_mask, 0)
    out_normal = normal.clone()[:3, :, :].masked_fill_(invalid_mask, 0.0)

    depth_list = []
    unique_ids = torch.unique(out_mask)
    for obj_id in unique_ids:
        if obj_id == 0:
            continue
        m_depth = torch.mean(out_depth[out_mask == obj_id])
        depth_list.append((obj_id.item(), m_depth))

    depth_list.sort(key=lambda x: x[1])

    sorted_mask = torch.zeros_like(out_mask)
    for new_id, (old_id, _) in enumerate(depth_list, start=1):
        sorted_mask[out_mask == old_id] = new_id

    return out_depth, out_normal, sorted_mask


class MelonDataset(Dataset):
    def __init__(
        self,
        dir_path,
        source_img_size=(1080, 1920),
        target_img_size=(256, 448),
        depth_minmax=(0.2, 2.2),
        transform=None,
    ) -> None:
        self.dir_path = dir_path
        self.source_img_size = source_img_size
        self.target_img_size = target_img_size
        self.depth_minmax = depth_minmax
        self.transform = transform

        self.sequence_length = 15
        self.total_frames = 9900
        self.num_sequences = self.total_frames // self.sequence_length

        self.preprocess_color = v2.Compose(
            [
                v2.Resize(self.target_img_size, interpolation=v2.InterpolationMode.BILINEAR),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        self.postprocess_color = v2.Compose(
            [
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
            ]
        )

        self.preprocess_depth = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(self.target_img_size, interpolation=v2.InterpolationMode.NEAREST),
                v2.ToDtype(torch.float32, scale=False),
            ]
        )
        self.postprocess_depth = v2.Compose(
            [
                AddNoise(max_depth=2.2, noise_std=0.005, missing_prob=0.02),
                Normalize_Depth(min_depth=0.2, max_depth=2.2),
            ]
        )

        self.preprocess_normal = v2.Compose(
            [
                v2.Resize(self.target_img_size, interpolation=v2.InterpolationMode.NEAREST),
                v2.ToDtype(torch.float32, scale=False),
            ]
        )
        self.postprocess_normal = v2.Compose(
            [
                Normalize_Normal(),
            ]
        )

        self.preprocess_mask = v2.Compose(
            [
                v2.Resize(self.target_img_size, interpolation=v2.InterpolationMode.NEAREST),
                v2.ToDtype(torch.float32, scale=False),
            ]
        )

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index: int):
        start_frame_idx = index * self.sequence_length
        seq_color, seq_depth, seq_mask, seq_normal, seq_bbox, seq_labels = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for i in range(start_frame_idx, start_frame_idx + self.sequence_length):
            frame_name = f"{i:05d}"

            color = read_image(
                os.path.join(self.dir_path, "rgb", f"rgb_{frame_name}.png"),
                ImageReadMode.RGB,
            )
            color = self.preprocess_color(color)

            depth = np.load(
                os.path.join(
                    self.dir_path,
                    "distance_to_camera",
                    f"distance_to_camera_{frame_name}.npy",
                )
            )
            depth = self.preprocess_depth(depth)

            normal = read_image(
                os.path.join(self.dir_path, "normals", f"normals_{frame_name}.png"),
                ImageReadMode.UNCHANGED,
            )
            normal = self.preprocess_normal(normal)

            mask = read_image(
                os.path.join(
                    self.dir_path,
                    "instance_segmentation",
                    f"instance_segmentation_{frame_name}.png",
                ),
                ImageReadMode.UNCHANGED,
            )
            mask = self.preprocess_mask(mask)

            depth, normal, mask = filtering(depth, normal, mask)
            # depth를 기준으로 필요 없는 부분 0으로 마스킹

            ids = torch.unique(mask)
            ids = ids[ids != 0]
            num_objs = len(ids)

            if num_objs > 0:
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
            else:
                masks = torch.zeros((0, *self.target_img_size), dtype=torch.bool)
                boxes_cxcywh = torch.zeros((0, 4), dtype=torch.float32)

            labels = torch.zeros((num_objs,), dtype=torch.int64)
            depth = self.postprocess_depth(depth)
            normal = self.postprocess_normal(normal)

            seq_color.append(color)  # (C, H, W), Normalize + Augmentation
            seq_depth.append(depth)  # (1, H, W)
            seq_normal.append(normal)  # (3, H, W) Normalized

            seq_mask.append(masks)  # (#Instances, H, W) Ordered by Depth
            seq_bbox.append(boxes_cxcywh)  # (#Instances, 4) [cx, cy, w, h] normalized
            seq_labels.append(labels)  # (#Instances,) Class 0 (Melon) Only

        return (
            torch.stack(seq_color, dim=0),
            torch.stack(seq_depth, dim=0),
            torch.stack(seq_normal, dim=0),
            {"masks": seq_mask, "bboxes": seq_bbox, "labels": seq_labels},
        )

        # color = torch.stack(seq_color, dim=0)
        # depth = torch.stack(seq_depth, dim=0)
        # masks = torch.stack(seq_mask, dim=0)
        # normal = torch.stack(seq_normal, dim=0)
        # bbox = torch.stack(seq_bbox, dim=0)

        # xmin = (cx - bw / 2) * w
        # ymin = (cy - bh / 2) * h
        # xmax = (cx + bw / 2) * w
        # ymax = (cy + bh / 2) * h
        # boxes_xyxy = torch.stack([xmin, ymin, xmax, ymax], dim=1)

        # labels = torch.ones((num_objs,), dtype=torch.int64)

        # target = {
        #     "labels": labels,
        #     "boxes": boxes_cxcywh,
        #     "masks": masks,
        #     "normals": normal,
        # }

        # return color, depth, target


def custom_collate_fn(batch):
    color_batch = []
    depth_batch = []
    normal_batch = []
    targets_batch = []

    for colors, depths, normals, targets in batch:
        color_batch.append(colors)  # [15, 3, H, W]
        depth_batch.append(depths)  # [15, 1, H, W]
        normal_batch.append(normals)  # [15, 3, H, W]
        targets_batch.append(targets)  # 딕셔너리 {"masks": [15, N, H, W], ...}

    color_batch = torch.stack(color_batch, dim=0)  # [Batch, 15, 3, H, W]
    depth_batch = torch.stack(depth_batch, dim=0)  # [Batch, 15, 1, H, W]
    normal_batch = torch.stack(normal_batch, dim=0)  # [Batch, 15, 3, H, W]

    return color_batch, depth_batch, normal_batch, targets_batch


if __name__ == "__main__":
    K = np.array([[911.2097, 0, 963.1977], [0, 911.2242, 549.0802], [0, 0, 1]])

    train_dataset = MelonDataset(
        "/home/hyeonjun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/_melon_seq_dataset",
        source_img_size=(1080, 1920),
        target_img_size=(256, 448),
        depth_minmax=(0.2, 2.2),
        transform=None,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # img, target = train_dataset[0]
    for colors, depths, normals, targets in train_loader:
        print(f"배치 이미지 Shape: {colors.shape}")
        print(f"배치 깊이 맵 Shape: {depths.shape}")
        print(f"첫 번째 이미지의 라벨 수: {len(targets[0]['labels'])}")
        print(f"첫 번째 이미지의 마스크 Shape: {targets[0]['masks'][0].shape}")
        # break # 한 배치만 확인하고 종료
        print()
