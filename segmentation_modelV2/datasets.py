import os
import numpy as np
import torch
import cv2
import json
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.ops import masks_to_boxes, box_convert
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import matplotlib.pyplot as plt
import open3d as o3d

def compute_normals_from_depth(depth_map):
    """
    depth_map: m 단위의 float32 이미지
    """
    # 1. 가로, 세로 방향의 미분(Gradient) 계산
    zx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
    zy = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)

    # 2. 법선 벡터 계산 (Normal = [-zx, -zy, 1])
    normal = np.dstack((-zx, -zy, np.ones_like(depth_map)))
    
    # 3. 벡터 정규화 (크기를 1로)
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)

    # 4. 시각화 및 학습용 범위 조정 (-1~1 -> 0~1)
    normal_map = (normal + 1.0) / 2.0
    return normal_map.astype(np.float32)

def clipping_depth(depth, min_depth, max_depth):
    clipied_depth = depth.copy()
    invalid_mask = (depth < min_depth) | (depth > max_depth)
    clipied_depth[invalid_mask] = 0
    return clipied_depth

def add_noise(depth, max_depth, noise_std, missing_prob):
    noisy_depth = depth.copy()

    valid_mask = (noisy_depth > 0)
    noise = np.random.normal(0, noise_std * (depth + 0.1), size=depth.shape)
    noisy_depth[valid_mask] += noise[valid_mask]

    drop_mask = np.random.rand(*depth.shape) < missing_prob
    noisy_depth[drop_mask] = 0
    noisy_depth = np.clip(noisy_depth, 0, max_depth)

    return noisy_depth.astype(np.float32)

class MelonDataset(Dataset):
    def __init__(self, 
                 path_data, 
                 path_txt, 
                 img_size = (896, 512), 
                 min_depth = 0.2, 
                 max_depth = 2.2, 
                 noise_std = 0.005, 
                 missing_prob = 0.02, 
                 transform=None) -> None:
        
        self.path_data = path_data
        self.path_txt = path_txt
        self.img_size = img_size
        # Depth range parameters
        self.min_depth = min_depth
        self.max_depth = max_depth
        # Augmentation parameters
        self.noise_std = noise_std
        self.missing_prob = missing_prob
        self.transform = transform

        with open(path_txt, "r") as f:
            file_numbers = [line.strip() for line in f.readlines()]
        
        self.color_filenames = [os.path.join(path_data, f"rgb_{num}.png") for num in file_numbers]
        self.mask_filenames = [os.path.join(path_data, f"instance_segmentation_{num}.png") for num in file_numbers]
        self.depth_filenames = [os.path.join(path_data, f"depth_{num}.npy") for num in file_numbers]
        
    def preprocess_all(self, color, depth, mask):
        h, w = color.shape[:2]
        target_w, target_h = self.img_size

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2

        img_rs = cv2.resize(color, (new_w, new_h), interpolation=cv2.INTER_AREA)
        color_canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        color_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_rs

        depth_rs = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        depth_canvas = np.zeros((target_h, target_w), dtype=np.float32)
        depth_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = depth_rs

        mask_rs = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        mask_canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        mask_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = mask_rs

        return color_canvas, depth_canvas, mask_canvas

    def __len__(self):
        return len(self.color_filenames)
    
    def __getitem__(self, index: int):
        # 1. Load data(color, depth, mask)
        color_filenames = self.color_filenames[index]
        color = cv2.imread(color_filenames, cv2.IMREAD_COLOR)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        depth_filename = self.depth_filenames[index]
        depth = np.load(depth_filename)
        depth = clipping_depth(depth, self.min_depth, self.max_depth)
        depth = add_noise(depth, self.max_depth, self.noise_std, self.missing_prob)
        
        mask_filename = self.mask_filenames[index]
        mask = cv2.imread(mask_filename, cv2.IMREAD_UNCHANGED)

        color, depth, mask = self.preprocess_all(color, depth, mask)
        color = F.to_tensor(color)

        ids = np.unique(mask)
        ids = ids[ids!=0]
        num_objs = len(ids)

        masks = (mask == ids[:, None, None])
        masks = torch.as_tensor(masks, dtype=torch.float32)

        boxes = masks_to_boxes(masks)
        h, w = self.img_size
        cx = ((boxes[:, 0] + boxes[:, 2]) / 2) / w
        cy = ((boxes[:, 1] + boxes[:, 3]) / 2) / h
        bw = (boxes[:, 2] - boxes[:, 0]) / w
        bh = (boxes[:, 3] - boxes[:, 1]) / h
        boxes_cxcywh = torch.stack([cx, cy, bw, bh], dim=1)

        labels = torch.ones((num_objs,), dtype=torch.int64)
        depths = torch.as_tensor(depth, dtype=torch.float32)
        
        target = {
            'labels': labels,
            'boxes': boxes_cxcywh,
            'masks': masks,
            'depths': depths
        }

        return color, target
    
def custom_collate_fn(batch):
    colors = []
    targets = []
    
    for color, target in batch:
        colors.append(color)
        targets.append(target)
        
    colors = torch.stack(colors, dim=0)
    return colors, targets


def to_cvimg(tensor):
    if tensor.dim() == 4:
        tensor = tensor.squeeze()
    img = tensor.permute(1, 2, 0).detach().cpu().numpy() # numpy 변환 추가
    img = (img * 255).astype(np.uint8)
    return img

def get_bbox(bboxes, img_size):
    W, H = img_size
    new_boxes = torch.zeros_like(bboxes)
    new_boxes[:, 0] = (bboxes[:, 0] - bboxes[:, 2] / 2) * W  # xmin
    new_boxes[:, 1] = (bboxes[:, 1] - bboxes[:, 3] / 2) * H  # ymin
    new_boxes[:, 2] = (bboxes[:, 0] + bboxes[:, 2] / 2) * W  # xmax
    new_boxes[:, 3] = (bboxes[:, 1] + bboxes[:, 3] / 2) * H  # ymax
    new_boxes = new_boxes.to(torch.int16)
    return new_boxes

if __name__ == "__main__":
    from tqdm import tqdm

    train_dataset = MelonDataset("/home/hyeonjun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/_melon_dataset/data",
                                 "/home/hyeonjun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/_melon_dataset/train.txt")
    train_laoder = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

    for batch_imgs, batch_targets in tqdm(train_laoder):
        color = batch_imgs[0]
        target = batch_targets[0]
        
        label = target['labels']
        boxes = target['boxes']
        masks = target['masks']
        depths = target['depths']

        print(label)
        print(boxes)
        print(masks.shape)
        print(depths.shape)
