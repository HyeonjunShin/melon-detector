import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.ops import masks_to_boxes


def get_color_path(x):
    return f"/home/hyeonjun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/_train_dataset/rgb_{x:04d}.png"  # noqa: E501


def get_mask_path(x):
    return f"/home/hyeonjun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/_train_dataset/instance_segmentation_{x:04d}.png"  # noqa: E501


def get_src_and_dest(filenames):
    src = set()
    for filename in filenames:
        segmentation = read_image(filename, mode=ImageReadMode.UNCHANGED).numpy()
        segmentation[segmentation == 1] = 0 # To remote the object for except the other objects.
        src.update(np.unique(segmentation).tolist())
    src = np.array(sorted(list(set(src))))
    dest = np.arange(len(src))
    return src, dest

class MelonDataset(Dataset):
    def __init__(
        self, num_frames=10000, num_seq=10, target_image_size=(256, 448), transform=None
    ) -> None:
        self.num_frames = num_frames
        self.num_seq = num_seq
        self.num_data = num_frames // num_seq
        self.h, self.w = target_image_size
        self.transform = transform

    def __len__(self):
        return self.num_data

    def __getitem__(self, index: int):
        start_idx = index * self.num_seq
        end_idx = (index + 1) * self.num_seq

        color_files = [get_color_path(i) for i in range(start_idx, end_idx)]
        mask_files = [get_mask_path(i) for i in range(start_idx, end_idx)]
        src, dest = get_src_and_dest(mask_files)

        colors = []
        targets = []
        for color_file, mask_file in zip(color_files, mask_files):
            color = read_image(color_file, mode=ImageReadMode.RGB)
            mask = read_image(mask_file, mode=ImageReadMode.UNCHANGED)
            new_mask = np.zeros_like(mask)
            for s, d in zip(src, dest):
                new_mask[mask == s] = d

            ids = np.unique(new_mask)
            ids = ids[ids != 0]
            num_objs = len(ids)

            if num_objs == 0:
                target = {
                    "labels": torch.zeros((0,), dtype=torch.int64),
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "masks": torch.zeros(
                        (0, self.h, self.w),
                        dtype=torch.float32,
                    ),
                }
                colors.append(color)
                targets.append(target)
                continue

            new_masks = (new_mask == ids[:, None, None]).astype(np.float32)
            new_masks = torch.as_tensor(new_masks, dtype=torch.int64)

            boxes = masks_to_boxes(new_masks)
            cx = ((boxes[:, 0] + boxes[:, 2]) / 2) / self.w
            cy = ((boxes[:, 1] + boxes[:, 3]) / 2) / self.h
            bw = (boxes[:, 2] - boxes[:, 0]) / self.w
            bh = (boxes[:, 3] - boxes[:, 1]) / self.h
            boxes_cxcywh = torch.stack([cx, cy, bw, bh], dim=1)

            labels = torch.zeros((num_objs,), dtype=torch.int64)
            target = {"labels": labels, "boxes": boxes_cxcywh, "masks": new_masks}
            colors.append(color)
            targets.append(target)
        colors = torch.stack(colors, dim=0)
        return colors, targets


def custom_collate_fn(batch):
    images = []
    targets = []

    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)

    images = torch.stack(images, dim=0)

    return images, targets


if __name__ == "__main__":
    train_dataset = MelonDataset()
    train_laoder = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn
    )

    # img, target = train_dataset[0]
    for batch_imgs, batch_targets in train_laoder:
        print()
        # print(f"배치 이미지 Shape: {batch_imgs.shape}")  # [8, 3, 640, 640]
        # print(f"첫 번째 이미지의 라벨 수: {len(batch_targets[0]['labels'])}")
        # print(f"첫 번째 이미지의 마스크 Shape: {batch_targets[0]['masks'].shape}")  # [N, 640, 640]
        break  # 한 배치만 확인하고 종료

    # label = json.load(open(json_files[0]))

    # cv2.imshow("img", img)

    # for instance_mask in instance_masks:
    # target_idx = mask == melons[i]
    # target_idx = target_idx.astype(np.uint8) * 255
    # cv2.imshow("mask", instance_mask)
    # cv2.waitKey(0)

    # print(target_idx!=0)
    # print(np.sum(target_idx==0))
    # print(mask[target_idx].shape)
    # print(melons)
    # print(img.shape)
    # print(mask.shape)

    # cv2.imshow("img", img)

    # target_idx = cv2.cvtColor(target_idx, cv2.COLOR_GRAY2BGR)
