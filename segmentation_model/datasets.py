import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.ops import masks_to_boxes
from torchvision.transforms import v2



def get_src_and_dest(masks):
    src = set()
    for mask in masks:
        mask[mask == 1] = 0  # To remote the object for except the other objects.
        src.update(np.unique(mask).tolist())
    src = np.array(sorted(list(set(src))))
    dest = np.arange(len(src))
    return src, dest


class MelonDataset(Dataset):
    def __init__(
        self,
        is_train=True,
        num_frames=10000,
        num_seq=10,
        target_image_size=(256, 448),
        transform=None,
    ) -> None:
        self.is_train = is_train
        if self.is_train:
            self.dir_path = (
                "/home/hyeonjun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/_train_dataset"
            )
        else:
            self.dir_path = (
                "/home/hyeonjun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/_test_dataset"
            )

        self.num_frames = num_frames
        self.num_seq = num_seq
        self.num_data = num_frames // num_seq
        self.h, self.w = target_image_size
        self.transform = transform
        self.target_image_size = target_image_size

        self.color_transforms = v2.Compose(
            [
                v2.Resize(self.target_image_size, interpolation=v2.InterpolationMode.BILINEAR),
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=0.5,  # 0.5 ~ 1.5 사이에서 밝기 결정
                            contrast=0.5,  # 0.5 ~ 1.5 사이에서 대비 결정
                            saturation=0.3,  # 0.7 ~ 1.3 사이에서 채도 결정
                            # hue=0.1,  # -0.1 ~ 0.1 사이에서 색상 결정
                        ),
                        v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
                    ],
                    p=0.8,
                ),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        self.mask_transforms = v2.Compose(
            [
                v2.Resize(self.target_image_size, interpolation=v2.InterpolationMode.NEAREST),
                v2.ToDtype(torch.float32, scale=False),
            ]
        )

    def get_color(self, idx):
        if self.is_train:
            return f"{self.dir_path}/rgb_{idx:05d}.png"
        else:
            return f"{self.dir_path}/rgb_{idx:04d}.png"

    def get_mask(self, idx):
        if self.is_train:
            return f"{self.dir_path}/instance_segmentation_{idx:05d}.png"
        else:
            return f"{self.dir_path}/instance_segmentation_{idx:04d}.png"

    def __len__(self):
        return self.num_data

    def __getitem__(self, index: int):
        start_idx = index * self.num_seq
        end_idx = (index + 1) * self.num_seq

        color_files = [self.get_color(i) for i in range(start_idx, end_idx)]
        mask_files = [self.get_mask(i) for i in range(start_idx, end_idx)]

        colors = torch.stack(
            [read_image(color_file, mode=ImageReadMode.RGB) for color_file in color_files], dim=0
        )
        colors = self.color_transforms(colors)
        masks = torch.stack(
            [read_image(mask_file, mode=ImageReadMode.UNCHANGED) for mask_file in mask_files], dim=0
        )
        masks = self.mask_transforms(masks)
        src, dest = get_src_and_dest(masks)

        # fig, axes = plt.subplots(1, 10, figsize=(20, 4))
        # fig.suptitle("Sequence of 10 Frames with Transforms Applied", fontsize=16)
        # vis_colors = colors.permute(0, 2, 3, 1).cpu().numpy()
        # for i in range(10):
        #     axes[i].imshow(vis_colors[i])
        #     axes[i].set_title(f"Frame {i}")
        #     axes[i].axis("off")
        # plt.tight_layout()
        # plt.show()

        targets = []
        for mask in masks:
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
                targets.append(target)
                continue

            new_masks = (new_mask == ids[:, None, None]).astype(np.float32)
            new_masks = torch.as_tensor(new_masks, dtype=torch.int64)

            boxes = masks_to_boxes(new_masks)
            img_whwh = torch.tensor([self.w, self.h, self.w, self.h], device=boxes.device)
            boxes_norm = boxes / img_whwh
            cx = (boxes_norm[:, 0] + boxes_norm[:, 2]) / 2
            cy = (boxes_norm[:, 1] + boxes_norm[:, 3]) / 2
            bw = boxes_norm[:, 2] - boxes_norm[:, 0]
            bh = boxes_norm[:, 3] - boxes_norm[:, 1]
            boxes_cxcywh = torch.stack([cx, cy, bw, bh], dim=1)

            labels = torch.zeros((num_objs,), dtype=torch.int64)
            target = {"labels": labels, "boxes": boxes_cxcywh, "masks": new_masks}
            targets.append(target)

        return colors, targets


def melon_collate_fn(batch):
    images = []
    targets = []

    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)

    images = torch.stack(images, dim=0)

    return images, targets


if __name__ == "__main__":
    train_dataset = MelonDataset(
        is_train=True,
        num_frames=10000,
    )
    train_laoder = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=melon_collate_fn
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
