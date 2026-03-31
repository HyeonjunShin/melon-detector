import os
import numpy as np
import torch
import cv2
import json
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.ops import masks_to_boxes, box_convert
from torchvision.utils import draw_bounding_boxes

class MelonDataset(Dataset):
    def __init__(self, dir_path: str, img_size = (640,640), transform=None) -> None:
        def key_fn(x):
            return int(x.split("_")[1].split(".")[0])

        self.dir_path = dir_path
        self.all_files = os.listdir(self.dir_path)
        self.color_files = [file for file in self.all_files if "rgb" in file]
        self.mask_files = [file for file in self.all_files if "segmentation" in file]
        self.json_files = [file for file in self.all_files if "labels" in file]

        self.color_files = sorted(self.color_files, key=key_fn)
        self.mask_files = sorted(self.mask_files, key=key_fn)
        self.json_files = sorted(self.json_files, key=key_fn)

        self.color_files = [os.path.join(self.dir_path, file) for file in self.color_files]
        self.mask_files = [os.path.join(self.dir_path, file) for file in self.mask_files]
        self.json_files = [os.path.join(self.dir_path, file) for file in self.json_files]

        self.img_size = img_size
        self.transform = transform
        
    def __len__(self):
        return len(self.color_files)
    
    def __getitem__(self, index: int):
        img_path = self.color_files[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)
        image = F.to_tensor(image)
            
        mask_path = self.mask_files[index]
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        ids = np.unique(mask)
        ids = ids[ids!=0]
        num_objs = len(ids)

        if num_objs == 0:
            target = {
                'labels': torch.zeros((0,), dtype=torch.int64),
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'masks': torch.zeros((0, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            }
            return image, target
        masks = (mask == ids[:, None, None]).astype(np.float32)
        masks = torch.as_tensor(masks, dtype=torch.int64)

        boxes = masks_to_boxes(masks)
        h, w = self.img_size
        cx = ((boxes[:, 0] + boxes[:, 2]) / 2) / w
        cy = ((boxes[:, 1] + boxes[:, 3]) / 2) / h
        bw = (boxes[:, 2] - boxes[:, 0]) / w
        bh = (boxes[:, 3] - boxes[:, 1]) / h
        boxes_cxcywh = torch.stack([cx, cy, bw, bh], dim=1)

        labels = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {
            'labels': labels,
            'boxes': boxes_cxcywh,
            'masks': masks
        }

        return image, target
    
        # from torchvision.transforms.functional import to_pil_image, pil_to_tensor
        # ret = draw_bounding_boxes(img, boxes)
        # img_pil = F.to_pil_image(ret)
        # img_pil.show()  # 기본 이미지 뷰어로 열기
    
def custom_collate_fn(batch):
    images = []
    targets = []
    
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
        
    images = torch.stack(images, dim=0)
    
    return images, targets

if __name__ == "__main__":
    train_dataset = MelonDataset("/home/hyeonjun/Desktop/melon_dataset-v3")
    train_laoder = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)


    # img, target = train_dataset[0]
    for batch_imgs, batch_targets in train_laoder:
        print(f"배치 이미지 Shape: {batch_imgs.shape}") # [8, 3, 640, 640]
        print(f"첫 번째 이미지의 라벨 수: {len(batch_targets[0]['labels'])}")
        print(f"첫 번째 이미지의 마스크 Shape: {batch_targets[0]['masks'].shape}") # [N, 640, 640]
        break # 한 배치만 확인하고 종료



    
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
