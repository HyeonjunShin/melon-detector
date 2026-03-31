import os
import torch
from torchvision.ops import masks_to_boxes
import cv2
import numpy as np


def sort_key(x):
    return int(x[-9:].split(".png")[0])

def main():
    from tqdm import tqdm

    PATH = "/home/hyeonjun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/_melon_dataset/train"
    color_files = [file for file in os.listdir(PATH) if "rgb" in file]
    img_size = (640,640)
    for color_file in tqdm(color_files):
        color_path = os.path.join(PATH, color_file)
        seg_path = color_path.replace("rgb", "instance_segmentation")
        mask = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        # mask = torch.as_tensor(mask, dtype=torch.int64)
    
        # ids = np.unique(mask)
        # ids = ids[ids != 0]
        # # print(ids)
        # if len(ids) == 0:
        #     print(color_file, seg_path)

        mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)

        ids = np.unique(mask)
        ids = ids[ids!=0]
        num_objs = len(ids)

        if num_objs == 0:
            print(color_path, seg_path)
            # raise RuntimeError(f"{img_path}, {mask_path}")

    # dirs = sorted(os.listdir(PATH), key=sort_key)
    # print(dirs)

    # comm_ratio = 1000
    # for i in range(len(dirs)):
        # dir_name = dirs[i]
        # colors = [os.path.join(PATH, dir_name, 'rgb', f) for f in sorted(os.listdir(os.path.join(PATH, dir_name, 'rgb')), key=sort_key)]
        # print(colors)
        # segmentations = [os.path.join(PATH, dir_name, 'instance_segmentation', f) for f in sorted(os.listdir(os.path.join(PATH, dir_name, 'instance_segmentation')), key=sort_key)]
        
        # for segmentation_filename in segmentations:
        #     # print(segmentation_filename)
        #     mask = cv2.imread(segmentation_filename, cv2.IMREAD_UNCHANGED)
        #     mask = torch.as_tensor(mask, dtype=torch.int64)
            
        #     ids = np.unique(mask)
        #     ids = ids[ids != 0]
        #     if len(ids) == 0:
        #         print()
                
            # bboxs = masks_to_boxes(mask)
            # print(np.unique(mask))
        # print(colors)
        # print(segmentations)

        # masks_to_boxes(segmentations)

if __name__ == "__main__":
    main()