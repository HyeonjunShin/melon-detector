import torch
from torch.utils.data import DataLoader
from segmentation_model.datasets import MelonDataset, custom_collate_fn
import numpy as np

def debug_dataloader():
    print("--- Dataloader Verification Start ---")
    try:
        # 데이터셋 인스턴스 생성 (검증을 위해 작은 숫자로 설정)
        dataset = MelonDataset(num_frames=100, num_seq=10)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)

        for batch_idx, (images, targets) in enumerate(loader):
            print(f"\n[Batch {batch_idx}]")
            print(f"Images Shape: {images.shape} (Expected: [B, T, C, H, W])")
            
            # 1. 이미지 값 범위 확인 (ToDtype에 의해 0~1 사이여야 함)
            print(f"Images Value Range: min={images.min():.4f}, max={images.max():.4f}")

            # 2. 시퀀스 내 프레임간 차이 확인 (Jitter가 적용되었으므로 프레임마다 달라야 함)
            # 주의: 원본 코드에서는 stack 후 한 번에 transform을 적용하므로 
            # 10개 프레임이 '동일한' 파라미터로 변환되었는지 확인
            diff = torch.abs(images[0, 0] - images[0, 1]).sum()
            if diff == 0:
                print("Sequence frames are identical (Global transform applied correctly).")
            else:
                print(f"Sequence frames differ (Diff sum: {diff:.4f}). Check if this is intended.")

            # 3. 타겟 데이터 구조 확인
            for b in range(len(targets)):
                for t in range(len(targets[b])):
                    target = targets[b][t]
                    labels = target['labels']
                    boxes = target['boxes']
                    masks = target['masks']
                    
                    print(f"  Batch {b}, Frame {t}:")
                    print(f"    Labels: {labels.shape}, {labels.tolist()}")
                    print(f"    Boxes: {boxes.shape} (Normalized: {boxes.max() <= 1.0})")
                    print(f"    Masks: {masks.shape}, Unique values: {torch.unique(masks).tolist()}")

                    if len(labels) > 0:
                        # 박스와 마스크 존재 여부 확인
                        if boxes.shape[0] != labels.shape[0] or masks.shape[0] != labels.shape[0]:
                            print(f"    [ERROR] Mismatch in counts: Labels({len(labels)}), Boxes({len(boxes)}), Masks({len(masks)})")
            
            if batch_idx >= 0: break # 첫 배치만 확인

    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_dataloader()
