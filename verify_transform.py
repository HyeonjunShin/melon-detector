import torch
from torchvision.transforms import v2

def test_transform_consistency():
    # 1. 동일한 10개의 프레임 생성 (3, 256, 448)
    single_frame = torch.randint(0, 256, (3, 256, 448), dtype=torch.uint8)
    colors = torch.stack([single_frame.clone() for _ in range(10)], dim=0) # [10, 3, 256, 448]

    # 2. datasets.py와 동일한 변환 정의
    color_transforms = v2.Compose([
        v2.Resize((256, 448), interpolation=v2.InterpolationMode.BILINEAR),
        v2.RandomApply([
            v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.1),
            v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
        ], p=1.0), # 무조건 적용되도록 p=1.0 설정
        v2.ToDtype(torch.float32, scale=True),
    ])

    # 3. 변환 적용
    transformed_colors = color_transforms(colors)

    # 4. 프레임 간 차이 계산
    is_consistent = True
    for i in range(1, 10):
        diff = torch.abs(transformed_colors[0] - transformed_colors[i]).sum()
        if diff > 0:
            is_consistent = False
            print(f"Frame 0 and Frame {i} are different! Diff: {diff.item()}")
            break
    
    if is_consistent:
        print("Success: All 10 frames have the EXACT SAME transformation applied.")
    else:
        print("Failure: Transformations are different across frames.")

if __name__ == "__main__":
    test_transform_consistency()
