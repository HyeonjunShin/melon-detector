import argparse

import torch

# 앞서 작성한 클래스와 함수들을 import 한다고 가정합니다.
# 실제 환경에서는 이들을 모듈(예: dataset.py, model.py, loss.py)로 분리하는 것이 좋습니다.
# from segmentation_model.datasets import MelonDataset, custom_collate_fn
from datasets2 import MelonDataset, custom_collate_fn
from matcher import SimpleHungarianMatcher
from model import FastMelonSegmenter
from torch.utils.data import DataLoader
from train import train_model


def get_args():
    parser = argparse.ArgumentParser(description="Segmentation Training Script")

    # 1. 경로 설정 (Path arguments)
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="데이터셋 루트 경로"
    )
    parser.add_argument(
        "--train_img", type=str, default="train/images", help="학습 이미지 폴더명"
    )
    parser.add_argument(
        "--train_mask", type=str, default="train/masks", help="학습 마스크 폴더명"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./saved_models", help="모델 저장 경로"
    )

    # 2. 하이퍼파라미터 (Hyperparameters) - 나중에 W&B Sweep 대상
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--model_name", type=str, default="unet", help="unet, deeplabv3 등"
    )

    return parser.parse_args()


def main():
    # ==========================================
    # 1. 하이퍼파라미터 및 환경 설정
    # ==========================================
    # 경로 설정 (실제 멜론 데이터셋 경로로 변경해주세요)
    # dir_path = "/home/hyeonjun/Desktop/melon_dataset-v3"
    dir_path = "/home/hyeonjun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/_melon_dataset/train"
    check_point_path = "./_check_points/melon_segmenter_epoch_200.pth"
    
    # 학습 파라미터
    BATCH_SIZE = 16  # GPU 메모리(VRAM)에 맞춰 조절 (OOM 발생 시 4로 낮춤)
    NUM_EPOCHS = 100  # 전체 데이터셋 반복 횟수
    LEARNING_RATE = 1e-4  # AdamW의 기본 학습률

    # 입력 해상도 (2m 높이에서 촬영된 영역의 디테일을 살리기 위해 최소 640x640 권장)
    IMG_SIZE = (640, 640)

    # 디바이스 설정 (NVIDIA GPU가 있다면 cuda, Mac은 mps, 없으면 cpu)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"✅ Using device: {device}")

    # ==========================================
    # 2. 데이터셋 및 데이터 로더 초기화
    # ==========================================
    print("📦 Loading Dataset...")
    # train_dataset = MelonDataset(txtfile=dir_path, img_size=IMG_SIZE)

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,           # 학습 시에는 데이터를 섞어주는 것이 필수
    #     num_workers=4,          # 데이터 로딩 병렬 처리 (CPU 코어 수에 맞게 조절)
    #     collate_fn=custom_collate_fn, # 타겟 딕셔너리 리스트를 유지하기 위한 필수 함수
    #     drop_last=True          # 마지막 배치가 BATCH_SIZE보다 작을 경우 버림 (안정성 확보)
    # )
    train_dataset = MelonDataset(
        "/home/hyeonjun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/_melon_dataset/data",
        "/home/hyeonjun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/_melon_dataset/train.txt",
        source_img_size=(1080, 1920),
        target_img_size=(256, 448),
        transform=None,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    print(f"✅ Dataset loaded! Total batches per epoch: {len(train_loader)}")

    # ==========================================
    # 3. 딥러닝 모델 및 매처 초기화
    # ==========================================
    print("🧠 Initializing Model and Matcher...")
    # 클래스는 1개(멜론), 마스크 생성용 프로토타입은 32장 사용
    model = FastMelonSegmenter(num_classes=1, num_prototypes=32)
    model.load_state_dict(torch.load(check_point_path, map_location=device), strict=False)

    # 헝가리안 매처 가중치 (현업에서 자주 쓰이는 기본 비율)
    matcher = SimpleHungarianMatcher(weight_class=2.0, weight_bbox=5.0, weight_giou=2.0)
    print("✅ Model initialized!")

    # ==========================================
    # 4. 학습 시작! (Training Loop)
    # ===================torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 250.00 MiB. GPU 0 has a total capacity of 23.49 GiB of which 262.31 MiB is free. Process 3416666 has 18.21 GiB memory in use. Including non-PyTorch memory, this process has 3.26 GiB memory in use. Of the allocated memory 2.79 GiB is allocated by PyTorch, and 31.57 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)=======================
    print("🚀 Starting Training Process...")
    trained_model = train_model(
        model=model,
        dataloader=train_loader,
        matcher=matcher,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device,
    )

    # ==========================================
    # 5. 최종 가중치 저장
    # ==========================================
    save_path = "melon_segmenter_final.pth"
    torch.save(trained_model.state_dict(), save_path)
    print(f"🎉 Training Complete! Final weights saved to {save_path}")


if __name__ == "__main__":
    main()
