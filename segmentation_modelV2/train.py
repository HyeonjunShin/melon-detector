import torch
import torch.optim as optim
from tqdm import tqdm
from model import RealTimeMelonSegmenter
from matcher import SimpleHungarianMatcher
from criterion import MelonCriterion
from datasets import MelonDataset, custom_collate_fn, to_cvimg
from torch.utils.data import DataLoader

def train_one_epoch(model, criterion, dataloader, optimizer, device):
    model.train()
    criterion.train()
    
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training")

    for images, targets in progress_bar:
        # 1. 데이터 GPU 이동
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 2. 순전파 (Forward)
        # 모델 출력: pred_logits, pred_boxes, pred_coeffs, prototypes
        outputs = model(images) 
        
        # 3. 손실 계산 (Criterion)
        # outputs를 Criterion이 기대하는 튜플/리스트 형태로 전달
        loss = criterion(outputs, targets)

        # 4. 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping (Transformer 계열은 필수 권장)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # 로깅
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    return epoch_loss / len(dataloader)

@torch.no_grad()
def validate_model(model, criterion, dataloader, device):
    model.eval()
    criterion.eval()
    val_loss = 0.0
    
    for images, targets in dataloader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)
        loss = criterion(outputs, targets)
        val_loss += loss.item()

    return val_loss / len(dataloader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 모델 초기화
    # 로봇 제어를 위해 100개의 쿼리를 사용하는 모델
    model = RealTimeMelonSegmenter(num_classes=1, num_queries=100).to(device)
    
    # 2. 매처 및 손실 함수 설정
    matcher = SimpleHungarianMatcher(weight_class=2.0, weight_bbox=5.0, weight_giou=2.0)
    
    # 가중치 딕셔너리 (필요에 따라 조절)
    weight_dict = {
        'loss_cls': 2.0,    # 클래스 분류
        'loss_bbox': 5.0,   # L1 박스 거리
        'loss_giou': 2.0,   # GIoU (박스 정교함)
        'loss_mask': 10.0   # 마스크 (YOLACT 조립 손실) - 가중치를 높게 잡는 것이 유리함
    }
    
    criterion = MelonCriterion(matcher, num_classes=1, weight_dict=weight_dict).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    train_dataset = MelonDataset("/home/hyeonjun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/_melon_dataset/data",
                                 "/home/hyeonjun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/_melon_dataset/train.txt")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True )

    val_dataset = MelonDataset("/home/hyeonjun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/_melon_dataset/data",
                                 "/home/hyeonjun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/_melon_dataset/val.txt")
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True )

    best_val_loss = float('inf')
    num_epochs = 50
    for epoch in range(num_epochs):
        avg_train_loss = train_one_epoch(model, criterion, train_loader, optimizer, device)

        avg_val_loss = validate_model(model, criterion, val_loader, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 스케줄러 업데이트
        scheduler.step(avg_val_loss)

        # Best Model 저장 (Validation Loss 기준)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_melon_model.pth")
            print("  ⭐ Best model saved!")

if __name__ == "__main__":
    main()