import os
import torch
import torch.optim as optim
from tqdm import tqdm
from model import FastMelonSegmenter
from matcher import SimpleHungarianMatcher
from datasets import MelonDataset

import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss, generalized_box_iou, box_convert

def compute_batch_loss(pred_logits, pred_boxes, pred_coeffs, pred_prototypes, targets, matcher):
    B = pred_logits.shape[0] # 배치 사이즈 (예: 8)
    device = pred_logits.device

    indices = []
    
    # ---------------------------------------------------------
    # 1. 헝가리안 매칭 (배치 내 이미지별로 순회)
    # ---------------------------------------------------------
    for i in range(B):
        p_logits = pred_logits[i] # [8400, Num_Classes]
        p_boxes = pred_boxes[i]   # [8400, 4]
        t_classes = targets[i]['labels'] # [N]
        t_boxes = targets[i]['boxes']    # [N, 4]
        
        # 이미지에 정답 객체가 하나도 없는 경우 예외 처리
        if len(t_classes) == 0:
            indices.append((torch.empty(0, dtype=torch.int64, device=device), 
                            torch.empty(0, dtype=torch.int64, device=device)))
            continue

        # 앞서 구현한 매처 실행
        matched_p_idx, matched_t_idx = matcher.match(p_logits, p_boxes, t_classes, t_boxes)
        # 매칭 결과를 GPU로 다시 올려서 리스트에 저장
        indices.append((matched_p_idx.to(device), matched_t_idx.to(device)))

    # ---------------------------------------------------------
    # 2. 배치 차원 인덱스 1차원으로 펴기 (Flattening)
    # ---------------------------------------------------------
    # 각 매칭된 결과가 어느 이미지(batch index)에서 왔는지 기록
    batch_idx = torch.cat([torch.full_like(p, i) for i, (p, _) in enumerate(indices)])
    pred_idx = torch.cat([p for (p, _) in indices])
    target_idx = torch.cat([t for (_, t) in indices])

    # ---------------------------------------------------------
    # 3. Classification Loss (Focal Loss) - 전체 8400개 대상
    # ---------------------------------------------------------
    target_classes = torch.zeros_like(pred_logits) # 모두 배경(0)으로 초기화
    
    if len(target_idx) > 0:
        # 매칭된 인덱스에만 실제 클래스 할당
        target_classes_concat = torch.cat([t['labels'][tgt] for t, (_, tgt) in zip(targets, indices)])
        target_classes[batch_idx, pred_idx, target_classes_concat] = 1.0
    
    loss_class = sigmoid_focal_loss(pred_logits, target_classes, alpha=0.25, gamma=2.0, reduction='mean')

    # 매칭된 객체가 아예 없다면(배경만 있는 이미지들) 박스/마스크 Loss는 0으로 처리
    if len(target_idx) == 0:
        return loss_class * 2.0 

    # ---------------------------------------------------------
    # 4. BBox Loss (L1 + GIoU) - 매칭된 객체만 대상
    # ---------------------------------------------------------
    matched_pred_boxes = pred_boxes[batch_idx, pred_idx]
    matched_target_boxes = torch.cat([t['boxes'][tgt] for t, (_, tgt) in zip(targets, indices)])

    loss_bbox_l1 = F.l1_loss(matched_pred_boxes, matched_target_boxes, reduction='mean')
    
    pred_xyxy = box_convert(matched_pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    target_xyxy = box_convert(matched_target_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    loss_giou = 1 - torch.diag(generalized_box_iou(pred_xyxy, target_xyxy)).mean()

    # ---------------------------------------------------------
    # 5. Mask Loss (Dice + BCE) - 매칭된 객체만 대상
    # ---------------------------------------------------------
    matched_coeffs = pred_coeffs[batch_idx, pred_idx] # [총 매칭 객체 수, 32]
    
    # 💡 주의: 각 객체가 자신이 속한 이미지의 prototype과 곱해지도록 batch_idx로 추출
    selected_prototypes = pred_prototypes[batch_idx]  # [총 매칭 객체 수, 32, H, W]
    
    # 마스크 조립 (einsum)
    # n: 매칭된 총 객체 수, c: 32 (프로토타입 수), h, w: 해상도
    pred_masks_logits = torch.einsum('nc,nchw->nhw', matched_coeffs, selected_prototypes)
    pred_masks = pred_masks_logits.sigmoid()

    # 정답 마스크 가져오기
    matched_target_masks = torch.cat([t['masks'][tgt] for t, (_, tgt) in zip(targets, indices)])
    
    # 🔥 에러 해결: 정수형(Long/Bool) 텐서를 실수형(Float)으로 변환!
    matched_target_masks = matched_target_masks.float()
    
    # 💡 주의: 프로토타입은 원본의 1/4 해상도(160x160)이고...
    h_mask, w_mask = pred_masks.shape[1], pred_masks.shape[2]
    matched_target_masks = F.interpolate(
        matched_target_masks.unsqueeze(1), size=(h_mask, w_mask), mode='nearest'
    ).squeeze(1)

    loss_mask_bce = F.binary_cross_entropy_with_logits(pred_masks_logits, matched_target_masks, reduction='mean')

    intersection = (pred_masks * matched_target_masks).sum(dim=(1, 2))
    union = pred_masks.sum(dim=(1, 2)) + matched_target_masks.sum(dim=(1, 2))
    loss_mask_dice = 1 - (2. * intersection / (union + 1e-5)).mean()

    # ---------------------------------------------------------
    # 6. Total Loss 가중치 합산
    # ---------------------------------------------------------
    total_loss = (2.0 * loss_class) + (5.0 * loss_bbox_l1) + (2.0 * loss_giou) + (5.0 * loss_mask_bce) + (5.0 * loss_mask_dice)
    
    return total_loss

def train_model(model, dataloader, matcher, num_epochs=50, learning_rate=1e-4, device='cuda'):
    model.to(device)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_imgs, batch_depths, batch_targets in progress_bar:
            batch_imgs = batch_imgs.to(device)
            batch_depths = batch_depths.to(device)

            # 타겟은 딕셔너리 리스트이므로 내부 텐서들을 각각 GPU로 이동
            targets = []
            for t in batch_targets:
                targets.append({
                    'labels': t['labels'].to(device),
                    'boxes': t['boxes'].to(device),
                    'masks': t['masks'].to(device)
                })
            
            # --- [B] 순전파 (Forward Pass) ---
            # 모델에 이미지를 넣어 8400개의 예측값과 32장의 프로토타입 마스크 추출
            pred_logits, pred_boxes, pred_coeffs, pred_prototypes = model(batch_imgs)
            
            # --- [C] 헝가리안 매칭 & Loss 계산 ---
            # 앞서 논의한 배치 단위 처리 로직(process_batch)을 호출하여 Loss를 한 번에 계산
            # (이 함수 안에서 matcher.match()가 for문으로 실행되고, 
            # 그 인덱스를 모아 Focal, L1, GIoU, Dice Loss를 합산합니다.)
            loss = compute_batch_loss(
                pred_logits, pred_boxes, pred_coeffs, pred_prototypes, 
                targets, matcher
            )
            
            # --- [D] 역전파 및 가중치 업데이트 (Backward & Optimize) ---
            optimizer.zero_grad() # 이전 배치의 기울기(Gradient) 초기화
            loss.backward()       # 오차를 바탕으로 각 가중치의 기울기 계산
            
            # 기울기 폭발(Gradient Explosion) 방지를 위한 클리핑 (선택사항, 안정성 향상)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()      # 계산된 기울기를 바탕으로 가중치 업데이트
            
            # --- [E] 로깅 (Logging) ---
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # 1 에포크 종료 후 평균 Loss 출력
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
        
        # 모델 체크포인트 저장 (10 에포크마다)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"melon_segmenter_epoch_{epoch+1}.pth")
            print(f"Model checkpoint saved!")

    return model


# 사용 예시
if __name__ == "__main__":
    # args = get_args()
    
    # # 아까 만든 Dataset에 경로 전달
    # img_path = os.path.join(args.data_dir, args.train_img)
    # mask_path = os.path.join(args.data_dir, args.train_mask)
    
    # print(f"학습을 시작합니다. 모델: {args.model_name}, 배치 사이즈: {args.batch_size}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 앞서 만든 모델과 매처 초기화
    model = FastMelonSegmenter(num_classes=1).to(device)
    matcher = SimpleHungarianMatcher(weight_class=2.0, weight_bbox=5.0, weight_giou=2.0)
    
    # DataLoader (이전 단계에서 생성한 dataloader 변수 사용)
    # dataloader = ... 
    
    # 2. 학습 시작!
    print("Training Started...")
    trained_model = train_model(model, dataloader, matcher, num_epochs=100)
    print("Training Finished!")