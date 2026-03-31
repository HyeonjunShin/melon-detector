import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision.ops import sigmoid_focal_loss, generalized_box_iou, box_convert
from model import FastMelonSegmenter, OnlineTemporalShift

def compute_batch_loss(pred_logits, pred_boxes, pred_coeffs, pred_prototypes, pred_normals, targets, matcher):
    B = pred_logits.shape[0]
    device = pred_logits.device
    indices = []
    
    # 1. 헝가리안 매칭
    for i in range(B):
        p_logits, p_boxes = pred_logits[i], pred_boxes[i]
        t_classes, t_boxes = targets[i]['labels'], targets[i]['boxes']
        
        if len(t_classes) == 0:
            indices.append((torch.empty(0, dtype=torch.int64, device=device), torch.empty(0, dtype=torch.int64, device=device)))
            continue
        matched_p_idx, matched_t_idx = matcher.match(p_logits, p_boxes, t_classes, t_boxes)
        indices.append((matched_p_idx.to(device), matched_t_idx.to(device)))

    batch_idx = torch.cat([torch.full_like(p, i) for i, (p, _) in enumerate(indices)])
    pred_idx = torch.cat([p for (p, _) in indices])
    target_idx = torch.cat([t for (_, t) in indices])

    # 2. Classification Loss
    target_classes = torch.zeros_like(pred_logits)
    if len(target_idx) > 0:
        target_classes_concat = torch.cat([t['labels'][tgt] for t, (_, tgt) in zip(targets, indices)])
        target_classes[batch_idx, pred_idx, target_classes_concat] = 1.0
    loss_class = sigmoid_focal_loss(pred_logits, target_classes, alpha=0.25, gamma=2.0, reduction='mean')

    # 3. Normal Loss (배경 포함 전체 픽셀)
    gt_normals = torch.stack([t['normals'] for t in targets])
    gt_normals_res = F.interpolate(gt_normals, size=pred_normals.shape[2:], mode='bilinear', align_corners=False)
    loss_normal = F.l1_loss(pred_normals, gt_normals_res)

    if len(target_idx) == 0:
        return (2.0 * loss_class) + (10.0 * loss_normal)

    # 4. Box Loss
    matched_pred_boxes = pred_boxes[batch_idx, pred_idx]
    matched_target_boxes = torch.cat([t['boxes'][tgt] for t, (_, tgt) in zip(targets, indices)])
    loss_bbox_l1 = F.l1_loss(matched_pred_boxes, matched_target_boxes)
    loss_giou = 1 - torch.diag(generalized_box_iou(box_convert(matched_pred_boxes, "cxcywh", "xyxy"), 
                                                   box_convert(matched_target_boxes, "cxcywh", "xyxy"))).mean()

    # 5. Mask Loss
    matched_coeffs = pred_coeffs[batch_idx, pred_idx]
    selected_protos = pred_prototypes[batch_idx]
    pred_masks_logits = torch.einsum('nc,nchw->nhw', matched_coeffs, selected_protos)
    
    matched_target_masks = torch.cat([t['masks'][tgt] for t, (_, tgt) in zip(targets, indices)]).float()
    matched_target_masks = F.interpolate(matched_target_masks.unsqueeze(1), size=pred_masks_logits.shape[1:], mode='nearest').squeeze(1)
    
    loss_mask_bce = F.binary_cross_entropy_with_logits(pred_masks_logits, matched_target_masks)
    pred_masks = pred_masks_logits.sigmoid()
    intersection = (pred_masks * matched_target_masks).sum(dim=(1, 2))
    union = pred_masks.sum(dim=(1, 2)) + matched_target_masks.sum(dim=(1, 2))
    loss_mask_dice = 1 - (2. * intersection / (union + 1e-5)).mean()

    return (2.0 * loss_class) + (5.0 * loss_bbox_l1) + (2.0 * loss_giou) + (5.0 * loss_mask_bce) + (5.0 * loss_mask_dice) + (10.0 * loss_normal)

def train_model(model, dataloader, matcher, num_epochs=100, lr=1e-4, device='cuda'):
    model.to(device)
    # 가중치 감쇠(weight_decay)를 포함한 AdamW 옵티마이저
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for seq_colors, seq_depths, seq_normals, seq_targets in pbar:
            # seq_colors: [B, S, 3, H, W]
            # seq_targets: [B] 리스트 (각 원소는 'bboxes', 'masks', 'labels'를 가진 딕셔너리)
            B, S, _, H, W = seq_colors.shape
            
            # 1. 새 시퀀스 시작 시 TSM 버퍼 초기화 (이전 비디오의 잔상 제거)
            for m in model.modules():
                if hasattr(m, 'reset_buffer'):
                    m.reset_buffer()
            
            optimizer.zero_grad()
            seq_total_loss = 0
            
            # 2. 시퀀스 내 프레임을 순서대로(t=0 to S-1) 처리
            for t in range(S):
                # 현재 프레임 데이터 결합 (RGB + Depth = 4채널)
                img_t = seq_colors[:, t].to(device)
                dep_t = seq_depths[:, t].to(device)
                input_t = torch.cat([img_t, dep_t], dim=1) # [B, 4, H, W]
                
                # 3. 타겟 딕셔너리 재구성 (bboxes -> boxes 키 변경)
                targets_t = []
                for b in range(B):
                    # 데이터셋의 'bboxes' 키를 'boxes'로 매핑하여 loss 함수와 호환시킴
                    targets_t.append({
                        'labels': seq_targets[b]['labels'][t].to(device),
                        'boxes':  seq_targets[b]['bboxes'][t].to(device), # bboxes 사용
                        'masks':  seq_targets[b]['masks'][t].to(device),
                        'normals': seq_normals[b, t].to(device)
                    })
                
                # 4. 순전파 (Forward Pass)
                # OnlineTemporalShift 덕분에 t-1 프레임의 특징이 t프레임에 영향을 줍니다.
                pred_logits, pred_boxes, pred_coeffs, prototypes, pred_normals = model(input_t)
                
                # 5. Loss 계산
                loss = compute_batch_loss(
                    pred_logits, pred_boxes, pred_coeffs, prototypes, pred_normals,
                    targets_t, matcher
                )
                
                # 6. 시퀀스 평균 Loss를 위해 S로 나누어 역전파 (메모리 절약형)
                (loss / S).backward() 
                seq_total_loss += (loss.item() / S)
            
            # 기울기 폭발 방지 및 가중치 업데이트
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            epoch_loss += seq_total_loss
            pbar.set_postfix({'loss': f"{seq_total_loss:.4f}"})
            
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} 완료. 평균 Loss: {avg_epoch_loss:.4f}")
        
        # 10 에포크마다 모델 저장
        if (epoch + 1) % 10 == 0:
            save_path = f"melon_tsm_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"💾 모델 저장 완료: {save_path}")

    return model