import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 사용자 정의 모듈 (파일명에 맞게 import 확인)
from datasets import MelonDataset, melon_collate_fn
from model import FastMelonSegmenter
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision


# ---------------------------------------------------------
# 1. 헝가리안 매처 (Hungarian Matcher)
# ---------------------------------------------------------
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_bbox=5.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox

    @torch.no_grad()
    def forward(self, outputs, targets):
        num_preds = outputs["pred_logits"].shape[0]
        num_gt = targets["labels"].shape[0]
        if num_gt == 0:
            return None, None

        prob = outputs["pred_logits"].sigmoid()
        cost_class = -prob[:, 0:1].expand(num_preds, num_gt)
        cost_bbox = torch.cdist(outputs["pred_boxes"], targets["boxes"], p=1)

        C = (self.cost_class * cost_class + self.cost_bbox * cost_bbox).cpu()
        src_idx, tgt_idx = linear_sum_assignment(C.numpy())
        return torch.as_tensor(src_idx, dtype=torch.int64), torch.as_tensor(
            tgt_idx, dtype=torch.int64
        )


# ---------------------------------------------------------
# 2. 결과 시각화 및 파일 저장 함수
# ---------------------------------------------------------
def save_visual_result(model, dataset, device, epoch, save_dir="./visual_results"):
    """
    학습 중인 모델을 사용하여 한 프레임에 대한 예측 결과를 이미지로 저장합니다.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # 데이터셋에서 첫 번째 샘플 가져오기
    images, targets = dataset[0]  # [T, C, H, W]
    img_input = images.unsqueeze(0).to(device)  # [1, T, C, H, W]

    with torch.no_grad():
        pred_logits, pred_boxes, pred_coeffs, prototypes = model(img_input)

    # 첫 번째 프레임 결과만 시각화
    idx = 0
    scores = pred_logits[idx].sigmoid().squeeze(-1)
    conf_mask = scores > 0.3  # 시각화 시에는 조금 높은 임계값 사용

    final_scores = scores[conf_mask]
    final_boxes = pred_boxes[idx][conf_mask]
    final_coeffs = pred_coeffs[idx][conf_mask]

    # 배경 이미지 준비 (정규화 해제)
    bg_img = images[0].permute(1, 2, 0).cpu().numpy()
    bg_img = (bg_img * 255).astype(np.uint8)
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_RGB2BGR)

    if len(final_scores) > 0:
        # 마스크 조립
        m = torch.sigmoid(torch.einsum("nc,chw->nhw", final_coeffs, prototypes[idx]))
        m = F.interpolate(m.unsqueeze(1), size=(256, 448), mode="bilinear").squeeze(1)
        masks = (m > 0.5).cpu().numpy()

        for i in range(len(final_scores)):
            # 1. 박스 그리기
            pb = final_boxes[i]
            x1 = int((pb[0] - pb[2] / 2) * 448)
            y1 = int((pb[1] - pb[3] / 2) * 256)
            x2 = int((pb[0] + pb[2] / 2) * 448)
            y2 = int((pb[1] + pb[3] / 2) * 256)
            cv2.rectangle(bg_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 2. 마스크 씌우기 (색상 입히기)
            mask = masks[i]
            color_mask = np.zeros_like(bg_img)
            color_mask[mask] = [0, 255, 255]  # 노란색 멜론
            bg_img = cv2.addWeighted(bg_img, 1.0, color_mask, 0.5, 0)

            # 3. 점수 표시
            cv2.putText(
                bg_img,
                f"{final_scores[i]:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    cv2.imwrite(os.path.join(save_dir, f"epoch_{epoch:03d}.png"), bg_img)
    model.train()


# ---------------------------------------------------------
# 3. 손실 함수 (Loss Function)
# ---------------------------------------------------------
def compute_loss(preds, targets, matcher):
    pred_logits, pred_boxes, pred_coeffs, prototypes = preds
    batch_size_total = pred_logits.shape[0]
    device = pred_logits.device
    l_cls, l_box, l_mask = 0, 0, 0

    for i in range(batch_size_total):
        gt = targets[i]
        gt_labels = gt["labels"].to(device)
        gt_boxes = gt["boxes"].to(device)
        gt_masks = gt["masks"].to(device)
        out = {"pred_logits": pred_logits[i], "pred_boxes": pred_boxes[i]}

        if len(gt_labels) == 0:
            l_cls += F.binary_cross_entropy_with_logits(
                pred_logits[i], torch.zeros_like(pred_logits[i])
            )
            continue

        src, tgt = matcher(out, {"labels": gt_labels, "boxes": gt_boxes})
        if src is None:
            continue

        # Class Loss
        t_cls = torch.zeros_like(pred_logits[i])
        t_cls[src, 0] = 1.0
        l_cls += F.binary_cross_entropy_with_logits(pred_logits[i], t_cls)
        # Box Loss
        l_box += F.l1_loss(pred_boxes[i][src], gt_boxes[tgt])
        # Mask Loss
        m_p = torch.sigmoid(torch.einsum("nc,chw->nhw", pred_coeffs[i][src], prototypes[i]))
        m_gt = F.interpolate(
            gt_masks[tgt].unsqueeze(1).float(), size=m_p.shape[-2:], mode="bilinear"
        ).squeeze(1)
        l_mask += F.binary_cross_entropy(m_p, m_gt)

    return (l_cls * 1.0 + l_box * 2.0 + l_mask * 5.0) / batch_size_total


# ---------------------------------------------------------
# 4. 평가 함수 (mAP)
# ---------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="segm")
    for images, targets in loader:
        images = images.to(device)
        flat_targets = [f for b in targets for f in b]
        p_logits, p_boxes, p_coeffs, p_protos = model(images)

        preds_list, gts_list = [], []
        for i in range(p_logits.shape[0]):
            scores = p_logits[i].sigmoid().squeeze(-1)
            k = min(100, scores.size(0))
            topk_scores, topk_idx = torch.topk(scores, k)
            keep = topk_scores > 0.05

            final_scores = topk_scores[keep]
            pb = p_boxes[i][topk_idx[keep]]

            boxes = torch.zeros_like(pb)
            boxes[:, 0] = (pb[:, 0] - pb[:, 2] / 2) * 448
            boxes[:, 1] = (pb[:, 1] - pb[:, 3] / 2) * 256
            boxes[:, 2] = (pb[:, 0] + pb[:, 2] / 2) * 448
            boxes[:, 3] = (pb[:, 1] + pb[:, 3] / 2) * 256

            if len(final_scores) > 0:
                m = torch.sigmoid(
                    torch.einsum("nc,chw->nhw", p_coeffs[i][topk_idx[keep]], p_protos[i])
                )
                m = F.interpolate(m.unsqueeze(1), size=(256, 448), mode="bilinear").squeeze(1)
                masks = m > 0.5
            else:
                masks = torch.zeros((0, 256, 448), device=device, dtype=torch.bool)

            preds_list.append(
                {
                    "boxes": boxes,
                    "scores": final_scores,
                    "labels": torch.zeros(len(final_scores), dtype=torch.int64, device=device),
                    "masks": masks,
                }
            )

            gt = flat_targets[i]
            gb = gt["boxes"].to(device)
            gt_boxes = torch.zeros_like(gb)
            gt_boxes[:, 0] = (gb[:, 0] - gb[:, 2] / 2) * 448
            gt_boxes[:, 1] = (gb[:, 1] - gb[:, 3] / 2) * 256
            gt_boxes[:, 2] = (gb[:, 0] + gb[:, 2] / 2) * 448
            gt_boxes[:, 3] = (gb[:, 1] + gb[:, 3] / 2) * 256
            gts_list.append(
                {
                    "boxes": gt_boxes,
                    "labels": gt["labels"].to(device),
                    "masks": gt["masks"].to(device).bool(),
                }
            )

        metric.update(preds_list, gts_list)
    results = metric.compute()
    model.train()
    return results


# ---------------------------------------------------------
# 5. 메인 학습 루프
# ---------------------------------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_seq = 10
    num_epochs = 200  # 에포크 확장
    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    load_ckpt = "./checkpoints/best_model.pth"

    dataset = MelonDataset(num_frames=10000, num_seq=num_seq, target_image_size=(256, 448))
    train_loader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=melon_collate_fn, num_workers=4
    )
    # 테스트용 일부 데이터셋 분리 (속도를 위해 앞부분 20개 시퀀스만)
    test_subset = torch.utils.data.Subset(dataset, range(20))
    test_loader = DataLoader(test_subset, batch_size=4, shuffle=False, collate_fn=melon_collate_fn)

    model = FastMelonSegmenter(num_classes=1, num_prototypes=32, n_segment=num_seq).to(device)
    if load_ckpt is not None:
        checkpoint = torch.load(load_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # 스케줄러: CosineAnnealingLR (정확도 향상 핵심)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    matcher = HungarianMatcher()

    best_map = 0.0

    print("--- 멜론 탐지 시스템 학습 시작 ---")
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            flat_targets = [f for b in targets for f in b]

            optimizer.zero_grad()
            preds = model(images)
            loss = compute_loss(preds, flat_targets, matcher)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item()
            if i % 50 == 0:
                print(f"Epoch {epoch} Step {i} Loss: {loss.item():.4f}")

        # 에포크 종료 후 평가 및 시각화 저장
        res = evaluate(model, test_loader, device)
        cur_map = res["map_50"].item()
        print(
            f"Epoch {epoch} 완료 -> 평균 Loss: {epoch_loss / len(train_loader):.4f}, Mask mAP: {cur_map:.4f}"
        )

        # 이미지 결과 저장
        save_visual_result(model, dataset, device, epoch)

        # 베스트 모델 저장
        if cur_map > best_map:
            best_map = cur_map
            torch.save(
                {"model": model.state_dict(), "map": best_map},
                os.path.join(ckpt_dir, "best_model.pth"),
            )
            print(f"*** Best 모델 저장됨 (mAP: {best_map:.4f}) ***")

        # 20 에포크마다 중간 저장
        if epoch % 20 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch}.pth"))

        scheduler.step()  # 학습률 조정


if __name__ == "__main__":
    train()
