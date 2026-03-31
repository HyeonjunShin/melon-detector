import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert, generalized_box_iou

class MelonCriterion(nn.Module):
    def __init__(self, matcher, num_classes=1, weight_dict=None):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        # 기본 가중치 설정
        self.weight_dict = weight_dict or {
            'loss_cls': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0, 'loss_mask': 1.0
        }

    def loss_labels(self, pred_logits, target_classes, indices):
        """클래스 예측 손실 (CrossEntropy 또는 Focal Loss)"""
        idx = self._get_src_permutation_idx(indices)
        
        # 기본적으로 모든 쿼리를 '배경'으로 설정 (배경 클래스 인덱스 = self.num_classes)
        target_classes_o = torch.full(pred_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=pred_logits.device)
        
        # 매칭된 쿼리에만 실제 클래스 할당
        target_classes_o[idx] = target_classes
        
        loss_cls = F.cross_entropy(pred_logits.transpose(1, 2), target_classes_o)
        return loss_cls

    def loss_boxes(self, pred_boxes, target_boxes, indices, num_boxes):
        """박스 예측 손실 (L1 + GIoU)"""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[idx]
        target_boxes = torch.cat([t[i] for t, (_, i) in zip(target_boxes, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none').sum() / num_boxes
        
        # GIoU 계산
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_convert(src_boxes, "cxcywh", "xyxy"),
            box_convert(target_boxes, "cxcywh", "xyxy")
        ))
        loss_giou = loss_giou.sum() / num_boxes
        
        return loss_bbox, loss_giou

    def loss_masks(self, prototypes, pred_coeffs, target_masks, indices, num_boxes):
        """마스크 조립 및 손실 (YOLACT 방식)"""
        src_idx = self._get_src_permutation_idx(indices)
        
        # 1. 매칭된 쿼리의 계수 추출
        coeffs = pred_coeffs[src_idx] # [num_total_matched, 32]
        
        # 2. 배치별 프로토타입 확장
        # target_masks는 리스트 형태이므로 배치 인덱스에 맞춰 프로토타입 참조
        batch_idx = src_idx[0]
        proto = prototypes[batch_idx] # [num_total_matched, 32, H/4, W/4]
        
        # 3. 마스크 조립 (Linear Combination)
        # coeffs: [N, 32], proto: [N, 32, H, W] -> [N, H, W]
        pred_masks = torch.einsum('nc,nchw->nhw', coeffs, proto)
        
        # 4. 정답 마스크 준비 및 리사이즈
        gt_masks = torch.cat([t[i] for t, (_, i) in zip(target_masks, indices)], dim=0)
        # 정답 마스크를 프로토타입 크기(H/4, W/4)에 맞게 축소
        gt_masks = F.interpolate(gt_masks.unsqueeze(1), size=pred_masks.shape[-2:], mode='nearest').squeeze(1)

        loss_mask = F.binary_cross_entropy_with_logits(pred_masks, gt_masks)
        return loss_mask

    def _get_src_permutation_idx(self, indices):
        # 배치를 가로질러 인덱스를 생성하는 헬퍼 함수
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        """
        outputs: {'pred_logits', 'pred_boxes', 'pred_coeffs', 'prototypes'}
        targets: [{'labels', 'boxes', 'masks'}, ...]
        """
        pred_logits, pred_boxes, pred_coeffs, prototypes = outputs
        
        # 헝가리안 매칭 수행
        indices = []
        for i in range(len(targets)):
            match_idx = self.matcher.match(pred_logits[i], pred_boxes[i], 
                                          targets[i]['labels'], targets[i]['boxes'])
            indices.append(match_idx)

        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = max(num_boxes, 1)

        # 각 Loss 계산
        l_cls = self.loss_labels(pred_logits, torch.cat([t['labels'] for t in targets]), indices)
        l_bbox, l_giou = self.loss_boxes(pred_boxes, [t['boxes'] for t in targets], indices, num_boxes)
        l_mask = self.loss_masks(prototypes, pred_coeffs, [t['masks'] for t in targets], indices, num_boxes)

        # 최종 가중합
        total_loss = (self.weight_dict['loss_cls'] * l_cls +
                      self.weight_dict['loss_bbox'] * l_bbox +
                      self.weight_dict['loss_giou'] * l_giou +
                      self.weight_dict['loss_mask'] * l_mask)
        
        return total_loss