import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou, box_convert

class SimpleHungarianMatcher:
    def __init__(self, weight_class=2.0, weight_bbox=5.0, weight_giou=2.0):
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou

    @torch.no_grad()
    def match(self, pred_logits, pred_boxes, target_classes, target_boxes):
        """
        pred_logits: [Queries, Num_Classes]
        pred_boxes: [Queries, 4] (cxcywh)
        target_classes: [N]
        target_boxes: [N, 4] (cxcywh)
        """
        # --- [추가] 예외 처리: 정답 객체가 없으면 바로 빈 텐서 반환 ---
        num_queries = pred_logits.shape[0] 
        num_gts = target_boxes.shape[0]
        if num_gts == 0:
            return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)

        # 1. Class Cost (Logits -> Sigmoid)
        out_prob = pred_logits.sigmoid() 
        # [Queries, N] 형태의 비용 행렬 생성
        cost_class = -out_prob[:, target_classes] 

        # 2. BBox L1 Cost
        cost_bbox = torch.cdist(pred_boxes, target_boxes, p=1)

        # 3. GIoU Cost
        # --- [추가] clamp(0, 1)로 학습 초기 수치 안정성 확보 ---
        p_boxes = box_convert(pred_boxes, "cxcywh", "xyxy").clamp(0, 1)
        t_boxes = box_convert(target_boxes, "cxcywh", "xyxy").clamp(0, 1)
        
        # GIoU는 1에 가까울수록 좋으므로 음수 처리
        cost_giou = -generalized_box_iou(p_boxes, t_boxes)

        # 4. Final Cost Matrix
        C = (self.weight_class * cost_class + 
             self.weight_bbox * cost_bbox + 
             self.weight_giou * cost_giou)
        
        # --- [추가] NaN/Inf 값 체크 (학습 터짐 방지) ---
        C = torch.where(torch.isnan(C) | torch.isinf(C), torch.full_like(C, 1e6), C)

        # SciPy는 CPU 전용이므로 numpy 변환
        C = C.cpu().numpy()
        pred_idx, target_idx = linear_sum_assignment(C)

        return torch.as_tensor(pred_idx, dtype=torch.int64), torch.as_tensor(target_idx, dtype=torch.int64)