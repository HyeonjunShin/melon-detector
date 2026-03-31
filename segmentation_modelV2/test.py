import torch
import torch.nn.functional as F
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import draw_segmentation_masks, make_grid
from torchvision.ops import nms
from datasets import MelonDataset, custom_collate_fn
from model import RealTimeMelonSegmenter

def visualize_batch_results(model, dataloader, device='cuda', threshold=0.6, iou_threshold=0.3):
    model.to(device)
    model.eval()
    
    # DataLoader에서 첫 번째 배치 가져오기
    images, targets = next(iter(dataloader))
    batch_size = images.shape[0]
    input_tensor = images.to(device)
    
    # 1. 배치 추론 및 순수 모델 시간 측정 (Warm-up 포함)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        for _ in range(2): # Warm-up
            _ = model(input_tensor)
            
    torch.cuda.synchronize()
    start_event.record()
    with torch.no_grad():
        # 모델 출력: (logits, boxes, coeffs, features)
        prediction = model(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    
    model_time_ms = start_event.elapsed_time(end_event)
    
    # 2. 배치 전체 사후 처리 (NMS + 마스크 합성) 시간 측정 시작
    post_start = time.time()
    
    batch_vis_images = [] # 각 이미지의 시각화 결과를 저장할 리스트
    total_melons = 0
    
    # 배치 내의 각 이미지에 대해 루프 수행 (NMS 및 마스크 합성은 이미지별로 필요)
    for b in range(batch_size):
        # b번째 이미지의 데이터 추출
        img_b = images[b] # [3, H, W] (0~1 float)
        
        logits_b = prediction[0][b]  # [100, 2]
        boxes_b = prediction[1][b]   # [100, 4] (cxcywh)
        coeffs_b = prediction[2][b]  # [100, 32]
        features_b = prediction[3][b] # [32, H_f, W_f]

        # 확률 계산 및 1차 필터링
        probs_b = torch.softmax(logits_b, dim=-1)
        scores_b = probs_b[:, 1] # 1번 인덱스가 참외
        
        initial_keep = scores_b > threshold
        if not initial_keep.any():
            # 검출 안 된 경우 원본 이미지 그대로 추가
            img_int_b = (img_b * 255).to(torch.uint8)
            # draw_segmentation_masks는 최소 1개의 마스크가 필요하므로 더미 마스크 생성
            dummy_mask = torch.zeros((1, img_int_b.shape[1], img_int_b.shape[2]), dtype=torch.bool)
            res_b = draw_segmentation_masks(img_int_b, dummy_mask, alpha=0.0)
            batch_vis_images.append(res_b)
            continue

        # NMS (중복 제거)
        v_boxes = boxes_b[initial_keep]
        v_scores = scores_b[initial_keep]
        
        boxes_xyxy = v_boxes.clone()
        boxes_xyxy[:, 0] = v_boxes[:, 0] - v_boxes[:, 2] / 2
        boxes_xyxy[:, 1] = v_boxes[:, 1] - v_boxes[:, 3] / 2
        boxes_xyxy[:, 2] = v_boxes[:, 0] + v_boxes[:, 2] / 2
        boxes_xyxy[:, 3] = v_boxes[:, 1] + v_boxes[:, 3] / 2

        nms_idx = nms(boxes_xyxy, v_scores, iou_threshold)
        final_indices = torch.where(initial_keep)[0][nms_idx]
        num_melons = len(final_indices)
        total_melons += num_melons

        # 마스크 합성 (Matrix Multiplication) 및 원본 크기 복원
        f_coeffs = coeffs_b[final_indices]
        C_f, H_f, W_f = features_b.shape
        masks = torch.mm(f_coeffs, features_b.view(C_f, -1)).view(-1, 1, H_f, W_f)
        
        img_int_b = (img_b * 255).to(torch.uint8)
        _, target_h, target_w = img_int_b.shape
        
        masks = torch.sigmoid(masks)
        # NMS를 통과한 마스크만 리사이즈
        masks = F.interpolate(masks, size=(target_h, target_w), mode='bilinear', align_corners=False)
        masks_tensor = (masks.squeeze(1) > 0.5).cpu() # [N_keep, H, W] 형태의 Boolean 텐서

        # 시각화 (draw_segmentation_masks 활용)
        res_tensor_b = draw_segmentation_masks(img_int_b, masks_tensor, alpha=0.4, colors="yellow")
        
        # 무게 중심 시각화 (Tensor 상에서 원을 그리는 것은 까다로워 OpenCV 잠시 활용)
        vis_img_np = res_tensor_b.permute(1, 2, 0).cpu().numpy().copy()
        vis_img_np = cv2.cvtColor(vis_img_np, cv2.COLOR_RGB2BGR)
        
        # Numpy 마스크로 변경하여 Centroid 계산
        masks_np = masks_tensor.numpy()
        for i, mask in enumerate(masks_np):
            y_coords, x_coords = np.where(mask)
            if len(x_coords) > 0:
                cx, cy = int(np.mean(x_coords)), int(np.mean(y_coords))
                # 빨간 점 + 흰색 테두리
                cv2.circle(vis_img_np, (cx, cy), 6, (0, 0, 255), -1) 
                cv2.circle(vis_img_np, (cx, cy), 10, (255, 255, 255), 2)
                # ID 텍스트
                cv2.putText(vis_img_np, f"ID:{i}", (cx + 12, cy - 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 다시 Tensor로 변환하여 리스트에 추가
        final_vis_tensor = torch.from_numpy(cv2.cvtColor(vis_img_np, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
        batch_vis_images.append(final_vis_tensor)

    # 3. 사후 처리 시간 측정 종료
    post_end = time.time()
    post_time_ms = (post_end - post_start) * 1000
    
    print(f"\n" + "="*45)
    print(f"📊 [Batch Report - Size: {batch_size}]")
    print(f"1. Model Pure Inference (Batch): {model_time_ms:.2f} ms")
    print(f"2. Post-processing (NMS+Mask, Batch): {post_time_ms:.2f} ms")
    print(f"🚀 FPS (Total Process): {1000 / (model_time_ms + post_time_ms):.2f}")
    print(f"✅ Total Detected Melons: {total_melons}")
    print("="*45)

    # 4. 이미지 그리드 생성 (torchvision.utils.make_grid 활용)
    # batch_vis_images는 Tensor 리스트이므로 stack을 통해 [B, 3, H, W] 텐서로 변환
    vis_grid_tensor = torch.stack(batch_vis_images)
    
    # nrow: 한 줄에 보여줄 이미지 개수 (배치 크기의 제곱근 정도로 설정)
    nrow = int(np.sqrt(batch_size))
    grid_img = make_grid(vis_grid_tensor, nrow=nrow, padding=1, normalize=False)
    
    # 5. 화면 출력
    plt.figure(figsize=(16, 12))
    # make_grid 결과는 [3, H_grid, W_grid]이므로 permute(1, 2, 0)
    plt.imshow(grid_img.permute(1, 2, 0).numpy())
    plt.title(f"Batch Visualization | Total Melons: {total_melons} | Pure Model: {model_time_ms:.1f}ms")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 모델 로드
    model = RealTimeMelonSegmenter()
    model.load_state_dict(torch.load("best_melon_model.pth", map_location=device))
    
    # 2. 데이터셋 및 DataLoader 로드 (배치 크기 설정)
    test_dataset = MelonDataset(
        path_data="/home/hyeonjun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/_melon_dataset/data",
        path_txt="/home/hyeonjun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/_melon_dataset/test.txt",
        img_size=(512, 896)
    )
    
    # BATCH_SIZE를 원하는 크기로 조절하세요 (예: 4, 9, 16 등 제곱수가 보기 좋습니다.)
    BATCH_SIZE = 9 
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    
    # 3. 실행
    visualize_batch_results(model, test_loader, device=device, threshold=0.4, iou_threshold=0.1)