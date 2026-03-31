import zenoh
import cv2
import numpy as np
import struct
import time
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from queue import Queue
from model import FastMelonSegmenter

# --- 전역 설정 ---
HEADER_SIZE = 20
MODEL_PATH = "melon_segmenter_epoch_30.pth"
CONF_THRESHOLD = 0.6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 추적 및 지터링 개선 설정 ---
ALPHA = 0.1
MAX_LOST_FRAMES = 15 
DIST_THRESHOLD = 30
tracked_objects = []

# 1. 모델 로드
model = FastMelonSegmenter(num_classes=1, num_prototypes=32).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

frame_queue = Queue(maxsize=1)

def frame_handler(sample):
    if frame_queue.full():
        try: frame_queue.get_nowait()
        except: pass
    frame_queue.put(sample.payload.to_bytes())

@torch.no_grad()
def process_inference(img_bgr):
    img_resized = cv2.resize(img_bgr, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = TF.to_tensor(img_rgb).unsqueeze(0).to(DEVICE)

    logits, boxes, coeffs, prototypes = model(img_tensor)
    scores = logits.sigmoid().squeeze() 
    if scores.dim() > 1: scores = scores[:, 0]

    keep_idx = torch.where(scores > CONF_THRESHOLD)[0]
    if len(keep_idx) == 0:
        return [], [], img_resized

    valid_scores = scores[keep_idx]
    valid_coeffs = coeffs[0, keep_idx]

    pred_masks = torch.einsum('nc,chw->nhw', valid_coeffs, prototypes[0])
    pred_masks = F.interpolate(pred_masks.unsqueeze(1), size=(640, 640), 
                               mode='bilinear', align_corners=False).squeeze(1).sigmoid()
    
    binary_masks = (pred_masks > 0.5).cpu().numpy().astype(np.uint8)
    return binary_masks, valid_scores.cpu().numpy(), img_resized

# Zenoh 설정
conf = zenoh.Config()
conf.insert_json5("transport/shared_memory/enabled", "true")
session = zenoh.open(conf)
sub = session.declare_subscriber("camera/raw", frame_handler)

# --- [윈도우 설정 추가] ---
WINDOW_NAME = "FastMelon Segmentation (Resizable)"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) # 마우스로 크기 조절 가능하게 설정
cv2.resizeWindow(WINDOW_NAME, 1280, 720)        # 초기 창 크기 설정

prev_time = time.perf_counter()

try:
    while True:
        if frame_queue.empty(): continue
            
        raw = frame_queue.get()
        send_time, c_size, rows, cols = struct.unpack('<dIII', raw[:HEADER_SIZE])
        color_np = np.frombuffer(raw[HEADER_SIZE : HEADER_SIZE + c_size], dtype=np.uint8)
        img_raw = cv2.imdecode(color_np, cv2.IMREAD_COLOR)
        
        if img_raw is not None:
            new_masks, new_scores, vis_img = process_inference(img_raw)
            
            detected_now = []
            for i in range(len(new_scores)):
                mask = new_masks[i]
                M = cv2.moments(mask)
                if M["m00"] > 300:
                    cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    detected_now.append({'center': (cX, cY), 'score': new_scores[i], 'mask': mask})

            # 추적 로직
            updated_tracked_objects = []
            matched_indices = set()

            for obj in tracked_objects:
                best_match_idx = -1
                min_dist = DIST_THRESHOLD
                for idx, det in enumerate(detected_now):
                    if idx in matched_indices: continue
                    dist = np.sqrt((obj['center'][0]-det['center'][0])**2 + (obj['center'][1]-det['center'][1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_match_idx = idx

                if best_match_idx != -1:
                    det = detected_now[best_match_idx]
                    new_center = (
                        int(ALPHA * det['center'][0] + (1 - ALPHA) * obj['center'][0]),
                        int(ALPHA * det['center'][1] + (1 - ALPHA) * obj['center'][1])
                    )
                    obj.update({'center': new_center, 'score': det['score'], 'mask': det['mask'], 'lost_count': 0})
                    updated_tracked_objects.append(obj)
                    matched_indices.add(best_match_idx)
                else:
                    obj['lost_count'] += 1
                    if obj['lost_count'] <= MAX_LOST_FRAMES:
                        updated_tracked_objects.append(obj)

            for idx, det in enumerate(detected_now):
                if idx not in matched_indices:
                    det['lost_count'] = 0
                    det['color'] = np.random.randint(0, 255, 3).tolist()
                    updated_tracked_objects.append(det)

            tracked_objects = updated_tracked_objects

            # 시각화
            mask_overlay = np.zeros_like(vis_img)
            for obj in tracked_objects:
                color = obj['color']
                mask_overlay[obj['mask'] == 1] = color
                cv2.circle(vis_img, obj['center'], 6, (255, 255, 255), -1)
                cv2.putText(vis_img, f"ID:{id(obj)%100} {obj['score']:.2f}", (obj['center'][0]+10, obj['center'][1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            vis_img = cv2.addWeighted(vis_img, 0.7, mask_overlay, 0.3, 0)

            # FPS 표시
            curr_time = time.perf_counter()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(vis_img, f"FPS: {fps:.1f} Tracked: {len(tracked_objects)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # [수정] 위에서 정의한 WINDOW_NAME 사용
            cv2.imshow(WINDOW_NAME, vis_img)
            
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    session.close()
    cv2.destroyAllWindows()