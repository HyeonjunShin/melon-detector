from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from camera_devices.kinect_wrapper import KinectCamera
from model import FastMelonSegmenter
from scipy.optimize import linear_sum_assignment
from torchvision.ops import nms
from torchvision.transforms import v2


# ==========================================
# 1. 개별 객체 상태 관리 클래스 (안정성 강화)
# ==========================================
class TrackedMelon:
    def __init__(self, melon_id, pos, norm, mask, pixel_pos, buffer_size=10):
        self.id = melon_id
        self.lost_count = 0
        self.dt = 0.1  # 예상 프레임 간격

        # 데이터 안정화 버퍼
        self.pos_history = deque([pos], maxlen=buffer_size)
        self.norm_history = deque([norm], maxlen=buffer_size)

        # 🔥 깜빡임 방지 핵심: 마지막 유효 시각 정보 저장
        self.last_mask = mask
        self.last_pixel_pos = pixel_pos  # (cX, cY)

        # 상태 변수 (위치, 속도)
        self.kf_pos = pos.copy()
        self.kf_vel = np.zeros(3)

    def predict(self):
        """관성 기반 위치 예측"""
        return self.kf_pos + self.kf_vel * self.dt

    def update(self, pos, norm, mask, pixel_pos):
        """새로운 측정값으로 업데이트"""
        # 속도 추정 및 스무딩 (0.9 가중치로 급격한 변화 억제)
        new_vel = (pos - self.kf_pos) / self.dt
        self.kf_vel = 0.9 * self.kf_vel + 0.1 * new_vel
        self.kf_pos = pos

        self.pos_history.append(pos)
        self.norm_history.append(norm)

        # 시각 정보 갱신
        self.last_mask = mask
        self.last_pixel_pos = pixel_pos
        self.lost_count = 0

    def get_stable_state(self):
        """안정화된 데이터 반환"""
        avg_pos = np.mean(list(self.pos_history), axis=0)
        avg_norm = np.mean(list(self.norm_history), axis=0)

        norm_val = np.linalg.norm(avg_norm)
        if norm_val > 0:
            avg_norm /= norm_val
        return avg_pos, avg_norm


# ==========================================
# 2. 월드 모델 매니저 (Global Manager)
# ==========================================
class WorldModelManager:
    def __init__(self, dist_threshold=0.15, max_lost=50):
        self.tracks = []
        self.next_id = 0
        self.dist_threshold = dist_threshold  # 15cm 매칭 임계값
        self.max_lost = max_lost  # 가려져도 50프레임(약 1.5초) 유지

    def update(self, current_detections):
        """
        current_detections: [(pos, norm, pixel_coord, mask), ...]
        """
        num_tracks = len(self.tracks)
        num_dets = len(current_detections)

        # 1. 기존 객체 예측 위치
        preds = [t.predict() for t in self.tracks]

        matched_track_indices = set()
        matched_det_indices = set()

        if num_tracks > 0 and num_dets > 0:
            # 2. 헝가리안 매칭용 Cost Matrix
            cost_matrix = np.zeros((num_tracks, num_dets))
            for i in range(num_tracks):
                for j in range(num_dets):
                    cost_matrix[i, j] = np.linalg.norm(preds[i] - current_detections[j][0])

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for t_idx, d_idx in zip(row_ind, col_ind):
                if cost_matrix[t_idx, d_idx] < self.dist_threshold:
                    d = current_detections[d_idx]
                    self.tracks[t_idx].update(d[0], d[1], d[3], d[2])
                    matched_track_indices.add(t_idx)
                    matched_det_indices.add(d_idx)

        # 3. 매칭 실패 기존 트랙: 관성 이동 및 Lost 증가
        for i, track in enumerate(self.tracks):
            if i not in matched_track_indices:
                track.lost_count += 1
                track.kf_pos = preds[i]

        # 4. 새로운 탐지: 신규 등록
        for j in range(num_dets):
            if j not in matched_det_indices:
                d = current_detections[j]
                self.tracks.append(TrackedMelon(self.next_id, d[0], d[1], d[3], d[2]))
                self.next_id += 1

        # 5. 수명 초과 제거
        self.tracks = [t for t in self.tracks if t.lost_count < self.max_lost]
        return self.tracks


# ==========================================
# 3. 유틸리티 함수
# ==========================================
def compute_physical_normal(depth_map, cx, cy, fx, fy, k=15):
    h, w = depth_map.shape
    x_min, x_max = max(0, cx - k), min(w - 1, cx + k)
    y_min, y_max = max(0, cy - k), min(h - 1, cy + k)

    z_center = float(depth_map[cy, cx]) / 1000.0
    if z_center <= 0:
        return np.array([0, 0, 1.0])

    dz = (float(depth_map[cy, x_max]) - float(depth_map[cy, x_min])) / 1000.0
    dx = (x_max - x_min) * z_center / fx
    dw = (float(depth_map[y_max, cx]) - float(depth_map[y_min, cx])) / 1000.0
    dy = (y_max - y_min) * z_center / fy

    normal = np.array([-dz / dx if dx != 0 else 0, -dw / dy if dy != 0 else 0, 1.0])
    norm = np.linalg.norm(normal)
    return normal / norm if norm > 0 else np.array([0, 0, 1.0])


# ==========================================
# 4. 메인 실행 루프
# ==========================================
if __name__ == "__main__":
    check_point_path = "segmentation_model/checkpoints/best_model.pth"
    target_img_size = (256, 448)
    device = torch.device("cuda")

    # 모델 초기화 (에폭 180 기반)
    model = FastMelonSegmenter(num_classes=1, num_prototypes=32, n_segment=1).to(device)
    model.load_state_dict(torch.load(check_point_path)["model"])
    model.eval()

    # 월드 모델 초기화
    world_model = WorldModelManager(dist_threshold=0.15, max_lost=50)

    color_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(target_img_size, interpolation=v2.InterpolationMode.BILINEAR),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    camera = KinectCamera()
    camera.start()
    K, D = camera.K, camera.D
    fx, fy = K[0, 0], K[1, 1]

    # Camera to World Matrix (m 단위)
    T_world = np.array(
        [
            [-0.99995885, -0.00566316, 0.00708717, -0.13613424],
            [-0.0085511, 0.84927882, -0.52787533, 1.26736509],
            [-0.00302954, -0.52791421, -0.8492923, 0.62092793],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        color, depth, _ = frame
        h, w = color.shape[:2]

        # 1. AI 추론 및 후처리
        img_t = color_transforms(color).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, boxes, coeffs, protos = model(img_t)

        scores = logits.sigmoid()[0, :, 0]
        mask_idx = scores > 0.4  # 낮은 임계값으로 인식 유지력 강화

        current_frame_dets = []
        if mask_idx.any():
            v_scores, v_boxes, v_coeffs = scores[mask_idx], boxes[0, mask_idx], coeffs[0, mask_idx]
            keep = nms(v_boxes, v_scores, 0.3)

            # 마스크 조립 (한 번에 수행)
            all_masks = torch.einsum("nc,chw->nhw", v_coeffs[keep], protos[0]).sigmoid()
            all_masks = F.interpolate(all_masks.unsqueeze(1), size=(h, w), mode="bilinear").squeeze(
                1
            )
            binary_masks = (all_masks > 0.5).cpu().numpy().astype(np.uint8)

            for i, idx in enumerate(keep):
                M = cv2.moments(binary_masks[i])
                if M["m00"] == 0:
                    continue
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

                z_m = float(depth[cy, cx]) / 1000.0
                if z_m <= 0:
                    continue

                p_n = cv2.undistortPoints(np.array([[[cx, cy]]], dtype=np.float32), K, D)
                un, vn = p_n[0][0]
                pw = (T_world @ np.array([un * z_m, vn * z_m, z_m, 1.0]))[:3]
                norm = compute_physical_normal(depth, cx, cy, fx, fy, k=15)

                current_frame_dets.append((pw, norm, (cx, cy), binary_masks[i]))

        # 2. 월드 모델 업데이트 (기억 매칭)
        active_tracks = world_model.update(current_frame_dets)

        # 3. 시각화 (깜빡임 없는 렌더링)
        vis_img = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        for track in active_tracks:
            avg_pos, avg_norm = track.get_stable_state()
            cx, cy = track.last_pixel_pos
            mask = track.last_mask

            # 🔥 LOST 상태에 따라 색상과 투명도 조절
            is_lost = track.lost_count > 0
            color_val = (0, 165, 255) if is_lost else (0, 255, 0)  # LOST: 오렌지, Tracking: 녹색
            alpha = 0.2 if is_lost else 0.4

            # 마스크 렌더링 (가려져도 마지막 마스크를 그림)
            vis_img[mask == 1] = (
                vis_img[mask == 1] * (1 - alpha) + np.array(color_val) * alpha
            ).astype(np.uint8)

            # UI 표시
            cv2.circle(vis_img, (cx, cy), 5, (255, 255, 255), -1)
            label = f"ID:{track.id} {'[LOST]' if is_lost else ''}"
            cv2.putText(
                vis_img, label, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_val, 2
            )

            # 노멀 벡터 화살표
            end_p = (int(cx + avg_norm[0] * 60), int(cy + avg_norm[1] * 60))
            cv2.arrowedLine(vis_img, (cx, cy), end_p, (255, 50, 0), 2)

        cv2.imshow("Zero-Flicker Harvest System", vis_img)
        if cv2.waitKey(1) == ord("q"):
            break

    camera.stop()
    cv2.destroyAllWindows()
