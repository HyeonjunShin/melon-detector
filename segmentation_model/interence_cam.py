import cv2
import numpy as np
import torch
import torch.nn.functional as F

# 가상의 임포트 (사용자 환경에 맞게 유지)
from camera_devices.kinect_wrapper import KinectCamera
from model import FastMelonSegmenter
from torchvision.ops import nms  # 중복 박스 제거용
from torchvision.transforms import v2


# ==========================================
# 1. 트래킹 및 지터링 방지 로직
# ==========================================
class TrackedObject:
    def __init__(self, id, center, mask, score, alpha=0.2):
        self.id = id
        self.center = np.array(center, dtype=float)
        self.mask = mask
        self.score = score
        self.alpha = alpha  # 낮을수록 더 부드러움 (0.1 ~ 0.3 권장)
        self.missing_frames = 0
        self.hit_stretch = 0  # 연속 탐지 횟수

    def update(self, new_center, new_mask, new_score):
        # 지수 이동 평균(EMA)으로 중심점 좌표의 떨림(Jitter) 잡기
        self.center = self.alpha * np.array(new_center) + (1 - self.alpha) * self.center
        self.mask = new_mask
        self.score = new_score
        self.missing_frames = 0
        self.hit_stretch += 1


class MelonTracker:
    def __init__(self, dist_threshold=100):
        self.objects = []
        self.next_id = 0
        self.dist_threshold = dist_threshold  # 동일 객체로 판단할 최대 거리(픽셀)

    def step(self, detections):
        if not detections:
            for obj in self.objects:
                obj.missing_frames += 1
            self.objects = [obj for obj in self.objects if obj.missing_frames < 5]
            return self.objects

        current_centers = [d[0] for d in detections]
        current_masks = [d[1] for d in detections]
        current_scores = [d[2] for d in detections]

        matched_indices = set()

        # 1. 기존 추적 대상과 현재 탐지 결과 매칭
        for obj in self.objects:
            if not current_centers:
                break

            # 유클리드 거리 기반 최단 거리 찾기
            distances = [np.linalg.norm(obj.center - np.array(c)) for c in current_centers]
            if not distances:
                continue

            min_idx = np.argmin(distances)

            if distances[min_idx] < self.dist_threshold and min_idx not in matched_indices:
                obj.update(
                    current_centers[min_idx],
                    current_masks[min_idx],
                    current_scores[min_idx],
                )
                matched_indices.add(min_idx)
            else:
                obj.missing_frames += 1

        # 2. 매칭되지 않은 새로운 탐지 결과에 새 ID 부여
        for i, (center, mask, score) in enumerate(
            zip(current_centers, current_masks, current_scores)
        ):
            if i not in matched_indices:
                # 이미 추적 중인 객체와 너무 가까우면 중복으로 간주하고 버림
                is_duplicate = any(
                    np.linalg.norm(np.array(center) - o.center) < 50 for o in self.objects
                )
                if not is_duplicate:
                    self.objects.append(TrackedObject(self.next_id, center, mask, score))
                    self.next_id += 1

        # 3. 화면에서 사라진 지 오래된 객체 제거
        self.objects = [obj for obj in self.objects if obj.missing_frames < 10]
        return self.objects


# ==========================================
# 2. 모델 추론 클래스 (NMS 포함)
# ==========================================
class Estimator:
    def __init__(
        self,
        model,
        check_point,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.device = device
        self.model = model.to(self.device)
        self.model.load_state_dict(torch.load(check_point, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def get_prediction(self, input_tensor, orig_size=(1080, 1920)):
        # 모델 포워드 패스
        logits, boxes, coeffs, prototypes = self.model(input_tensor)

        # 1. 스코어 필터링
        scores = logits.sigmoid()[0, :, 0]
        conf_threshold = 0.4
        mask_idx = scores > conf_threshold

        if not mask_idx.any():
            return None

        valid_scores = scores[mask_idx]
        valid_boxes = boxes[0, mask_idx]
        valid_coeffs = coeffs[0, mask_idx]

        # 2. [핵심] 중복 탐지 제거를 위한 NMS 적용
        # 같은 참외에 여러 박스가 생기는 것을 방지 (IoU 0.3 이상 겹치면 낮은 점수 삭제)
        keep_nms = nms(valid_boxes, valid_scores, iou_threshold=0.3)

        valid_scores = valid_scores[keep_nms]
        valid_coeffs = valid_coeffs[keep_nms]
        # valid_boxes = valid_boxes[keep_nms] # 필요한 경우 박스도 리턴 가능

        # 3. 마스크 조립 및 리사이즈
        pred_masks = torch.einsum("nc,chw->nhw", valid_coeffs, prototypes[0]).sigmoid()
        pred_masks = F.interpolate(
            pred_masks.unsqueeze(1),
            size=orig_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        binary_masks = (pred_masks > 0.5).cpu().numpy().astype(np.uint8)

        # 결과 정리 (중심점 계산 포함)
        results = []
        for i in range(len(valid_scores)):
            M = cv2.moments(binary_masks[i])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                results.append(((cX, cY), binary_masks[i], valid_scores[i].item()))

        return results


# ==========================================
# 3. 메인 실행 루프
# ==========================================
if __name__ == "__main__":
    check_point_path = "./melon_segmenter_epoch_200.pth"
    target_img_size = (256, 448)  # 모델 학습 규격에 맞게 수정

    # 모델 및 트래커 초기화
    estimator = Estimator(FastMelonSegmenter(num_classes=1, num_prototypes=32), check_point_path)
    tracker = MelonTracker(dist_threshold=120)  # 컨베이어 속도에 따라 조절

    color_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(target_img_size, interpolation=v2.InterpolationMode.BILINEAR),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    camera = KinectCamera()
    camera.start()

    K = camera.K
    D = camera.D
    T = np.array(
        [
            [-0.99995885, -0.00566316, 0.00708717, -0.13613424],
            [-0.0085511, 0.84927882, -0.52787533, 1.26736509],
            [-0.00302954, -0.52791421, -0.8492923, 0.62092793],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    cv2.namedWindow("Stable Melon Tracking")
    save_count = 0

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        color, depth, timestamp = frame
        h, w = color.shape[:2]

        # 1. 딥러닝 추론 (NMS로 중복 마스크 1차 제거)
        input_tensor = color_transforms(color).unsqueeze(0).to(estimator.device)
        detections = estimator.get_prediction(input_tensor, orig_size=(h, w))

        # 2. 트래커 실행 (프레임 간 ID 연결 및 좌표 스무딩)
        tracked_melons = tracker.step(detections if detections else [])

        # 3. 시각화 (RGB -> BGR 주의)
        vis_img = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        for obj in tracked_melons:
            # 부드럽게 계산된 중심점 좌표
            cZ = depth[int(obj.center[1]), int(obj.center[0])]
            cX, cY = int(obj.center[0]), int(obj.center[1])

            p_normal = cv2.undistortPoints(np.array([[[cX, cY]]], dtype=np.float32), K, D)
            u_n = p_normal[0][0][0]
            v_n = p_normal[0][0][1]
            X_c = u_n * cZ
            Y_c = v_n * cZ
            Z_c = cZ

            P_c = np.array([X_c, Y_c, Z_c, 1.0])
            P_w = T @ P_c
            print(
                cX,
                cY,
                cZ,
                X_c,
                Y_c,
                Z_c,
            )
            print(P_w)
            # ID별 고정 색상 부여
            np.random.seed(obj.id)
            color_val = np.random.randint(0, 255, (3,)).tolist()

            # 마스크 씌우기
            mask_indices = obj.mask == 1
            vis_img[mask_indices] = (
                vis_img[mask_indices] * 0.6 + np.array(color_val) * 0.4
            ).astype(np.uint8)

            # Picking Point 표시
            cv2.circle(vis_img, (cX, cY), 7, (0, 0, 255), -1)  # 빨간 점
            cv2.circle(vis_img, (cX, cY), 12, (255, 255, 255), 2)  # 외곽선

            # 정보 텍스트
            label = f"ID:{obj.id} ({obj.score:.2f})"
            cv2.putText(
                vis_img,
                label,
                (cX + 15, cY - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Stable Melon Tracking", vis_img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        if key == ord("s"):
            cv2.imwrite(f"tracked_melons_{save_count}.png", vis_img)
            save_count += 1

    camera.stop()
    cv2.destroyAllWindows()
