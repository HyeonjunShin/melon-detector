import cv2
import numpy as np
import torch
import torch.nn.functional as F
from camera_devices.kinect_wrapper import KinectCamera

# 사용자 정의 모듈
from model import FastMelonSegmenter
from torchvision.ops import nms
from torchvision.transforms import v2


class MelonTSMEngineV3:
    def __init__(self, model_path, device=None):
        self.device = (
            device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # 1. 모델 초기화 (4채널 입력)
        self.model = FastMelonSegmenter(num_classes=1, num_prototypes=32)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # TSM 버퍼 초기화
        self.reset_tsm()

        self.target_size = (256, 448)
        self.depth_min, self.depth_max = 0.2, 2.2

        # RGB 전처리
        self.rgb_transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(self.target_size, interpolation=v2.InterpolationMode.BILINEAR),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def reset_tsm(self):
        for m in self.model.modules():
            if hasattr(m, "reset_buffer"):
                m.reset_buffer()
        print("🔄 TSM Buffer Reset.")

    @torch.no_grad()
    def process_frame(self, frame_rgb, frame_depth, conf_threshold=0.9):
        h, w = frame_rgb.shape[:2]

        # --- [1] RGB 전처리 ---
        img_t = self.rgb_transforms(frame_rgb).to(self.device)

        # --- [2] Depth 전처리 ---
        depth_m = frame_depth.astype(np.float32) / 1000.0
        invalid_mask = (depth_m < self.depth_min) | (depth_m > self.depth_max)
        depth_m[invalid_mask] = 0.0

        dep_t = torch.from_numpy(depth_m).unsqueeze(0).unsqueeze(0)
        dep_t = F.interpolate(dep_t, size=self.target_size, mode="nearest").squeeze(0)

        # 정규화 (배경 1.0)
        final_dep = torch.zeros_like(dep_t)
        v_mask = dep_t > 0.0
        final_dep[v_mask] = (
            torch.clamp(
                (dep_t[v_mask] - self.depth_min) / (self.depth_max - self.depth_min), 0.0, 1.0
            )
            * 0.5
        )
        final_dep[~v_mask] = 1.0
        final_dep = final_dep.to(self.device)

        # --- [3] 모델 추론 (에러 해결 지점) ---
        input_combined = torch.cat([img_t, final_dep], dim=0).unsqueeze(0)

        # outputs[:4] 대신 전체를 받아오거나 정확히 5개를 지정해야 합니다.
        outputs = self.model(input_combined)

        # 모델의 return 순서가 [logits, boxes, coeffs, prototypes, normals]인 경우
        logits, boxes, coeffs, prototypes, pred_normals = outputs

        # --- [4] 인스턴스 세그멘테이션 후처리 ---
        scores = logits.sigmoid()[0, :, 0]
        mask_idx = scores > conf_threshold

        instances = []
        if mask_idx.any():
            v_scores, v_boxes, v_coeffs = scores[mask_idx], boxes[0, mask_idx], coeffs[0, mask_idx]
            keep = nms(v_boxes, v_scores, iou_threshold=0.1)

            # 마스크 조립
            p_masks = torch.einsum("nc,chw->nhw", v_coeffs[keep], prototypes[0]).sigmoid()
            p_masks = F.interpolate(p_masks.unsqueeze(1), size=(h, w), mode="bilinear").squeeze(1)
            binary_masks = (p_masks > 0.5).cpu().numpy().astype(np.uint8)

            for i in range(len(keep)):
                if np.sum(binary_masks[i]) < 1500:
                    continue
                M = cv2.moments(binary_masks[i])
                if M["m00"] != 0:
                    cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    instances.append(((cX, cY), binary_masks[i], v_scores[keep][i].item()))

        # --- [5] Normal Map 시각화 ---
        norm_map = pred_normals[0].permute(1, 2, 0).cpu().numpy()
        norm_vis = ((norm_map + 1.0) / 2.0 * 255).astype(np.uint8)
        norm_vis = cv2.resize(norm_vis, (w, h))

        return instances, norm_vis


def main():
    MODEL_PATH = "melon_tsm_epoch_100.pth"
    engine = MelonTSMEngineV3(MODEL_PATH)
    camera = KinectCamera()
    camera.start()

    cv2.namedWindow("Melon Detection")
    cv2.namedWindow("Normal Map")

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue
            color, depth, _ = frame

            detections, normal_map = engine.process_frame(color, depth, conf_threshold=0.9)

            vis_img = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            for (cX, cY), mask, score in detections:
                vis_img[mask == 1] = (
                    vis_img[mask == 1] * 0.7 + np.array([0, 255, 0]) * 0.3
                ).astype(np.uint8)
                cv2.circle(vis_img, (cX, cY), 5, (0, 0, 255), -1)

            cv2.imshow("Melon Detection", vis_img)
            cv2.imshow("Normal Map", cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
