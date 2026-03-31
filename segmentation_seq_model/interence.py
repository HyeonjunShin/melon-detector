import os

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import v2

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from model import FastMelonSegmenter


class Estimator:
    def __init__(
        self,
        model,
        check_point,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.check_point = torch.load(check_point, map_location=self.device)
        self.model.load_state_dict(self.check_point)
        self.model.eval()

    @torch.no_grad()
    def get_prediction(self, input_tensor):
        logits, boxes, coeffs, prototypes = self.model(input_tensor)

        # 확률값(Sigmoid) 계산
        scores = logits.sigmoid()[0, :, 0]  # [8400] 개의 예측 점수

        print(f"🧐 모델이 예측한 가장 높은 확률(Max Score): {scores.max().item():.4f}")
        conf_threshold = 0.3

        # 💥 핵심: NMS 없이 단순 Threshold 필터링으로 진짜 멜론만 쏙 골라냅니다!
        keep_idx = scores > conf_threshold
        valid_scores = scores[keep_idx]
        valid_coeffs = coeffs[0, keep_idx]  # [N, 32]

        print(f"✨ 8400개의 예측 중 {len(valid_scores)}개의 멜론이 탐지되었습니다!")

        if len(valid_scores) == 0:
            print("탐지된 멜론이 없습니다.")
            return

        # ==========================================
        # 4. 초고속 마스크 조립 및 원본 크기 복원
        # ==========================================
        # 살아남은 N개의 객체에 대해서만 행렬 곱(einsum)을 수행하여 마스크 생성 [N, 160, 160]
        pred_masks = torch.einsum("nc,chw->nhw", valid_coeffs, prototypes[0])
        pred_masks = pred_masks.sigmoid()

        # 160x160 마스크를 원본 이미지 크기(640x640)로 부드럽게(bilinear) 확대
        pred_masks = F.interpolate(
            pred_masks.unsqueeze(1),
            size=(1080, 1920),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # 0.5를 기준으로 이진 마스크(Binary Mask)로 변환 후 NumPy 배열로 변경
        binary_masks = (pred_masks > 0.5).cpu().numpy().astype(np.uint8)

        return (valid_scores.cpu().numpy(), binary_masks)
        # ==========================================
        # 5. 시각화 및 파지점(Picking Point) 계산
        # ==========================================
        vis_img = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
        vis_img = vis_img.astype(np.uint8)
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        vis_img = vis_img.copy()

        for i in range(len(valid_scores)):
            mask = binary_masks[i]
            score = valid_scores[i]

            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            color = (int(color[0]), int(color[1]), int(color[2]))

            vis_img[mask == 1] = vis_img[mask == 1] * 0.5 + np.array(color) * 0.5

            M = cv2.moments(mask)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                cv2.circle(vis_img, (cX, cY), 6, (255, 0, 0), -1)
                cv2.circle(vis_img, (cX, cY), 12, (255, 255, 255), 2)
                cv2.putText(
                    vis_img,
                    f"Melon {score:.2f}",
                    (cX + 15, cY - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        # 화면에 창 띄우기!
        plt.figure(figsize=(10, 10))
        plt.imshow(vis_img)
        plt.title("Melon Picking Point Detection (NMS-Free)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


@torch.no_grad()  # 추론 시에는 기울기 계산을 끕니다 (속도 및 메모리 최적화)
def run_inference_and_find_picking_points(image_path, model_path, conf_threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 1. 모델 초기화 및 학습된 가중치 불러오기
    model = FastMelonSegmenter(num_classes=1, num_prototypes=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 평가 모드로 전환 (Dropout, BatchNorm 등 고정)

    # 2. 이미지 로드 및 전처리 (학습할 때와 동일하게 640x640으로 맞춤)
    orig_image = cv2.imread(image_path)
    if orig_image is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

    # 시각화를 위해 640x640으로 리사이즈된 이미지를 베이스로 사용합니다.
    img_resized = cv2.resize(orig_image, (640, 640))
    img_tensor = (
        TF.to_tensor(img_resized).unsqueeze(0).to(device)
    )  # Shape: [1, 3, 640, 640]

    # ==========================================
    # 3. 모델 추론 (Forward Pass)
    # ==========================================
    logits, boxes, coeffs, prototypes = model(img_tensor)

    # 확률값(Sigmoid) 계산
    scores = logits.sigmoid()[0, :, 0]  # [8400] 개의 예측 점수

    print(f"🧐 모델이 예측한 가장 높은 확률(Max Score): {scores.max().item():.4f}")
    conf_threshold = 0.3

    # 💥 핵심: NMS 없이 단순 Threshold 필터링으로 진짜 멜론만 쏙 골라냅니다!
    keep_idx = scores > conf_threshold
    valid_scores = scores[keep_idx]
    valid_coeffs = coeffs[0, keep_idx]  # [N, 32]

    print(f"✨ 8400개의 예측 중 {len(valid_scores)}개의 멜론이 탐지되었습니다!")

    if len(valid_scores) == 0:
        print("탐지된 멜론이 없습니다.")
        return

    # ==========================================
    # 4. 초고속 마스크 조립 및 원본 크기 복원
    # ==========================================
    # 살아남은 N개의 객체에 대해서만 행렬 곱(einsum)을 수행하여 마스크 생성 [N, 160, 160]
    pred_masks = torch.einsum("nc,chw->nhw", valid_coeffs, prototypes[0])
    pred_masks = pred_masks.sigmoid()

    # 160x160 마스크를 원본 이미지 크기(640x640)로 부드럽게(bilinear) 확대
    pred_masks = F.interpolate(
        pred_masks.unsqueeze(1), size=(1080, 1920), mode="bilinear", align_corners=False
    ).squeeze(1)

    # 0.5를 기준으로 이진 마스크(Binary Mask)로 변환 후 NumPy 배열로 변경
    binary_masks = (pred_masks > 0.5).cpu().numpy().astype(np.uint8)

    # ==========================================
    # 5. 시각화 및 파지점(Picking Point) 계산
    # ==========================================
    vis_img = img_resized.copy()

    for i in range(len(valid_scores)):
        mask = binary_masks[i]
        score = valid_scores[i]

        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        color = (int(color[0]), int(color[1]), int(color[2]))

        vis_img[mask == 1] = vis_img[mask == 1] * 0.5 + np.array(color) * 0.5

        M = cv2.moments(mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv2.circle(vis_img, (cX, cY), 6, (255, 0, 0), -1)
            cv2.circle(vis_img, (cX, cY), 12, (255, 255, 255), 2)
            cv2.putText(
                vis_img,
                f"Melon {score:.2f}",
                (cX + 15, cY - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

    # 화면에 창 띄우기!
    plt.figure(figsize=(10, 10))
    plt.imshow(vis_img)
    plt.title("Melon Picking Point Detection (NMS-Free)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 테스트할 컨베이어 벨트 이미지 경로와 학습된 가중치 경로를 입력하세요.
    dir_path = "/home/hyeonjun/1_data/dummy_melon_data/20260320_141216"
    check_point_path = "melon_segmenter_epoch_100.pth"
    origianl_img_size = (1080, 1920)
    target_img_size = (256, 448)

    estimator = Estimator(
        FastMelonSegmenter(num_classes=1, num_prototypes=32),
        check_point_path,
    )

    color_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(target_img_size, interpolation=v2.InterpolationMode.BILINEAR),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    data_files = os.listdir(dir_path)
    for data_file in data_files:
        data = np.load(os.path.join(dir_path, data_file))
        timestamp = data["timestamp"]
        color = data["color"]
        depth = data["depth"]

        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        input_tensor = color_transforms(color).unsqueeze(0).to(estimator.device)
        ret = estimator.get_prediction(input_tensor)

        if ret is not None:
            valid_scores, binary_masks = ret

            vis_img = color.copy()

            for i in range(len(valid_scores)):
                mask = binary_masks[i]
                score = valid_scores[i]

                color = np.random.randint(0, 255, (3,), dtype=np.uint8)
                color = (int(color[0]), int(color[1]), int(color[2]))

                vis_img[mask == 1] = vis_img[mask == 1] * 0.5 + np.array(color) * 0.5

                M = cv2.moments(mask)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    cv2.circle(vis_img, (cX, cY), 6, (255, 0, 0), -1)
                    cv2.circle(vis_img, (cX, cY), 12, (255, 255, 255), 2)
                    cv2.putText(
                        vis_img,
                        f"Melon {score:.2f}",
                        (cX + 15, cY - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

            # 화면에 창 띄우기!
            plt.figure(figsize=(10, 10))
            plt.imshow(vis_img)
            plt.title("Melon Picking Point Detection (NMS-Free)")
            plt.axis("off")
            plt.tight_layout()
            plt.show()

# run_inference_and_find_picking_points(TEST_IMAGE_PATH, MODEL_WEIGHTS_PATH, conf_threshold=0.5)
