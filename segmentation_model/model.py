import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class TemporalShift(nn.Module):
    def __init__(self, n_segment=10, n_div=8):
        super().__init__()
        self.n_segment = n_segment
        self.n_div = n_div

    def forward(self, x):
        bt, c, h, w = x.size()
        t = self.n_segment
        b = bt // t

        x = x.view(b, t, c, h, w).contiguous()
        fold = c // self.n_div

        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]
        out[:, :, 2 * fold :] = x[:, :, 2 * fold :]

        return out.view(bt, c, h, w).contiguous()


def get_out_channels(resnet, input_size=(1, 3, 224, 224)):
    resnet.eval()
    x = torch.randn(*input_size)
    stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

    with torch.no_grad():
        c1 = resnet.layer1(stem(x))
        c2 = resnet.layer2(c1)
        c3 = resnet.layer3(c2)
        c4 = resnet.layer4(c3)

    return [c.shape[1] for c in [c1, c2, c3, c4]]


class BackboneWithFPN(nn.Module):
    def __init__(self, fpn_out_channels=256, n_segment=10):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        chennals = get_out_channels(resnet)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

        self.layer1 = nn.Sequential(TemporalShift(n_segment), resnet.layer1)
        self.layer2 = nn.Sequential(TemporalShift(n_segment), resnet.layer2)
        self.layer3 = nn.Sequential(TemporalShift(n_segment), resnet.layer3)
        self.layer4 = nn.Sequential(TemporalShift(n_segment), resnet.layer4)

        self.lat_c5 = nn.Conv2d(chennals[3], fpn_out_channels, kernel_size=1)
        self.lat_c4 = nn.Conv2d(chennals[2], fpn_out_channels, kernel_size=1)
        self.lat_c3 = nn.Conv2d(chennals[1], fpn_out_channels, kernel_size=1)

        self.smooth_p4 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        self.smooth_p3 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # 2. FPN을 통한 특징 융합 (Top-down)
        # P5: 가장 깊고 의미론적(Semantic) 정보가 풍부한 1/32 해상도
        p5 = self.lat_c5(c5)

        # P4: P5를 2배 키우고, C4의 디테일을 더함 (1/16 해상도)
        p4 = self.lat_c4(c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p4 = self.smooth_p4(p4)

        # P3: P4를 2배 키우고, C3의 디테일을 더함 (1/8 해상도)
        p3 = self.lat_c3(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        p3 = self.smooth_p3(p3)

        # 세그멘테이션을 위해 3가지 스케일의 Feature Map을 반환합니다.
        # - p3 (1/8): 고해상도 정보가 살아있어 '프로토타입 마스크'를 만들 때 사용하기 좋음
        # - p4, p5: 객체의 '바운딩 박스'나 '클래스'를 예측할 때 섞어서 사용함
        return p3, p4, p5


class Protonet(nn.Module):
    def __init__(self, in_channels=256, num_prototypes=32):
        super().__init__()
        coordconv_channels = 2
        self.conv1 = nn.Conv2d(in_channels + coordconv_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv_out = nn.Conv2d(256, num_prototypes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, p3):
        b, c, h, w = p3.shape

        # 1. P3 해상도에 맞는 X, Y 절대 좌표 그리드 생성 (0~1 정규화)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, dtype=p3.dtype, device=p3.device),
            torch.arange(w, dtype=p3.dtype, device=p3.device),
            indexing="ij",
        )
        grid_x = (grid_x + 0.5) / w
        grid_y = (grid_y + 0.5) / h

        # [B, 2, H, W] 형태로 묶기
        grids = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(b, 2, h, w)

        # 2. 기존 특징 맵(P3)에 좌표 데이터 결합 (CoordConv)
        p3_with_coords = torch.cat([p3, grids], dim=1)

        # 3. 좌표가 결합된 텐서로 프로토타입 마스크 생성
        x = self.relu(self.conv1(p3_with_coords))
        x = self.relu(self.conv2(x))
        x = self.upsample(x)
        prototypes = self.relu(self.conv_out(x))

        return prototypes


class PredictionHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=1, num_prototypes=32):
        super().__init__()
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes

        # 🔥 핵심 1: 기존 in_channels(256)에 X, Y 좌표 채널 2개가 추가되므로 +2를 해줍니다.
        in_channels_with_coords = in_channels + 2

        # 클래스 예측 브랜치
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels_with_coords, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=3, padding=1),
        )

        # 바운딩 박스 예측 브랜치
        self.box_conv = nn.Sequential(
            nn.Conv2d(in_channels_with_coords, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4, kernel_size=3, padding=1),
        )

        # 마스크 계수 예측 브랜치
        self.coeff_conv = nn.Sequential(
            nn.Conv2d(in_channels_with_coords, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_prototypes, kernel_size=3, padding=1),
        )

        # 모델 기절(Background Collapse) 방지를 위한 편향(Bias) 초기화
        import math

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_conv[2].bias, bias_value)

    def forward(self, features):
        all_cls_logits = []
        all_boxes = []
        all_coeffs = []

        for x in features:  #
            b, c, h, w = x.shape

            # 1. 현재 Feature Map의 픽셀별 절대 좌표 그리드 생성 (0~1 정규화)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, dtype=torch.float32, device=x.device),
                torch.arange(w, dtype=torch.float32, device=x.device),
                indexing="ij",
            )
            # 각 픽셀(셀)의 정중앙 좌표로 맞춤 (+0.5)
            grid_x = (grid_x + 0.5) / w
            grid_y = (grid_y + 0.5) / h

            # [B, 2, H, W] 형태로 확장
            grids = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(b, 2, h, w)

            # CoordConv: 기존 특징맵(256)에 GPS 좌표(2) 결합 -> 258 채널
            x_with_coords = torch.cat([x, grids], dim=1)

            # 2. 클래스와 마스크 계수는 그대로 예측
            cls_logits = self.cls_conv(x_with_coords).flatten(2).permute(0, 2, 1)
            coeffs = self.coeff_conv(x_with_coords).flatten(2).permute(0, 2, 1)

            # =========================================================
            # 🔥 핵심 해결책: Anchor-Free 기반 바운딩 박스 디코딩
            # =========================================================
            box_preds = self.box_conv(x_with_coords)  # [B, 4, H, W]

            # 중심점(cx, cy)은 예측값이 무한정 뻗어 나가지 못하도록 tanh를 씌우고,
            # 현재 자신의 그리드(grids) 위치에서 픽셀 1~2칸 정도의 미세 조정(offset)만 하도록 강제합니다.
            dx = box_preds[:, 0, :, :].tanh() / w
            dy = box_preds[:, 1, :, :].tanh() / h

            # 최종 중심점 = 내 위치(GPS) + 미세 조정(Offset)
            cx = grids[:, 0, :, :] + dx
            cy = grids[:, 1, :, :] + dy

            # 너비(w)와 높이(h)는 0~1 사이의 절대 크기로 예측
            bw = box_preds[:, 2, :, :].sigmoid()
            bh = box_preds[:, 3, :, :].sigmoid()

            # 4개의 좌표를 하나로 합침
            boxes = torch.stack([cx, cy, bw, bh], dim=1).flatten(2).permute(0, 2, 1)

            all_cls_logits.append(cls_logits)
            all_boxes.append(boxes)
            all_coeffs.append(coeffs)

        pred_logits = torch.cat(all_cls_logits, dim=1)
        pred_boxes = torch.cat(all_boxes, dim=1)
        pred_coeffs = torch.cat(all_coeffs, dim=1)

        # pred_boxes는 위에서 이미 처리했으므로 sigmoid를 다시 씌우지 않습니다!
        pred_coeffs = torch.tanh(pred_coeffs)

        return pred_logits, pred_boxes, pred_coeffs


class FastMelonSegmenter(nn.Module):
    def __init__(self, num_classes=1, num_prototypes=32, n_segment=10):
        super().__init__()
        # 1. 특징 추출기 (ResNet50 + FPN)
        self.backbone = BackboneWithFPN(fpn_out_channels=256, n_segment=n_segment)

        # 2. 32장의 베이스 마스크 생성기
        self.protonet = Protonet(in_channels=256, num_prototypes=num_prototypes)

        # 3. 객체 탐지 및 계수 생성기
        self.head = PredictionHead(
            in_channels=256, num_classes=num_classes, num_prototypes=num_prototypes
        )

    def forward(self, images):
        if images.dim() == 5:
            b, t, c, h, w = images.shape
            # TSM 처리를 위해 [B*T, C, H, W] 형태로 Flatten
            images = images.view(b * t, c, h, w)
        else:
            # 단일 이미지 입력 시 예외 처리 (T=1로 간주)
            bt, c, h, w = images.shape
            b, t = bt, 1

        # 1. Backbone 통과
        p3, p4, p5 = self.backbone(images)

        # 2. 고해상도 특징맵(P3)으로 프로토타입 생성
        prototypes = self.protonet(p3)

        # 3. 모든 피라미드 특징맵으로 8400개의 픽셀별 객체 예측
        pred_logits, pred_boxes, pred_coeffs = self.head([p3, p4, p5])

        # 최종 모델 출력 완성!
        # 학습 시: 이 출력값들과 정답(Ground Truth)을 헝가리안 매칭하여 Loss 계산
        # 추론 시: threshold 넘는 것만 골라내어 einsum으로 마스크 조립
        return pred_logits, pred_boxes, pred_coeffs, prototypes


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 현재 사용 중인 장치: {device} ---")

    # 설정: 배치 2, 프레임 5, RGB 3, 256x448 해상도
    batch_size = 2
    num_frames = 5
    dummy_video = torch.randn(batch_size, num_frames, 3, 256, 448).to(device)

    # 모델 생성 (프레임 수 전달)
    model = FastMelonSegmenter(n_segment=num_frames).to(device)

    # 추론
    logits, boxes, coeffs, protos = model(dummy_video)

    print("--- TSM 적용 모델 최종 출력 형태 ---")
    print(f"입력 데이터 형태     : {dummy_video.shape} [B, T, C, H, W]")
    # 256x448 입력 시 앵커 개수는 2352개 (32x56 + 16x28 + 8x14)
    print(f"Logits (클래스 확률) : {logits.shape}  [B*T, 2352, 1]")
    print(f"Boxes (바운딩 박스)  : {boxes.shape}  [B*T, 2352, 4]")
    print(f"Coeffs (마스크 계수) : {coeffs.shape}  [B*T, 2352, 32]")
    # 프로토타입은 P3(32x56)의 2배 해상도인 64x112
    print(f"Prototypes (특수물감): {protos.shape} [B*T, 32, 64, 112]")

    # 요약 출력
    from torchinfo import summary

    summary(model, input_size=(batch_size, num_frames, 3, 256, 448), device=device)
