import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchinfo import summary
import math

# 1. 고정된 CoordConv 그리드를 생성하는 헬퍼 함수 (메모리 효율화)
def get_coord_maps(h, w, device):
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing='ij'
    )
    grid_x = (grid_x + 0.5) / w
    grid_y = (grid_y + 0.5) / h
    return torch.stack([grid_x, grid_y], dim=0) # [2, H, W]

class BackboneWithFPN(nn.Module):
    def __init__(self, fpn_out_channels=256):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        
        # Feature Extraction Layers
        self.layer1 = resnet.layer1 # 1/4 해상도
        self.layer2 = resnet.layer2 # 1/8 
        self.layer3 = resnet.layer3 # 1/16
        self.layer4 = resnet.layer4 # 1/32

        # FPN 1x1 Conv (채널 맞추기)
        self.lat_c5 = nn.Conv2d(2048, fpn_out_channels, kernel_size=1)
        self.lat_c4 = nn.Conv2d(1024, fpn_out_channels, kernel_size=1)
        self.lat_c3 = nn.Conv2d(512,  fpn_out_channels, kernel_size=1)

        # FPN Smooth Conv (앨리어싱 제거)
        self.smooth_p4 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        self.smooth_p3 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Bottom-up
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down (FPN)
        p5 = self.lat_c5(c5)
        p4 = self.lat_c4(c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p4 = self.smooth_p4(p4)
        p3 = self.lat_c3(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        p3 = self.smooth_p3(p3)

        return p3 # 고해상도 특징만 Protonet으로 보냄

# 3. Protonet: 베이스 마스크 생성 (기존 코드 유지)
class Protonet(nn.Module):
    def __init__(self, in_channels=256, num_prototypes=32):
        super().__init__()
        # CoordConv 채널 (+2)
        self.conv1 = nn.Conv2d(in_channels + 2, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_out = nn.Conv2d(256, num_prototypes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, p3):
        b, c, h, w = p3.shape
        
        coords = get_coord_maps(h, w, p3.device).unsqueeze(0).expand(b, 2, h, w)
        
        p3_with_coords = torch.cat([p3, coords], dim=1)
        
        # 프로토타입 생성
        x = self.relu(self.conv1(p3_with_coords))
        x = self.relu(self.conv2(x))
        x = self.upsample(x) # 1/8 -> 1/4 해상도로 업샘플링
        prototypes = self.relu(self.conv_out(x)) # [B, 32, 1/4H, 1/4W]
        
        return prototypes

# 4. 🔥 핵심 변경: Query 기반 Prediction Head
class QueryPredictionHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=1, num_prototypes=32, num_queries=100):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes

        # 1. 학습 가능한 Object Queries (N명의 수사관)
        self.query_embed = nn.Embedding(num_queries, in_channels) # 100개의 학습 가능한 가중치

        # 2. 가벼운 Transformer Decoder (CNN 특징과 쿼리 융합)
        # 추론 속도를 위해 레이어를 3개로 제한 (일반적인 DETR은 6개)
        decoder_layer = nn.TransformerDecoderLayer(d_model=in_channels, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)

        # 3. 최종 예측 FFNs (MLP)
        # 클래스 예측 (배경 포함: num_classes + 1)
        self.class_head = nn.Linear(in_channels, num_classes + 1)
        # 바운딩 박스 예측 (cxcywh)
        self.box_head = nn.Linear(in_channels, 4)
        # 마스크 계수 예측
        self.coeff_head = nn.Linear(in_channels, num_prototypes)

        # 배경 붕괴 방지를 위한 편향 초기화
        prior_prob = 0.01
        bias_val = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_head.bias, bias_val)

    def forward(self, p3):
        b, c, h, w = p3.shape
        
        # 1. CNN 특징맵을 Transformer 입력 형식으로 변환 [B, C, H, W] -> [B, HW, C]
        # 좌표 정보(Positional Encoding)를 특징에 더해줍니다.
        pos_embed = get_coord_maps(h, w, p3.device).flatten(1).permute(1, 0) # [HW, 2]
        # 간단한 선형 변환으로 좌표 벡터를 256차원으로 확장하여 특징에 더함
        feat_flatten = p3.flatten(2).permute(0, 2, 1) # [B, HW, C]
        
        # 2. Object Queries 준비 [B, N, C]
        query_embed = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        
        # 3. Transformer Decoder 통과 (크로스 어텐션으로 이미지 정보 수집)
        # tgt: 쿼리, memory: 이미지 특징
        hs = self.decoder(query_embed, feat_flatten) # [B, N, C]

        # 4. 각 쿼리로부터 최종 예측 디코딩
        pred_logits = self.class_head(hs) # [B, N, num_classes + 1]
        
        # 박스 예측 (cxcywh, sigmoid로 0~1 정규화)
        pred_boxes = self.box_head(hs).sigmoid() # [B, N, 4]
        
        # 마스크 계수 예측 (tanh로 -1~1 범위)
        pred_coeffs = self.coeff_head(hs).tanh() # [B, N, 32]

        return pred_logits, pred_boxes, pred_coeffs

# 5. 최종 로봇 제어용 실시간 세그멘테이션 모델
class RealTimeMelonSegmenter(nn.Module):
    def __init__(self, num_classes=1, num_prototypes=32, num_queries=100):
        super().__init__()
        # 특징 추출기
        self.backbone = BackboneWithFPN(fpn_out_channels=256)
        # 마스크 베이스 생성기
        self.protonet = Protonet(in_channels=256, num_prototypes=num_prototypes)
        # 🔥 변경된 Query 기반 헤드
        self.head = QueryPredictionHead(in_channels=256, num_classes=num_classes, 
                                        num_prototypes=num_prototypes, num_queries=num_queries)

    def forward(self, images):
        # 1. Backbone 통과 (FPN의 p3 특징만 사용)
        p3 = self.backbone(images) # [B, 256, 1/8H, 1/8W]
        
        # 2. 프로토타입 마스크 생성
        prototypes = self.protonet(p3) # [B, 32, 1/4H, 1/4W]
        
        # 3. 🔥 100개의 고정된 쿼리로 객체 예측 (8400개 아님!)
        # 이제 출력 크기는 [B, 100, X] 입니다.
        pred_logits, pred_boxes, pred_coeffs = self.head(p3)
        
        return pred_logits, pred_boxes, pred_coeffs, prototypes

# --- 작동 테스트 및 구조 확인 ---
if __name__ == "__main__":
    # 로봇 제어용 입력 크기 (배치 2, RGB 3, 512x896)
    dummy_input = torch.randn(2, 3, 512, 896) 

    model = RealTimeMelonSegmenter(num_classes=1, num_queries=100)
    
    # 모델 구조 요약 출력
    summary(model, input_size=(2, 3, 512, 896), device="cpu")
    
    # 추론 테스트
    logits, boxes, coeffs, protos = model(dummy_input)
    
    print("\n--- 모델 최종 출력 형태 (배치당 100개 고정) ---")
    print(f"Logits (클래스 확률) : {logits.shape}")   # [2, 100, 2] (배경 클래스 포함)
    print(f"Boxes (바운딩 박스)  : {boxes.shape}")    # [2, 100, 4]
    print(f"Coeffs (마스크 계수) : {coeffs.shape}")   # [2, 100, 32]
    print(f"Prototypes (마스크 베이스): {protos.shape}") # [2, 32, 128, 224] (1/4 해상도)