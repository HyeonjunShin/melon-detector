import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.ops import masks_to_boxes
from torchvision.transforms import v2

# =================================================================
# 1. 모델 구성 요소 (Backbone, TSM, Head)
# =================================================================

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

class BackboneWithFPN(nn.Module):
    def __init__(self, fpn_out_channels=256, n_segment=10):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = nn.Sequential(TemporalShift(n_segment), resnet.layer1)
        self.layer2 = nn.Sequential(TemporalShift(n_segment), resnet.layer2)
        self.layer3 = nn.Sequential(TemporalShift(n_segment), resnet.layer3)
        self.layer4 = nn.Sequential(TemporalShift(n_segment), resnet.layer4)

        channels = [256, 512, 1024, 2048]
        self.lat_c5 = nn.Conv2d(channels[3], fpn_out_channels, 1)
        self.lat_c4 = nn.Conv2d(channels[2], fpn_out_channels, 1)
        self.lat_c3 = nn.Conv2d(channels[1], fpn_out_channels, 1)
        self.smooth = nn.Conv2d(fpn_out_channels, fpn_out_channels, 3, padding=1)

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x); c3 = self.layer2(c2); c4 = self.layer3(c3); c5 = self.layer4(c4)
        p5 = self.lat_c5(c5)
        p4 = self.lat_c4(c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p3 = self.lat_c3(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        return self.smooth(p3), self.smooth(p4), self.smooth(p5)

class PredictionHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=1, num_prototypes=32):
        super().__init__()
        # CoordConv를 위해 +2 채널
        self.cls_conv = nn.Sequential(nn.Conv2d(in_channels+2, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, num_classes, 3, padding=1))
        self.box_conv = nn.Sequential(nn.Conv2d(in_channels+2, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 4, 3, padding=1))
        self.coeff_conv = nn.Sequential(nn.Conv2d(in_channels+2, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, num_prototypes, 3, padding=1))
        
        prior_prob = 0.01
        nn.init.constant_(self.cls_conv[-1].bias, -math.log((1 - prior_prob) / prior_prob))

    def forward(self, features):
        all_cls, all_box, all_coeff = [], [], []
        for x in features:
            b, c, h, w = x.shape
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            grid = torch.stack([(grid_x+0.5)/w, (grid_y+0.5)/h], dim=0).to(x.device).unsqueeze(0).expand(b, 2, h, w)
            x_in = torch.cat([x, grid.float()], dim=1)
            
            cls = self.cls_conv(x_in).flatten(2).permute(0, 2, 1)
            coeff = torch.tanh(self.coeff_conv(x_in).flatten(2).permute(0, 2, 1))
            
            # Box decoding (Anchor-free)
            box_preds = self.box_conv(x_in)
            cx = grid[:, 0] + box_preds[:, 0].tanh() / w
            cy = grid[:, 1] + box_preds[:, 1].tanh() / h
            bw = box_preds[:, 2].sigmoid()
            bh = box_preds[:, 3].sigmoid()
            boxes = torch.stack([cx, cy, bw, bh], dim=1).flatten(2).permute(0, 2, 1)
            
            all_cls.append(cls); all_box.append(boxes); all_coeff.append(coeff)
        return torch.cat(all_cls, 1), torch.cat(all_box, 1), torch.cat(all_coeff, 1)

class FastMelonSegmenter(nn.Module):
    def __init__(self, num_classes=1, num_prototypes=32, n_segment=10):
        super().__init__()
        self.backbone = BackboneWithFPN(n_segment=n_segment)
        self.proto_net = nn.Sequential(nn.Conv2d(258, 256, 3, padding=1), nn.ReLU(), nn.Upsample(scale_factor=2), nn.Conv2d(256, num_prototypes, 1), nn.ReLU())
        self.head = PredictionHead(num_classes=num_classes, num_prototypes=num_prototypes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b*t, c, h, w)
        p3, p4, p5 = self.backbone(x)
        
        # Prototype 생성용 CoordConv
        ph, pw = p3.shape[-2:]
        grid_y, grid_x = torch.meshgrid(torch.arange(ph), torch.arange(pw), indexing="ij")
        grid = torch.stack([(grid_x+0.5)/pw, (grid_y+0.5)/ph], dim=0).to(p3.device).unsqueeze(0).expand(b*t, 2, ph, pw)
        protos = self.proto_net(torch.cat([p3, grid.float()], dim=1))
        
        cls, box, coeff = self.head([p3, p4, p5])
        return cls, box, coeff, protos

# =================================================================
# 2. 데이터셋 및 매칭 알고리즘
# =================================================================

class MelonDataset(Dataset):
    def __init__(self, root_dir, num_frames, num_seq=10, target_size=(256, 448)):
        self.root_dir = root_dir
        self.num_seq = num_seq
        self.num_data = num_frames // num_seq
        self.target_size = target_size
        self.color_trans = v2.Compose([v2.Resize(target_size), v2.ToDtype(torch.float32, scale=True)])
        self.mask_trans = v2.Compose([v2.Resize(target_size, interpolation=0), v2.ToDtype(torch.int64, scale=False)])

    def __len__(self): return self.num_data

    def __getitem__(self, index):
        start = index * self.num_seq
        c_list, m_list = [], []
        for i in range(start, start + self.num_seq):
            c_list.append(read_image(f"{self.root_dir}/rgb_{i:05d}.png"))
            m_list.append(read_image(f"{self.root_dir}/instance_segmentation_{i:05d}.png", mode=ImageReadMode.UNCHANGED))
        
        colors = self.color_trans(torch.stack(c_list))
        masks = self.mask_trans(torch.stack(m_list).squeeze(1))
        
        unique_ids = torch.unique(masks)
        unique_ids = unique_ids[(unique_ids != 0) & (unique_ids != 1)]
        
        seq_targets = []
        for t in range(self.num_seq):
            frame_m = masks[t]
            p_ids = [i for i in unique_ids if (frame_m == i).any()]
            if not p_ids:
                target = {"labels": torch.zeros(0, dtype=torch.int64), "boxes": torch.zeros(0, 4), "masks": torch.zeros(0, *self.target_size)}
            else:
                m_bin = torch.stack([(frame_m == i) for i in p_ids]).float()
                boxes = masks_to_boxes(m_bin)
                h, w = self.target_size
                boxes_norm = boxes / torch.tensor([w, h, w, h])
                cx, cy = (boxes_norm[:, 0] + boxes_norm[:, 2]) / 2, (boxes_norm[:, 1] + boxes_norm[:, 3]) / 2
                bw, bh = boxes_norm[:, 2] - boxes_norm[:, 0], boxes_norm[:, 3] - boxes_norm[:, 1]
                target = {"labels": torch.zeros(len(p_ids), dtype=torch.int64), "boxes": torch.stack([cx, cy, bw, bh], 1), "masks": m_bin}
            seq_targets.append(target)
        return colors, seq_targets

def custom_collate_fn(batch):
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_bbox=5.0):
        super().__init__()
        self.cost_class, self.cost_bbox = cost_class, cost_bbox

    @torch.no_grad()
    def forward(self, out_cls, out_box, targets):
        if targets["labels"].shape[0] == 0: return None
        C = self.cost_class * (-out_cls.sigmoid()[:, 0:1].expand(out_cls.shape[0], targets["labels"].shape[0])) + \
            self.cost_bbox * torch.cdist(out_box, targets["boxes"], p=1)
        src, tgt = linear_sum_assignment(C.cpu().numpy())
        return torch.as_tensor(src), torch.as_tensor(tgt)

# =================================================================
# 3. 손실 함수 및 학습 루프
# =================================================================

def compute_loss(preds, targets, matcher):
    cls_p, box_p, coeff_p, proto_p = preds
    l_cls, l_box, l_mask = 0, 0, 0
    total_bt = cls_p.shape[0]

    for i in range(total_bt):
        gt = targets[i]
        if len(gt["labels"]) == 0:
            l_cls += F.binary_cross_entropy_with_logits(cls_p[i], torch.zeros_like(cls_p[i]))
            continue
        
        indices = matcher(cls_p[i], box_p[i], {"labels": gt["labels"], "boxes": gt["boxes"].to(cls_p.device)})
        if indices is None: continue
        src, tgt = indices
        
        # Classification
        t_cls = torch.zeros_like(cls_p[i]); t_cls[src, 0] = 1.0
        l_cls += F.binary_cross_entropy_with_logits(cls_p[i], t_cls)
        
        # Box
        l_box += F.l1_loss(box_p[i][src], gt["boxes"][tgt].to(box_p.device))
        
        # Mask assembly
        m_p = torch.sigmoid(torch.einsum('nc,chw->nhw', coeff_p[i][src], proto_p[i]))
        m_gt = F.interpolate(gt["masks"][tgt].unsqueeze(1).to(m_p.device), size=m_p.shape[-2:], mode='bilinear').squeeze(1)
        l_mask += F.binary_cross_entropy(m_p, m_gt)

    return (l_cls * 1.0 + l_box * 2.0 + l_mask * 5.0) / total_bt

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 경로 설정 주의!
    dataset = MelonDataset(root_dir="/home/hyeonjun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/_train_dataset", num_frames=1000, num_seq=10)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn, num_workers=4)
    
    model = FastMelonSegmenter(n_segment=10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    matcher = HungarianMatcher()

    for epoch in range(50):
        for i, (imgs, tgts) in enumerate(loader):
            imgs = imgs.to(device)
            # targets 평탄화: [B, T] -> [B*T]
            flat_tgts = [f for b in tgts for f in b]
            
            preds = model(imgs)
            loss = compute_loss(preds, flat_tgts, matcher)
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if i % 10 == 0: print(f"Epoch {epoch} Step {i} Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()