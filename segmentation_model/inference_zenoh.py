import os
import sys

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import zenoh

# 카메라 및 모델 관련 임포트 (기존 유지)
from camera_devices.shm_sub import ShmCamera
from model import FastMelonSegmenter
from torchvision.ops import nms
from torchvision.transforms import v2

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from constants import CameraConfig


# ==========================================
# 1. 3D 및 TF 연산 함수
# ==========================================
def normal_to_tf(centroid, normal):
    """
    Centroid와 Normal을 결합하여 4x4 TF 행렬 생성
    Z축(Blue)이 normal 방향(물체 내부)을 향하도록 설정
    """
    z_axis = normal / np.linalg.norm(normal)
    
    # x_axis 계산을 위한 임의의 벡터 설정 (Singularity 방지)
    random_vec = np.array([0, 1, 0])
    if abs(np.dot(random_vec, z_axis)) > 0.99:
        random_vec = np.array([1, 0, 0])
        
    x_axis = np.cross(random_vec, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    tf_matrix = np.eye(4)
    tf_matrix[0:3, 0] = x_axis
    tf_matrix[0:3, 1] = y_axis
    tf_matrix[0:3, 2] = z_axis
    tf_matrix[0:3, 3] = centroid
    return tf_matrix


def compute_melon_3d(mask, depth_map, K):
    """
    마스크 영역 침식(Erosion) 후 3D 포인트 및 Suction용 Normal 계산
    """
    # [추가] 마스크 외곽의 벽면/배경 노이즈 제거를 위해 5x5 커널로 침식
    kernel = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    depth_map = depth_map.squeeze()
    h, w = depth_map.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    
    # 침식된 마스크 영역에서만 Depth 추출
    y_idx, x_idx = np.where((eroded_mask > 0) & (depth_map > 0))
    z = depth_map[y_idx, x_idx]

    if len(z) < 80:  # 유효 포인트가 너무 적으면 제외
        return None

    x_3d = (x_idx - cx) * z / fx
    y_3d = (y_idx - cy) * z / fy
    points_3d = np.vstack((x_3d, y_3d, z)).T
    centroid_3d = np.mean(points_3d, axis=0)

    # 곡률 반영을 위한 Local PCA (반경 25mm로 소폭 조정)
    dist = np.linalg.norm(points_3d - centroid_3d, axis=1)
    local_pts = points_3d[dist < 25]
    target_pts = local_pts if len(local_pts) > 20 else points_3d

    centered_pts = target_pts - np.mean(target_pts, axis=0)
    cov = np.cov(centered_pts.T)
    evals, evecs = np.linalg.eig(cov)
    normal = evecs[:, np.argmin(evals)]

    # [수정] Suction 방향 설정: Z축이 물체 내부(바닥 방향)를 향하도록 함
    # 카메라 좌표계에서 Z+는 앞쪽이므로, normal[2] > 0 이어야 물체 안쪽을 향함
    if normal[2] < 0:
        normal = -normal
        
    return {"points": points_3d, "normal": normal, "centroid": centroid_3d}


# ==========================================
# 2. 메인 실행 루프
# ==========================================
def run_realtime_inference():
    print("🌿 Zenoh 세션 연결 중...")
    try:
        conf = zenoh.Config()
        z_session = zenoh.open(conf)
        pub = z_session.declare_publisher("detector/response")
        print("✅ Zenoh 연결 성공 (Topic: detector/response)")
    except Exception as e:
        print(f"❌ Zenoh 연결 실패: {e}")
        return

    # 모델 설정
    check_point_path = ".checkpoints/res18/best_model.pth"
    target_img_size = (256, 448)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FastMelonSegmenter(num_classes=1, num_prototypes=32).to(device)
    model.load_state_dict(torch.load(check_point_path, map_location=device)["model"])
    model.eval()

    shm_camera = ShmCamera()
    K = CameraConfig.INTRINSIC

    # Open3D 설정
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Melon Suction Detector", width=1024, height=768)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=200))

    prev_geoms = []
    is_first_view = True
    color_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize(target_img_size, interpolation=v2.InterpolationMode.BILINEAR),
        v2.ToDtype(torch.float32, scale=True),
    ])

    with torch.no_grad():
        while True:
            frame_data = shm_camera.get_frame()
            if frame_data is None:
                continue
            color, depth, _ = frame_data
            h_orig, w_orig = color.shape[:2]

            input_tensor = color_transforms(color).unsqueeze(0).to(device)
            logits, boxes, coeffs, prototypes = model(input_tensor)

            scores = logits.sigmoid()[0, :, 0]
            keep_conf = scores > 0.4

            vis_frame = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            raw_candidates = []

            if keep_conf.any():
                v_scores, v_boxes = scores[keep_conf], boxes[0, keep_conf]
                v_coeffs = coeffs[0, keep_conf]
                keep_idx = nms(v_boxes, v_scores, iou_threshold=0.3)
                v_coeffs = v_coeffs[keep_idx]

                masks = torch.einsum("nc,chw->nhw", v_coeffs, prototypes[0]).sigmoid()
                masks = F.interpolate(
                    masks.unsqueeze(1), size=(h_orig, w_orig), mode="bilinear"
                ).squeeze(1)
                binary_masks = (masks > 0.5).cpu().numpy().astype(np.uint8)

                for i in range(len(binary_masks)):
                    res = compute_melon_3d(binary_masks[i], depth, K)
                    if res:
                        raw_candidates.append(res)

            # 상대적 스코어링 및 Zenoh 전송
            if raw_candidates:
                max_dist = max(c["centroid"][2] for c in raw_candidates)
                # Suction 방향(물체 내부)을 바라보는 벡터와 일치하는지 평가
                inner_vec = np.array([0, 0, 1])

                for c in raw_candidates:
                    # 거리 점수 + 방향 점수 (Z축 방향이 depth 방향과 잘 일치하는지)
                    c["total_score"] = (1.0 - (c["centroid"][2] / (max_dist * 1.1))) + \
                                       np.clip(np.dot(c["normal"], inner_vec), 0, 1)
                
                raw_candidates.sort(key=lambda x: x["total_score"], reverse=True)

                for rank, melon in enumerate(raw_candidates):
                    tf = normal_to_tf(melon["centroid"], melon["normal"])
                    tf_flat = tf.flatten().tolist()

                    payload = [rank, round(melon["total_score"], 4)] + [round(x, 4) for x in tf_flat]
                    msg = " ".join(map(str, payload))
                    pub.put(msg)

                    if rank == 0:
                        print(f"\r🎯 Target Locked | Score: {payload[1]:.3f} | Z: {melon['centroid'][2]:.1f}mm", end="")

            # 시각화 업데이트
            current_geoms = []
            for i, melon in enumerate(raw_candidates):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(melon["points"])
                pcd.paint_uniform_color([1.0, 0.8, 0.0] if i == 0 else [0.4, 0.4, 0.4])
                current_geoms.append(pcd)

                melon_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=80)
                melon_frame.transform(normal_to_tf(melon["centroid"], melon["normal"]))
                current_geoms.append(melon_frame)

                # 2D 화면 표시
                u = int(melon["centroid"][0] * K[0, 0] / melon["centroid"][2] + K[0, 2])
                v = int(melon["centroid"][1] * K[1, 1] / melon["centroid"][2] + K[1, 2])
                cv2.putText(vis_frame, f"Rank {i}", (u - 30, v - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            for g in prev_geoms:
                vis.remove_geometry(g, reset_bounding_box=False)
            for g in current_geoms:
                vis.add_geometry(g, reset_bounding_box=False)
            
            if current_geoms and is_first_view:
                vis.reset_view_point(True)
                is_first_view = False
                
            vis.poll_events()
            vis.update_renderer()
            prev_geoms = current_geoms

            cv2.imshow("Melon Detection (Zenoh Active)", vis_frame)
            if cv2.waitKey(1) == ord("q"):
                break

    cv2.destroyAllWindows()
    vis.destroy_window()
    z_session.close()


if __name__ == "__main__":
    run_realtime_inference()