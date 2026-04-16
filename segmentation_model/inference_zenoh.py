import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import zenoh
from camera_devices.kinect_wrapper import KinectCamera
from model import FastMelonSegmenter
from torchvision.ops import nms
from torchvision.transforms import v2

def normal_to_tf(centroid, normal):
    """
    Centroid와 Normal을 결합하여 4x4 TF 행렬 생성 (m 단위 유지)
    """
    z_axis = normal / np.linalg.norm(normal)
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
    tf_matrix[0:3, 3] = centroid  # 이미 m 단위
    return tf_matrix


def compute_melon_3d(mask, depth_map, K):
    """
    마스크 영역의 3D 포인트 클라우드 및 법선 벡터 계산
    결과값은 m(미터) 단위로 반환함
    """
    h, w = depth_map.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    y_idx, x_idx = np.where((mask > 0) & (depth_map > 0))
    
    # mm 단위를 m 단위로 변환
    z = depth_map[y_idx, x_idx] / 1000.0

    if len(z) < 100:
        return None

    x_3d = (x_idx - cx) * z / fx
    y_3d = (y_idx - cy) * z / fy
    points_3d = np.vstack((x_3d, y_3d, z)).T
    centroid_3d = np.mean(points_3d, axis=0)

    # 곡률 반영을 위한 Local PCA (반경 30mm -> 0.03m)
    dist = np.linalg.norm(points_3d - centroid_3d, axis=1)
    local_pts = points_3d[dist < 0.03]
    target_pts = local_pts if len(local_pts) > 20 else points_3d

    centered_pts = target_pts - np.mean(target_pts, axis=0)
    cov = np.cov(centered_pts.T)
    evals, evecs = np.linalg.eig(cov)
    normal = evecs[:, np.argmin(evals)]
    if normal[2] > 0:
        normal = -normal
    return {"points": points_3d, "normal": normal, "centroid": centroid_3d}

tf_camera = None
flange2camera = [
            [ 0.0, -0.93969262, -0.34202014,  0.09292 ],
            [ 1.0,  0.0,          0.0,          0.032   ],
            [ 0.0, -0.34202014,  0.93969262,  0.17445 ],
            [ 0.0,  0.0,          0.0,          1.0     ]
        ]

def on_request(sample):
    data = sample.payload.to_string()
    # state = data[0] # 필요시 사용
    tf_flange_str = data[1:]
    try:
        tf_flange = np.fromstring(tf_flange_str, sep=' ').reshape(4, 4)
        global tf_camera
        tf_camera = tf_flange @ flange2camera
    except Exception as e:
        print(f"TF Parsing Error: {e}")

# =========================================
# 2. 메인 실행 루프
# ==========================================
def run_realtime_inference():
    print("🌿 Zenoh 세션 연결 중...")
    try:
        conf = zenoh.Config()
        z_session = zenoh.open(conf)
        pub = z_session.declare_publisher("detector/response")
        sub = z_session.declare_subscriber("detector/request", on_request)
        print("✅ Zenoh 연결 성공 (Topic: detector/response)")
    except Exception as e:
        print(f"❌ Zenoh 연결 실패: {e}")
        return

    check_point_path = "./melon_segmenter_epoch_200.pth"
    target_img_size = (256, 448)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FastMelonSegmenter(num_classes=1, num_prototypes=32).to(device)
    model.load_state_dict(torch.load(check_point_path, map_location=device))
    model.eval()

    camera = KinectCamera()
    camera.start()
    K = camera.K

    # Open3D 설정 (단위가 m이므로 coordinate frame 사이즈 축소: 200mm -> 0.2m)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Detector Analysis", width=1024, height=768)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))

    prev_geoms = []
    is_first_view = True
    color_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(target_img_size, interpolation=v2.InterpolationMode.BILINEAR),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    rotation_matrix_y = np.array([
        [-1,  0,  0],
        [ 0,  1,  0],
        [ 0,  0, -1]
    ])

    with torch.no_grad():
        while True:
            frame_data = camera.get_frame()
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

            # --- 상대적 스코어링 및 Zenoh 전송 ---
            if raw_candidates:
                max_dist = max(c["centroid"][2] for c in raw_candidates)
                camera_vec = np.array([0, 0, -1])

                for c in raw_candidates:
                    # 거리와 법선 방향을 고려한 스코어링
                    c["total_score"] = (1.0 - (c["centroid"][2] / (max_dist * 1.1))) + np.clip(
                        np.dot(c["normal"], camera_vec), 0, 1
                    )
                raw_candidates.sort(key=lambda x: x["total_score"], reverse=True)

                for rank, melon in enumerate(raw_candidates):
                    tf = normal_to_tf(melon["centroid"], melon["normal"])
                    tf[:3, :3] = rotation_matrix_y @ tf[:3, :3]
                    
                    # 로봇 좌표계 변환이 필요한 경우 주석 해제하여 사용
                    if tf_camera is not None:
                        tf = tf_camera @ tf 
                    
                    tf_flat = tf.flatten().tolist()

                    # 데이터 포맷: rank score tf[0~15] (모두 m 단위)
                    payload = [rank, round(melon["total_score"], 4)] + [
                        round(x, 6) for x in tf_flat
                    ]
                    msg = " ".join(map(str, payload))

                    pub.put(msg)

                    if rank == 0:
                        # 출력 로그에서도 m 단위로 표시
                        print(
                            f"\r📡 Target #0 | Score: {payload[1]:.3f} | Z: {melon['centroid'][2]:.3f}m",
                            end="",
                        )

            # --- 시각화 (단위: m) ---
            current_geoms = []
            for i, melon in enumerate(raw_candidates):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(melon["points"])
                pcd.paint_uniform_color([1.0, 0.8, 0.0] if i == 0 else [0.4, 0.4, 0.4])
                current_geoms.append(pcd)

                # 멜론별 좌표축 (80mm -> 0.08m)
                melon_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
                tf_vis = normal_to_tf(melon["centroid"], melon["normal"])
                tf_vis[:3, :3] = rotation_matrix_y @ tf_vis[:3, :3]
                melon_frame.transform(tf_vis)
                current_geoms.append(melon_frame)

                # 2D 이미지상에 Rank 표시 (계산 시 m 단위를 다시 픽셀 좌표로 변환)
                z_m = melon['centroid'][2]
                u = int(melon["centroid"][0] * K[0, 0] / z_m + K[0, 2])
                v = int(melon["centroid"][1] * K[1, 1] / z_m + K[1, 2])
                cv2.putText(
                    vis_frame,
                    f"Rank {i}",
                    (u - 30, v - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

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

            cv2.imshow("Detection (Zenoh Active)", vis_frame)
            if cv2.waitKey(1) == ord("q"):
                break

    camera.stop()
    cv2.destroyAllWindows()
    vis.destroy_window()
    z_session.close()


if __name__ == "__main__":
    run_realtime_inference()