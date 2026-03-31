from uon_camera.camera_kinect import KinectCamera
import cv2
import numpy as np
import open3d as o3d
import copy

def depth_to_point_cloud(depth_map, K, rgb_image=None, mask=None, scale=1000.0):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    rows, cols = depth_map.shape
    
    # sparse=False로 전체 그리드 생성
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # 1. 기본 유효성 검사 (깊이 값이 존재하는지)
    valid = (depth_map > 0) & (depth_map < 65535)
    
    # 2. 마스크가 있다면 마스크 영역만 유효한 것으로 처리
    if mask is not None:
        # mask가 0보다 큰 부분(흰색 부분)만 AND 연산
        valid = valid & (mask > 0)
    
    z = depth_map[valid] / scale
    c = c[valid]
    r = r[valid]
    
    x = (c - cx) * z / fx
    y = (r - cy) * z / fy
    
    points = np.dstack((x, y, z)).reshape(-1, 3)
    
    colors = None
    if rgb_image is not None:
        rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        colors = rgb[valid].reshape(-1, 3) / 255.0
        
    return points, colors

def get_picking_points(pcd):
    """
    Point Cloud에서 물체를 클러스터링하고 각 물체의 중심과 자세(Picking Point)를 찾습니다.
    """
    picking_geometries = []
    
    # 디버깅: 입력된 점의 개수 확인
    print(f"클러스터링 입력 점 개수: {len(pcd.points)}")

    # 1. 노이즈 제거 (점이 너무 적으면 클러스터링 에러 날 수 있으므로 체크)
    if len(pcd.points) < 10: 
        return picking_geometries
        
    pcd_clean = pcd

    # 2. 클러스터링 (DBSCAN) 조건 완화
    # eps: 0.02(2cm) -> 0.05(5cm)로 늘려서 점들이 조금 떨어져 있어도 같은 물체로 인식하게 함
    # min_points: 50 -> 10으로 줄여서 작은 덩어리도 인식하게 함
    labels = np.array(pcd_clean.cluster_dbscan(eps=0.05, min_points=10, print_progress=False))
    
    if len(labels) == 0:
        print("클러스터링 실패: 라벨이 생성되지 않음")
        return picking_geometries
        
    max_label = labels.max()
    print(f"감지된 물체(참외) 개수: {max_label + 1}") 
    
    # 각 클러스터(물체)별로 반복
    for i in range(max_label + 1):
        # 현재 클러스터의 인덱스 추출
        cluster_indices = np.where(labels == i)[0]
        cluster_pcd = pcd_clean.select_by_index(cluster_indices)
        
        # 점 개수가 너무 적으면 노이즈로 간주하고 패스
        if len(cluster_pcd.points) < 10:
            continue

        # 3. 중심점(Centroid) 계산
        center = cluster_pcd.get_center()
        
        # 4. 자세(Orientation) 및 법선 계산 (PCA)
        try:
            mean, covariance = cluster_pcd.compute_mean_and_covariance()
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        except Exception as e:
            print("PCA 계산 오류:", e)
            continue
        
        # --- [수정] 표면 Picking Point 찾기 ---
        # 4-1. 접근 벡터(법선) 선택
        # eigenvectors[:, 0]: 가장 긴 축 (빨강), eigenvectors[:, 2]: 가장 짧은 축 (파랑, 보통 법선 방향)
        # 그리퍼가 접근할 방향을 파란색 축으로 가정합니다.
        approach_vector = eigenvectors[:, 2]

        # 4-2. 표면 점 탐색 (KDTree 활용)
        # 중심에서 접근 벡터 방향으로 약간 떨어진 가상의 점을 만들고,
        # 그 점에서 가장 가까운 실제 참외 포인트 클라우드의 점을 찾습니다.
        pcd_tree = o3d.geometry.KDTreeFlann(cluster_pcd)
        
        # 중심에서 법선 방향으로 10cm 떨어진 시험용 점 생성
        # (방향이 반대일 수도 있지만 우선 한쪽 방향으로 가정)
        test_point = center + approach_vector * 0.1
        
        # 시험용 점에서 가장 가까운 점 1개 찾기
        [k, idx, _] = pcd_tree.search_knn_vector_3d(test_point, 1)
        surface_point = np.asarray(cluster_pcd.points)[idx[0]]

        # 4-3. 좌표축 생성 (Picking Point 시각화)
        # origin을 중심점(center)에서 표면 점(surface_point)으로 변경
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.15, origin=surface_point)
        
        # 계산된 회전 행렬(eigenvectors)을 적용하여 물체 방향에 맞춤
        # 회전 중심도 표면 점으로 변경
        mesh_frame.rotate(eigenvectors, center=surface_point)
        picking_geometries.append(mesh_frame)
        # ------------------------------------
        
        # Bounding Box도 추가 (확인용)
        bbox = cluster_pcd.get_axis_aligned_bounding_box()
        bbox.color = (0, 1, 0) # 초록색
        picking_geometries.append(bbox)

    return picking_geometries

if __name__ == "__main__":
    camera = KinectCamera()
    camera.start()
    K = camera.K
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Real-time Picking Point", width=960, height=540)
    
    # 전체 포인트 클라우드 담을 객체
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    
    # 원점 좌표축
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    vis.add_geometry(axis)

    is_first_frame = True
    previous_geometries = []

    try:
        while True:
            frame = camera.getFrame()
            if frame is None:
                continue
            
            image, depth, timestamp = frame
            
            # 1. OpenCV에서 노란색 마스크 생성
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # 노란색 범위
            mask_yellow = cv2.inRange(hsv, (10, 100, 100), (40, 255, 255)) 
            
            # 노이즈 제거
            kernel = np.ones((5,5), np.uint8)
            mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)

            # 2. 마스크를 적용해서 '참외 부분만' Point Cloud로 변환
            points_3d, colors_3d = depth_to_point_cloud(depth, K, rgb_image=image, mask=mask_yellow)
            
            if points_3d.shape[0] > 0:
                pcd.points = o3d.utility.Vector3dVector(points_3d)
                pcd.colors = o3d.utility.Vector3dVector(colors_3d)
            else:
                pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
                pcd.colors = o3d.utility.Vector3dVector(np.zeros((1, 3)))

            # 3. Picking Point 계산 및 시각화 갱신
            # 이전 프레임의 지오메트리 삭제
            for geom in previous_geometries:
                vis.remove_geometry(geom, reset_bounding_box=False)
            previous_geometries.clear()

            # 다운샘플링 후 클러스터링
            # 입력 점 개수가 너무 적으면 다운샘플링 없이 진행하거나 패스
            if len(pcd.points) > 50:
                # 점이 너무 적어지는 걸 방지하기 위해 voxel_size 조절 가능 (0.01 -> 0.005 등)
                pcd_down = pcd.voxel_down_sample(voxel_size=0.01)
                
                # 피킹 포인트 계산
                pick_geoms = get_picking_points(pcd_down)
                
                # 시각화에 추가
                for geom in pick_geoms:
                    vis.add_geometry(geom, reset_bounding_box=False)
                    previous_geometries.append(geom)

            # 초기 카메라 뷰 설정
            if is_first_frame:
                ctr = vis.get_view_control()
                ctr.set_lookat([0, 0, 1.0]) 
                ctr.set_up([0, -1, 0])
                ctr.set_front([0, 0, -1])
                ctr.set_zoom(0.8) 
                is_first_frame = False

            # 렌더링 업데이트
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            # 디버깅 창
            cv2.imshow("Yellow Mask", mask_yellow)
            cv2.imshow("2D Image", image)

            key = cv2.waitKeyEx(1)
            if key == ord('q') or key == 27:
                break
                
    finally:
        camera.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()