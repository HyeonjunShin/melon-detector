import numpy as np
import cv2
import time
import math
from collections import deque
from tracker import MelonPosFilter



def img2world(u, v, K, D, rvec, tvec, height=0.0):
    # 1. 렌즈 왜곡 보정 (Undistort)
    points = np.array([[[u, v]]], dtype=np.float32)
    undistorted_points = cv2.undistortPoints(points, K, D, P=K)

    u_undist = undistorted_points[0][0][0]
    v_undist = undistorted_points[0][0][1]

    # 2. 좌표 변환 (cam to world coordinates)
    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    K_inv = np.linalg.inv(K)
    
    pixel_point = np.array([u_undist, v_undist, 1.0])
    point_camera = np.dot(K_inv, pixel_point)
    cam_world_center = -np.dot(R_inv, tvec)
    ray_world = np.dot(R_inv, point_camera)
    
    if abs(ray_world[2]) < 1e-6:
        return None 
        
    s = (height - cam_world_center[2, 0]) / ray_world[2]
    point_on_plane = cam_world_center + s * ray_world.reshape(3, 1)
    
    # [수정] numpy array가 아닌 순수 float 값으로 반환 (.item() 사용)
    # 이걸 안 하면 뒤에서 np.sqrt 계산할 때 에러가 발생함
    return point_on_plane[0, 0].item(), point_on_plane[1, 0].item(), point_on_plane[2, 0].item()


def img2world_with_depth(u, v, depth, K, rvec, tvec, D=None):
    # 1. 렌즈 왜곡 보정 (Undistort) - 기존과 동일
    if D is not None:
        points = np.array([[[u, v]]], dtype=np.float32)
        undistorted_points = cv2.undistortPoints(points, K, D, P=K)
        u_undist = undistorted_points[0][0][0]
        v_undist = undistorted_points[0][0][1]
    else:
        u_undist, v_undist = u, v

    # 2. 픽셀 -> 정규화된 카메라 좌표 (Normalized Camera Coordinates)
    # Z=1 인 평면상의 좌표를 구합니다.
    pixel_point = np.array([u_undist, v_undist, 1.0])
    K_inv = np.linalg.inv(K)
    
    # 정규화된 카메라 좌표 (x_n, y_n, 1)
    normalized_point = np.dot(K_inv, pixel_point)

    # 3. Depth 적용 (Camera Coordinate System에서의 실제 3D 좌표)
    # 정규화된 좌표에 깊이(Z)를 곱해 실제 크기를 복원합니다.
    # P_cam = [x_n * depth, y_n * depth, depth]
    point_in_cam = normalized_point * depth

    # 4. 좌표 변환 (Camera -> World)
    # P_cam = R * P_world + t  ==>  P_world = R_inv * (P_cam - t)
    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    
    # 3x1 벡터 연산을 위해 reshape
    point_in_cam = point_in_cam.reshape(3, 1)
    
    # 월드 좌표계로 변환
    point_world = np.dot(R_inv, point_in_cam - tvec)

    return point_world[0, 0], point_world[1, 0], point_world[2, 0]

def world2img(x, y, z, K, D, rvec, tvec):
    point_world = np.array([[x, y, z, 1]], dtype=np.float32).T

    R, _ = cv2.Rodrigues(rvec)
    RT = np.hstack((R, tvec)) # 3x4 Extrinsic Matrix
    point_camera = RT @ point_world

    point_image_homo = K @ point_camera[:3] # Z_c로 나누기 전 동차 좌표

    u_dist = point_image_homo[0] / point_image_homo[2]
    v_dist = point_image_homo[1] / point_image_homo[2]

    point_3d_for_project = np.array([[x, y, z]], dtype=np.float32)
    
    image_points, _ = cv2.projectPoints(point_3d_for_project, rvec, tvec, K, D)
    u_final = image_points[0, 0, 0]
    v_final = image_points[0, 0, 1]

    return u_final, v_final


def get_region_median_depth(depth_map, cx, cy, window_size=5):
    """
    특정 좌표(cx, cy) 주변의 window_size 영역에서 깊이 중앙값을 계산합니다.
    
    Args:
        depth_map (np.array): 깊이 이미지 (2D 배열)
        cx (int): 중심 x 좌표 (Column)
        cy (int): 중심 y 좌표 (Row)
        window_size (int): 윈도우 크기 (기본값 5)
        
    Returns:
        float: 계산된 중앙값. 유효한 값이 없으면 0.0을 반환.
    """
    
    # 1. 정수형 좌표 변환
    cx, cy = int(cx), int(cy)
    
    # 2. 이미지 크기 확인
    h, w = depth_map.shape
    
    # 3. 윈도우 범위 계산 (이미지 경계를 벗어나지 않도록 클리핑)
    # window_size가 5일 경우, 중심에서 -2 ~ +2 범위를 가짐 (반지름 = 2)
    radius = window_size // 2
    
    min_x = max(0, cx - radius)
    max_x = min(w, cx + radius + 1)
    min_y = max(0, cy - radius)
    max_y = min(h, cy + radius + 1)
    
    # 4. ROI(Region of Interest) 추출
    roi = depth_map[min_y:max_y, min_x:max_x]
    
    # 5. 유효한 깊이 값만 필터링 (0 또는 NaN 제외)
    # Depth 카메라 특성상 측정 실패 시 0이나 NaN이 들어오는 경우가 많음
    # 이를 포함해서 중앙값을 구하면 결과가 왜곡되므로 제거해야 함
    valid_pixels = roi[roi > 0]  # 0보다 큰 값만 추출
    
    # 만약 NaN이 포함된 float 형식이면 아래 코드를 사용하세요:
    # valid_pixels = roi[(roi > 0) & (~np.isnan(roi))]

    # 6. 중앙값 계산
    if len(valid_pixels) == 0:
        return 0.0  # 유효한 픽셀이 하나도 없으면 0 반환
        
    median_val = np.min(valid_pixels)
    
    return float(median_val)


def get_object_height_in_world(u, v, depth, K, rvec, tvec):
    """
    이미지 픽셀(u, v)과 깊이(depth)를 이용하여 월드 좌표계 기준의 높이(Z)를 반환합니다.
    
    Args:
        u, v: 픽셀 좌표
        depth: 해당 픽셀의 깊이 값 (단위: m 또는 mm)
        K: 카메라 내부 파라미터 (3x3)
        rvec: 회전 벡터 (Camera extrinsic)
        tvec: 이동 벡터 (Camera extrinsic)
        
    Returns:
        float: 월드 좌표계 기준의 높이 (World Z coordinate)
    """
    
    # 1. 픽셀 -> 정규화된 카메라 좌표 (Normalized Camera Coordinate)
    # 렌즈 왜곡이 없다고 가정하거나 이미 undistort 된 좌표를 넣었다고 가정
    pixel_point = np.array([u, v, 1.0])
    K_inv = np.linalg.inv(K)
    
    # 정규화 좌표 (x_n, y_n, 1)
    normalized_point = np.dot(K_inv, pixel_point)
    
    # 2. 카메라 기준 3D 좌표 복원 (Camera Coordinate)
    # P_cam = [x_c, y_c, z_c]
    point_in_camera = normalized_point * depth
    
    # 계산을 위해 shape 변경 (3,) -> (3, 1)
    point_in_camera = point_in_camera.reshape(3, 1)

    # 3. 월드 좌표계로 변환 (World Coordinate)
    # 공식: P_world = R_inv * (P_camera - tvec)
    # 설명: 카메라 좌표에서 tvec를 빼고, 회전의 역행렬을 곱함
    
    R, _ = cv2.Rodrigues(rvec) # 회전 벡터 -> 회전 행렬
    R_inv = R.T                # 회전 행렬의 역행렬 (Transpose)
    
    point_world = np.dot(R_inv, point_in_camera - tvec)
    
    # 4. 결과 반환
    # point_world[0] = World X
    # point_world[1] = World Y
    # point_world[2] = World Z (우리가 원하는 높이!)
    
    return point_world[2].item()

class ROIAverageSpeedCalculator:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        
        self.is_tracking = False
        self.prev_pos = None
        self.prev_time = 0
        self.speed_samples = []
        self.last_final_speed = 0.0

    def check_inside(self, x, y):
        return (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max)

    def process(self, curr_pos_3d, timestamp):        # [수정] timestamp를 사용하여 dt 계산
        curr_time = timestamp
        # curr_time = time.time()
        
        # [수정] 입력값을 확실하게 float로 변환 (Numpy array 에러 방지)
        wx, wy = 0.0, 0.0
        is_pos_valid = False
        
        if curr_pos_3d is not None:
            wx = float(curr_pos_3d[0])
            wy = float(curr_pos_3d[1])
            if self.check_inside(wx, wy):
                is_pos_valid = True

        # Case A: [진입]
        if not self.is_tracking and is_pos_valid:
            self.is_tracking = True
            self.prev_pos = (wx, wy)
            self.prev_time = curr_time
            self.speed_samples = []
            return None

        # Case B: [추적]
        elif self.is_tracking and is_pos_valid:
            dt = (curr_time - self.prev_time) / 1_000_000.0
            if dt > 0:
                # 거리 계산
                dist = math.sqrt((wx - self.prev_pos[0])**2 + (wy - self.prev_pos[1])**2)
                inst_speed = dist / dt
                
                # 노이즈 필터링 (0.01 ~ 5.0 m/s)
                if 0.01 < inst_speed < 5.0:
                    self.speed_samples.append(inst_speed)
            
            self.prev_pos = (wx, wy)
            self.prev_time = curr_time
            return None

        # Case C: [퇴장] (결과 반환)
        elif self.is_tracking and not is_pos_valid:
            self.is_tracking = False
            self.prev_pos = None
            
            final_avg_speed = 0.0
            if len(self.speed_samples) > 0:
                final_avg_speed = sum(self.speed_samples) / len(self.speed_samples)
            
            self.last_final_speed = final_avg_speed
            return final_avg_speed

        return None



if __name__ == "__main__":
    from kinect_playback import Playback
    from melon_detector import detect_melon
    import uuid # 고유 ID 생성을 위해 uuid 사용

    device = Playback("./data/recored_files/1080p.mkv")
    K = device.K
    D = device.D
    WIDTH = device.WIDTH
    HEIGHT = device.HEIGHT
    
    is_calibaration = False
    rvec = None
    tvec = None

    # Setting roi real scale
    roi_real = np.array([
        [0, 0, 0],
        [0.5, 0, 0]
    ], dtype=np.float32)

    new_mtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (WIDTH, HEIGHT), 0, (WIDTH, HEIGHT))

    # --- 멀티 오브젝트 트래커 상태 변수 ---
    trackers = {}  # {track_id: tracker_info} 형태로 저장
    next_track_id = 0
    
    # --- 월드 좌표계 기준 파라미터 ---
    # 감지된 객체와 기존 트래커를 매칭하기 위한 거리 임계값 (단위: 미터)
    WORLD_MATCH_THRESHOLD = 0.07 # 7cm
    # 트래커가 몇 프레임동안 보이지 않으면 삭제할지 결정
    FRAMES_TO_LIVE_WITHOUT_DETECTION = 30 
    
    # 칼만 필터 노이즈 설정 (월드 좌표계용)
    # 프로세스 노이즈: 모델(등속도)이 얼마나 부정확할 수 있는지. 컨베이어 벨트의 가속/진동.
    KF_PROCESS_NOISE = 1e-3 
    # 측정 노이즈: img2world 변환 결과가 얼마나 부정확할 수 있는지. (예: 5mm 오차 -> 0.005**2)
    KF_MEASUREMENT_NOISE = 2.5e-5

    # --- 평균 속도 계산을 위한 상태 변수 ---
    entry_pos = None
    entry_time = None
    last_inside_pos = None
    last_inside_time = None
    is_inside_prev = False

    melon = None

    fps_meter = FPSMeter()
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", (1920, 1080))
    
    # 최종적으로 저장될 Melon 객체들
    finalized_melons = {}

    melon_filter = MelonPosFilter()

    while True:
        try:
            frame = device.getImage()
        except EOFError as e:
            device.playback.seek(0)
        if frame is None: continue

        image = frame.image
        depth = frame.depth
        timestamp_frame = frame.timestamp_image

        # 1. Calibration
        if is_calibaration == False:
            results_aruco = detectArucoMarker(image, K, D, 0.027, 0.013, 0.405, 2, 5, cv2.aruco.DICT_4X4_50)
            if results_aruco:
                rvec, tvec = results_aruco
                roi_img, _ = cv2.projectPoints(roi_real, rvec, tvec, K, D)
                roi_Xmin = roi_img[0, 0, 0]
                roi_Xmax = roi_img[1, 0, 0]
                roi_img = roi_img.astype(int).reshape(-1, 2)
                is_calibaration = True
                print("Calibration Success!")
            else:
                continue

        # 2. Melon Detection, Tracking, and Prediction
        melon_results = detect_melon(image, image)
        
        # 현재 프레임에서 ROI 내부에 객체가 감지되었는지 여부
        is_inside_now = False
        if melon_results is not None:
            cx_raw, cy_raw, tx_raw, ty_raw = melon_results
            depth_target = get_region_median_depth(depth, cx_raw, cy_raw, 10)
            height = get_object_height_in_world(cx_raw, cy_raw, depth_target/ 1000.0, K, rvec, tvec)
            print(tvec)
            print(depth_target)
            print(height)
            cx_mm, cy_mm, _ = img2world(cx_raw, cy_raw, K, D, rvec, tvec, height=0.0)
            tx_mm, ty_mm, _ = img2world(tx_raw, ty_raw, K, D, rvec, tvec, height=0.0)

            # 감지된 객체가 ROI 내부에 있는지 확인
            is_inside_now = (cx_mm is not None) and (cx_mm > 0 and tx_mm > 0 and cx_mm < 0.7 and tx_mm < 0.7)

            if is_inside_now:
                if not is_inside_prev:  # Case 1: ROI 진입 시점
                    print("enter")
                    melon = Melon()
                    melon.setEntry(cx_mm, cy_mm, tx_mm, ty_mm, timestamp_frame)
                

        # --- ROI 상태 변화 감지 및 계산 (객체 감지 유무와 관계없이 매 프레임 실행) ---
        if not is_inside_now and is_inside_prev: # Case 3: ROI를 막 벗어난 시점
            print("out")
            melon.setLast(cx_mm, cy_mm, tx_mm, ty_mm, timestamp_frame)
            melon.calcAngle()
            melon.calcVelocity()

            print(melon.angle, melon.velocity, melon.cx_last, melon.cy_last)

        # 다음 프레임을 위해 현재 ROI 상태를 이전 상태로 저장
        is_inside_prev = is_inside_now


        cv2.polylines(image, [roi_img], True, (128, 128, 128), 20)
        drawMarker(image, rvec, tvec)

        fps_meter.draw(image)
        cv2.imshow("image", image)
        key = cv2.waitKeyEx(1)
        if key == ord('q') or key == 27:
            break
 
    device.close()
    cv2.destroyAllWindows()