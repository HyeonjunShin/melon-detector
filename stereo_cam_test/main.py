import cv2
import numpy as np

# device 모듈에서 카메라 로드 함수 임포트
try:
    from device import loadStereoCamera
except ImportError:
    print("Warning: 'device' 모듈을 찾을 수 없습니다.")
    def loadStereoCamera(): return None, None

# ==========================================
# 1. 파라미터 직접 정의 (Hardcoding)
# ==========================================
# 알려진 카메라 내부 파라미터 (K)
K_LEFT = np.array([[604.78833008, 0., 641.39819336],
                   [0., 604.62579346, 365.25283813],
                   [0., 0., 1.]])
D_LEFT = np.array([2.63843089e-01, -2.51909852e+00, -3.29987561e-05, -1.81387455e-04, 1.62435722e+00, 1.49601102e-01, -2.34095073e+00, 1.54302502e+00])

K_RIGHT = np.array([[608.21765137, 0., 641.05187988],
                    [0., 608.07714844, 366.93432617],
                    [0., 0., 1.]])
D_RIGHT = np.array([2.79455930e-01, -2.58234048e+00, 4.89634229e-04, -2.01776158e-04, 1.61805654e+00, 1.59334257e-01, -2.39073086e+00, 1.53187358e+00])

# 베이스라인 (mm) 및 이미지 크기
BASELINE = 150.0
IMG_SIZE = (1280, 720) # 카메라 해상도에 맞게 설정 (예: HD)

# ==========================================
# 2. 외부 파라미터 설정 (No Rotation Assumption)
# ==========================================
# 회전 없음 (단위 행렬)
R = np.eye(3)
# 이동 (X축으로 -Baseline만큼 이동, 오른쪽 카메라 기준)
T = np.array([[-BASELINE], [0], [0]])

# ==========================================
# 3. 정렬 맵 생성 (Rectification)
# ==========================================
print("정렬 맵(Rectification Map) 생성 중...")
# alpha=0: 검은 영역 제거(Zoom), alpha=1: 전체 유지
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K_LEFT, D_LEFT, K_RIGHT, D_RIGHT, IMG_SIZE, R, T, alpha=1
)

map1_l, map2_l = cv2.initUndistortRectifyMap(K_LEFT, D_LEFT, R1, P1, IMG_SIZE, cv2.CV_32FC1)
map1_r, map2_r = cv2.initUndistortRectifyMap(K_RIGHT, D_RIGHT, R2, P2, IMG_SIZE, cv2.CV_32FC1)

# ==========================================
# 4. 실시간 루프 (Real-time Loop)
# ==========================================
cam_left, cam_right = loadStereoCamera()
if not cam_left or not cam_right: exit()

window_name = "Stereo Depth (Overlay View)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

# 튜닝용 트랙바 설정
def nothing(x): pass
cv2.createTrackbar('Num Disp (x16)', window_name, 8, 20, nothing)
cv2.createTrackbar('Min Disp', window_name, 0, 100, nothing)
cv2.createTrackbar('View Mode', window_name, 0, 1, nothing) # 0: Side-by-Side, 1: Overlay

stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=3, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

while True:
    capture_l = cam_left.get_capture()
    capture_r = cam_right.get_capture()

    if capture_l is None or capture_r is None: continue

    # 4채널 -> 3채널 변환
    frame_l = capture_l.color[:, :, :3]
    frame_r = capture_r.color[:, :, :3]

    # 파라미터 업데이트
    num_disp = max(16, cv2.getTrackbarPos('Num Disp (x16)', window_name) * 16)
    min_disp = cv2.getTrackbarPos('Min Disp', window_name)
    view_mode = cv2.getTrackbarPos('View Mode', window_name)
    
    stereo.setNumDisparities(num_disp)
    stereo.setMinDisparity(min_disp)

    # 1. 정렬 (Rectification) - 미리 계산된 맵 사용
    rect_l = cv2.remap(frame_l, map1_l, map2_l, cv2.INTER_LINEAR)
    rect_r = cv2.remap(frame_r, map1_r, map2_r, cv2.INTER_LINEAR)

    # 2. 깊이 계산 (Disparity)
    disparity = stereo.compute(rect_l, rect_r).astype(np.float32) / 16.0

    # 3. 시각화
    disp_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_color = cv2.applyColorMap(disp_visual, cv2.COLORMAP_JET)

    # [수정] Overlay 모드 추가: 왼쪽 정렬 영상(rect_l) 기준으로 Disparity 겹쳐보기
    if view_mode == 1:
        # 왼쪽 영상 60% + Disparity 40% 합성
        display_frame = cv2.addWeighted(rect_l, 0.6, disp_color, 0.4, 0)
    else:
        # 기존: 나란히 보기 (왼쪽: rect_l, 오른쪽: disp_color)
        display_frame = np.hstack((rect_l, disp_color))

    cv2.imshow(window_name, display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

if hasattr(cam_left, 'close'): cam_left.close()
if hasattr(cam_right, 'close'): cam_right.close()
cv2.destroyAllWindows()