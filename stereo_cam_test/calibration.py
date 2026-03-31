import cv2
import numpy as np
import glob
import os
# device 모듈에서 카메라 로드 함수 임포트
try:
    from device import loadStereoCamera
except ImportError:
    print("Warning: 'device' 모듈을 찾을 수 없습니다.")
    def loadStereoCamera():
        return None, None

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
CHECKERBOARD_SIZE = (7, 10)
SQUARE_SIZE = 15  # mm 단위

# 이미지 경로
LEFT_IMAGE_DIR = './output/calib/left/*.png'
RIGHT_IMAGE_DIR = './output/calib/right/*.png'

# --- [수정] 알고 있는 카메라 파라미터 입력 구간 ---
# 실제 카메라 스펙이나 이전 보정 데이터를 여기에 입력하세요.
# 예시 값입니다 (자신의 카메라에 맞는 값으로 수정 필수)

# 왼쪽 카메라 Intrinsic Matrix (K)
K_LEFT = np.array([[1000.0, 0.0, 640.0],
                   [0.0, 1000.0, 360.0],
                   [0.0, 0.0, 1.0]])
# 왼쪽 카메라 Distortion Coefficients (D)
D_LEFT = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# 오른쪽 카메라 Intrinsic Matrix (K)
K_RIGHT = np.array([[1000.0, 0.0, 640.0],
                    [0.0, 1000.0, 360.0],
                    [0.0, 0.0, 1.0]])
# 오른쪽 카메라 Distortion Coefficients (D)
D_RIGHT = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# 알고 있는 베이스라인 길이 (검증용, 단위: mm)
KNOWN_BASELINE = 60.0 
# ------------------------------------------------

objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints_l = []
imgpoints_r = []

images_left = sorted(glob.glob(LEFT_IMAGE_DIR))
images_right = sorted(glob.glob(RIGHT_IMAGE_DIR))

print(f"이미지 로드: 왼쪽 {len(images_left)}장, 오른쪽 {len(images_right)}장")

img_size = None

# ==========================================
# 2. 코너 검출 (보정 데이터 수집)
# ==========================================
print("1단계: 체스보드 코너 검출 시작...")

if len(images_left) > 0:
    for i, (fname_l, fname_r) in enumerate(zip(images_left, images_right)):
        img_l = cv2.imread(fname_l)
        img_r = cv2.imread(fname_r)
        
        if img_size is None:
            img_size = (img_l.shape[1], img_l.shape[0])

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD_SIZE, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD_SIZE, None)

        if ret_l and ret_r:
            objpoints.append(objp)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            imgpoints_l.append(corners_l)
            imgpoints_r.append(corners_r)

    print("2단계: 스테레오 보정 (외부 파라미터 계산) 중...")
    if len(objpoints) > 0:
        # [중요] 이미 알고 있는 K, D 값을 초기값으로 사용
        mtx_l = K_LEFT.copy()
        dist_l = D_LEFT.copy()
        mtx_r = K_RIGHT.copy()
        dist_r = D_RIGHT.copy()

        # [중요] CALIB_FIX_INTRINSIC 플래그 사용
        # 내부 파라미터는 고정하고, 회전(R)과 이동(T)만 계산함
        flags = cv2.CALIB_FIX_INTRINSIC
        
        criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        
        ret_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_l, imgpoints_r,
            mtx_l, dist_l, mtx_r, dist_r,
            img_size, criteria=criteria_stereo, flags=flags
        )
        
        calculated_baseline = np.linalg.norm(T)
        print(f"스테레오 보정 완료 (RMS 오차: {ret_stereo:.4f})")
        print(f"알고 있는 베이스라인: {KNOWN_BASELINE} mm")
        print(f"계산된 베이스라인  : {calculated_baseline:.2f} mm")
        print(f"오차: {abs(KNOWN_BASELINE - calculated_baseline):.2f} mm")

        # 3단계: 정렬(Rectification) 맵 생성
        print("3단계: 정렬(Rectification) 맵 생성...")
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            mtx_l, dist_l, mtx_r, dist_r, img_size, R, T, alpha=1
        )

        map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_size, cv2.CV_32FC1)
        map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_size, cv2.CV_32FC1)
    else:
        print("유효한 체스보드 패턴을 찾지 못했습니다. 보정을 건너뜁니다.")
        map1_l, map2_l = None, None
        map1_r, map2_r = None, None
else:
    print("이미지가 없어 보정을 수행하지 않습니다.")
    map1_l, map2_l = None, None
    map1_r, map2_r = None, None

# ==========================================
# 4. 실시간 깊이 맵 생성
# ==========================================
print("\n=== 실시간 스테레오 깊이 맵 모드 시작 ===")
print("카메라를 엽니다... (종료하려면 'q'를 누르세요)")
print("팁: 윈도우의 Trackbar를 조절하여 깊이 맵 품질을 개선하세요.")

cam_left, cam_right = loadStereoCamera()

if cam_left is None or cam_right is None:
    print("오류: 카메라를 로드할 수 없습니다.")
    exit()

window_name = "Real-time Stereo Disparity"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

def nothing(x): pass

cv2.createTrackbar('Num Disparities (x16)', window_name, 8, 20, nothing)
cv2.createTrackbar('Min Disparity', window_name, 0, 100, nothing)
cv2.createTrackbar('Block Size (Odd)', window_name, 3, 21, nothing)
cv2.createTrackbar('Uniqueness Ratio', window_name, 10, 50, nothing)
cv2.createTrackbar('Speckle Window', window_name, 100, 200, nothing)
cv2.createTrackbar('Speckle Range', window_name, 32, 100, nothing)

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=3,
    P1=8 * 3 * 3**2,
    P2=32 * 3 * 3**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

while True:
    num_disp_val = cv2.getTrackbarPos('Num Disparities (x16)', window_name)
    num_disp = num_disp_val * 16 if num_disp_val > 0 else 16
    min_disp = cv2.getTrackbarPos('Min Disparity', window_name)
    blk_size = cv2.getTrackbarPos('Block Size (Odd)', window_name)
    if blk_size % 2 == 0: blk_size += 1
    if blk_size < 3: blk_size = 3
    uniqueness = cv2.getTrackbarPos('Uniqueness Ratio', window_name)
    speckle_win = cv2.getTrackbarPos('Speckle Window', window_name)
    speckle_range = cv2.getTrackbarPos('Speckle Range', window_name)

    stereo.setNumDisparities(num_disp)
    stereo.setMinDisparity(min_disp)
    stereo.setBlockSize(blk_size)
    stereo.setP1(8 * 3 * blk_size**2)
    stereo.setP2(32 * 3 * blk_size**2)
    stereo.setUniquenessRatio(uniqueness)
    stereo.setSpeckleWindowSize(speckle_win)
    stereo.setSpeckleRange(speckle_range)

    capture_left = cam_left.get_capture()
    capture_right = cam_right.get_capture()

    if capture_left is None or capture_right is None:
        continue
    
    frame_l = capture_left.color
    frame_r = capture_right.color

    if frame_l.shape[2] == 4:
        frame_l = cv2.cvtColor(frame_l, cv2.COLOR_BGRA2BGR)
    if frame_r.shape[2] == 4:
        frame_r = cv2.cvtColor(frame_r, cv2.COLOR_BGRA2BGR)

    if map1_l is not None and map1_r is not None:
        rect_l = cv2.remap(frame_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(frame_r, map1_r, map2_r, cv2.INTER_LINEAR)

        disparity = stereo.compute(rect_l, rect_r).astype(np.float32) / 16.0

        disp_visual = (disparity - min_disp) / num_disp
        disp_visual = np.clip(disp_visual, 0, 1)
        disp_visual = (disp_visual * 255).astype(np.uint8)

        disp_color = cv2.applyColorMap(disp_visual, cv2.COLORMAP_JET)

        display_frame = np.hstack((rect_l, disp_color))
    else:
        display_frame = np.hstack((frame_l, frame_r))
        cv2.putText(display_frame, "Calibration Data Missing", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow(window_name, display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if hasattr(cam_left, 'close'):
    cam_left.close()
if hasattr(cam_right, 'close'):
    cam_right.close()
    
cv2.destroyAllWindows()