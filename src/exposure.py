import cv2
import numpy as np

def get_safe_triangles(cnt):
    # 1. 외부 삼각형 (Outer Enclosing) - 무조건 3개 점 반환
    res, outer_tri = cv2.minEnclosingTriangle(cnt)
    outer_pts = outer_tri.reshape(3, 2)

    # 2. 내부 삼각형 찾기 (윤곽선 점들 중 외부 꼭짓점과 가장 가까운 점들)
    # cnt는 (N, 1, 2) 형태이므로 (N, 2)로 변환
    cnt_pts = cnt.reshape(-1, 2)
    
    inner_pts = []
    for op in outer_pts:
        # 외부 삼각형 각 꼭짓점에서 가장 가까운 윤곽선 위의 점 탐색
        dists = np.linalg.norm(cnt_pts - op, axis=1)
        closest_pt = cnt_pts[np.argmin(dists)]
        inner_pts.append(closest_pt)
    
    inner_pts = np.array(inner_pts)
    outer_pts = np.array(outer_pts).astype(int)
    return outer_pts, inner_pts

def getConterPoints(points):
    point = np.array(points).squeeze()
    center_point = np.mean(point, axis=0).astype(int)
    return center_point

def get_max_inscribed_triangle(contour):
    hull = cv2.convexHull(contour)
    hull = hull.reshape(-1, 2)
    
    max_area = 0
    best_triangle = None
    
    # 헐 점들 중 3개 조합으로 최대 면적 찾기 (단, 점이 많으면 샘플링 필요)
    # 점이 너무 많으면 성능을 위해 hull = hull[::step] 사용
    n = len(hull)
    step = max(1, n // 20) # 성능을 위해 최대 20개 점으로 제한
    sample_hull = hull[::step]
    n_sample = len(sample_hull)

    for i in range(n_sample):
        for j in range(i + 1, n_sample):
            for k in range(j + 1, n_sample):
                p1, p2, p3 = sample_hull[i], sample_hull[j], sample_hull[k]
                # 삼각형 면적 공식
                area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                if area > max_area:
                    max_area = area
                    best_triangle = np.array([p1, p2, p3])
    
    return best_triangle

def nothing(x):
    pass

# 1. 카메라 연결 (Windows의 경우 cv2.CAP_DSHOW 권장)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Focus
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
focus_value = 20 
cap.set(cv2.CAP_PROP_FOCUS, focus_value)

# Exposure
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
# exposure_value = -100
# cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)

is_recording = False
out = None
fourcc = cv2.VideoWriter_fourcc(*'XVID')


# 2. 제어용 윈도우 및 트랙바 생성
cv2.namedWindow('Control Panel', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Control Panel', 3840//2, 2160//2)

# 카메라 하드웨어 제어 트랙바 (기본값은 카메라마다 다를 수 있음)
cv2.createTrackbar('Focus', 'Control Panel', 0, 255, nothing)     # 초점
cv2.setTrackbarPos('Focus', 'Control Panel', focus_value)
cv2.createTrackbar('Exposure', 'Control Panel', 5, 10, nothing)  # 노출 (- 값 대신 0-10 범위로 설정)

# HSV 색상 제어 트랙바 (노란색 기본값 세팅)
cv2.createTrackbar('H_Low', 'Control Panel', 15, 179, nothing)
cv2.createTrackbar('S_Low', 'Control Panel', 100, 255, nothing)
cv2.createTrackbar('V_Low', 'Control Panel', 100, 255, nothing)

init_hsv = [3, 50, 51]
cv2.setTrackbarPos('H_Low', 'Control Panel', init_hsv[0])
cv2.setTrackbarPos('S_Low', 'Control Panel', init_hsv[1])
cv2.setTrackbarPos('V_Low', 'Control Panel', init_hsv[2])

# 초기에 자동 기능 끄기
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # 0.25가 보통 '수동' 모드

while True:
    # 트랙바 값 읽기
    focus = cv2.getTrackbarPos('Focus', 'Control Panel')
    exposure = cv2.getTrackbarPos('Exposure', 'Control Panel') - 10 # -10 ~ 0 범위로 변환
    h_l = cv2.getTrackbarPos('H_Low', 'Control Panel')
    s_l = cv2.getTrackbarPos('S_Low', 'Control Panel')
    v_l = cv2.getTrackbarPos('V_Low', 'Control Panel')

    # 카메라 설정 적용
    cap.set(cv2.CAP_PROP_FOCUS, focus)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

    ret, frame = cap.read()
    if not ret:
        break

    # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    # 이미지 처리 (노란색 추출)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([h_l, s_l, v_l])
    upper_yellow = np.array([35, 255, 255])
    
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    
    # 윤곽선 검출 및 그리기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None:
        continue

    cnt = max(contours, key=cv2.contourArea)
    # for cnt in contours:
    if cv2.contourArea(cnt) > 500:
        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

    outer, inner = get_safe_triangles(cnt)
    cv2.drawContours(frame, [outer], -1, (0, 0, 255), 2)
    cv2.drawContours(frame, [inner], -1, (255, 0, 0), 2)

    centroid = getConterPoints(inner)
    cv2.circle(frame, centroid, 8, (0, 255, 255), -1)

    edge_vectors = inner - np.roll(inner, -1, axis=0)
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    diffs = np.abs(edge_lengths - np.roll(edge_lengths, 1))
    apex_idx = np.argmin(diffs)
    apex = inner[apex_idx]
    cv2.circle(frame, tuple(apex.astype(int)), 8, (255, 0, 255), -1)
    cv2.line(frame, centroid, apex, (0, 0, 255), 2)


    # 방향 벡터 (중심 -> 꼬다리)
    dx, dy = apex - centroid

    # np.arctan2(x, -y)를 사용하면 12시 방향이 0도, 시계방향으로 증가하는 각도가 나옵니다.
    # 이미지 좌표계는 y가 아래로 증가하므로 -dy를 해줘야 수학적 위쪽 방향이 됩니다.
    angle = np.degrees(np.arctan2(dx, -dy))

    # -180~180 범위를 0~360 범위로 변환
    angle = (angle + 360) % 360

    print(f"꼬다리 좌표: {apex}")
    print(f"최종 회전 각도: {angle:.2f}도")

    text = f"Angle: {angle:.2f} deg"

    # 폰트 설정 및 텍스트 그리기
    # cv2.putText(이미지, 내용, 좌표, 폰트, 크기, 색상, 두께, 선타입)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (centroid[0] + 10, centroid[1] - 10), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    merged = np.hstack((mask, frame))
    # 화면 표시
    cv2.imshow('Control Panel', merged) # 마스크 화면을 컨트롤 판넬에 표시
    # cv2.imshow('Real-time Result', frame)


    key = cv2.waitKey(1) & 0xFF
    # 'r' 키를 누르면 녹화 상태 반전
    if key == ord('r'):
        is_recording = not is_recording
        
        if is_recording:
            # 녹화 시작: 파일명, 코덱, FPS, 해상도 설정
            h, w = frame.shape[:2]
            out = cv2.VideoWriter('output.avi', fourcc, 20.0, (w, h))
            print("녹화를 시작합니다...")
        else:
            # 녹화 중지: 객체 해제
            if out is not None:
                out.release()
                out = None
            print("녹화가 저장되었습니다.")

    # 녹화 중일 때 프레임을 파일에 쓰기
    if is_recording and out is not None:
        out.write(frame)
        # 화면에 녹화 중임을 표시 (빨간색 원)
        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 화면 출력
    cv2.imshow('Recording Tool', frame)

    # 'q' 키를 누르면 종료
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()