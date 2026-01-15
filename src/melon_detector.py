import cv2
import numpy as np
import math

def draw_tcp_axis(img, center, angle_deg, length=50):
    cx, cy = center
    # 그리기 위해 좌표계 변환 (상단 0도 -> 우측 0도)
    draw_angle_deg = angle_deg - 90
    angle_rad = math.radians(draw_angle_deg)
    
    # X축 (Red)
    x1 = int(cx + length * math.cos(angle_rad))
    y1 = int(cy + length * math.sin(angle_rad))
    
    # Y축 (Green)
    y_angle_rad = angle_rad + math.pi / 2
    x2 = int(cx + length * math.cos(y_angle_rad))
    y2 = int(cy + length * math.sin(y_angle_rad))
    
    cv2.line(img, (cx, cy), (x2, y2), (0, 255, 0), 3)
    cv2.line(img, (cx, cy), (x1, y1), (0, 0, 255), 3)
    cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)
    
    cv2.putText(img, "X", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, "Y", (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def get_fast_largest_triangle(contour):
    hull = cv2.convexHull(contour)
    hull = np.squeeze(hull)
    if hull.ndim == 1: hull = hull.reshape(-1, 2)
    n = len(hull)
    if n < 3: return None, None, None

    max_area = 0
    best_tri = None

    for i in range(n):
        k = (i + 2) % n
        for j in range(i + 1, n):
            while True:
                area_current = abs(np.cross(hull[j]-hull[i], hull[k]-hull[i]))
                next_k = (k + 1) % n
                area_next = abs(np.cross(hull[j]-hull[i], hull[next_k]-hull[i]))
                if area_next > area_current: k = next_k
                else: break
            if area_current > max_area:
                max_area = area_current
                best_tri = (hull[i], hull[j], hull[k])

    if best_tri is None: return None, None, None
    A, B, C = best_tri
    centroid = np.mean([A, B, C], axis=0)
    dists = [(np.linalg.norm(A-B), C), (np.linalg.norm(B-C), A), (np.linalg.norm(C-A), B)]
    top_vertex = min(dists, key=lambda x: x[0])[1]
    return best_tri, centroid, top_vertex

def get_angle(cx, cy, tx, ty):
    # [각도 계산 보정]
    dx = tx - cx
    dy = ty - cy

    # 1. 기본 계산 (-180 ~ 180)
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    # 2. 상단 0도 보정 (+90) 후 360 모듈러 연산으로 양수화
    final_angle = (angle_deg + 90) % 360

    # 3. float 변환 (안전장치)
    final_angle = float(final_angle)
    return final_angle

def detect_melon(frame, debug_frame=None):
    try:
        with open("./output/hsv.txt", "r") as f:
            lower_hsv_str = f.readline().strip()
            upper_hsv_str = f.readline().strip()
            h_low, s_low, v_low = map(int, lower_hsv_str.split(' '))
            h_high, s_high, v_high = map(int, upper_hsv_str.split(' '))
            lower_yellow = np.array([h_low, s_low, v_low])
            upper_yellow = np.array([h_high, s_high, v_high])
    except:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    scale_ratio = 0.25
    small_frame = cv2.resize(hsv, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_NEAREST)
    
    blurred = cv2.GaussianBlur(small_frame, (5, 5), 0)
    mask = cv2.inRange(blurred, lower_yellow, upper_yellow)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 가장 큰 contour 하나만 선택
    c = max(contours, key=cv2.contourArea)

    # 면적이 너무 작으면 무시
    if cv2.contourArea(c) < 200:
        return None

    best_tri, centroid_small, top_vertex_small = get_fast_largest_triangle(c)
    if best_tri is None:
        return None

    cx = int(centroid_small[0] / scale_ratio)
    cy = int(centroid_small[1] / scale_ratio)
    tx = int(top_vertex_small[0] / scale_ratio)
    ty = int(top_vertex_small[1] / scale_ratio)

    if debug_frame is not None:
        tri_pts = []
        for pt in best_tri:
            tri_pts.append([int(pt[0] / scale_ratio), int(pt[1] / scale_ratio)])
        tri_cnt = np.array(tri_pts).reshape((-1, 1, 2))
        cv2.drawContours(debug_frame, [tri_cnt], 0, (255, 200, 200), 3)
        cv2.line(debug_frame, (cx, cy), (tx, ty), (200, 255, 200), 3)
        cv2.circle(debug_frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        cv2.circle(debug_frame, (int(tx), int(ty)), 5, (0, 0, 255), -1)

        # draw_tcp_axis(debug_frame, (real_cx, real_cy), final_angle, length=60)

        # cv2.putText(debug_frame, f"{final_angle:.0f} deg", (real_cx - 40, real_cy - 40),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return (cx, cy, tx, ty)
    # [각도 계산 보정]
    # dx = real_tx - real_cx
    # dy = real_ty - real_cy

    # # 1. 기본 계산 (-180 ~ 180)
    # angle_rad = math.atan2(dy, dx)
    # angle_deg = math.degrees(angle_rad)

    # # 2. 상단 0도 보정 (+90) 후 360 모듈러 연산으로 양수화
    # final_angle = (angle_deg + 90) % 360

    # # 3. float 변환 (안전장치)
    # final_angle = float(final_angle)

    # # 시각화 (debug_frame이 제공된 경우에만 그림)


    # # 가장 큰 객체 하나에 대한 결과만 튜플로 반환
    # return (real_cx, real_cy, final_angle)