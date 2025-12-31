import numpy as np
import cv2

import cv2
import numpy as np
import math

def get_fast_largest_triangle(contour):
    # 1. 점의 개수를 획기적으로 줄이기 위해 Convex Hull 추출
    hull = cv2.convexHull(contour)
    hull = np.squeeze(hull) # (N, 1, 2) -> (N, 2)
    
    n = len(hull)
    if n < 3: return None, None, None

    max_area = 0
    best_tri = None

    # 2. 최적화된 최대 삼각형 탐색 (O(n^2))
    # 두 점 i, j를 정하고 k를 효율적으로 이동시킴
    for i in range(n):
        k = (i + 2) % n
        for j in range(i + 1, n):
            # 현재 i, j에 대해 면적이 최대가 되는 k를 찾음
            while True:
                area_current = abs(np.cross(hull[j]-hull[i], hull[k]-hull[i]))
                next_k = (k + 1) % n
                area_next = abs(np.cross(hull[j]-hull[i], hull[next_k]-hull[i]))
                
                if area_next > area_current:
                    k = next_k
                else:
                    break
            
            if area_current > max_area:
                max_area = area_current
                best_tri = (hull[i], hull[j], hull[k])

    if best_tri is None: return None, None, None

    # 3. 무게 중심 및 '가장 긴 두 변의 교점' 계산 (최적화)
    A, B, C = best_tri
    centroid = np.mean([A, B, C], axis=0).astype(int)

    # 변의 길이 계산
    dists = [
        (np.linalg.norm(A-B), C), # AB의 대점 C
        (np.linalg.norm(B-C), A), # BC의 대점 A
        (np.linalg.norm(C-A), B)  # CA의 대점 B
    ]
    # 가장 짧은 변의 대점이 '가장 긴 두 변의 교점'
    top_vertex = min(dists, key=lambda x: x[0])[1].astype(int)

    return best_tri, tuple(centroid), tuple(top_vertex)


class MelonDetector:
    def __init__(self, hsv=np.array([3, 50, 51])):
        self.lower_yellow = hsv
        self.upper_yellow = np.array([35, 255, 255])

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        # cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        contours = max(contours, key=cv2.contourArea)
        triangle, centroid, top_vertex = get_fast_largest_triangle(contours)
        print(triangle, centroid, top_vertex)
        # 5. 결과 시각화
        if triangle is not None:
            # 윤곽선 그리기: main_contour를 리스트로 감싸서 전달
            cv2.drawContours(frame, [contours], -1, (0, 255, 0), 2)  
            
            # 삼각형 그리기
            pts = np.array(triangle, np.int32)
            cv2.polylines(frame, [pts], True, (0, 0, 255), 3)  
            
            # 무게 중심(Centroid)과 꼭짓점(Top Vertex) 표시
            cv2.circle(frame, centroid, 5, (255, 0, 0), -1)      # 파란색 점
            cv2.circle(frame, top_vertex, 7, (0, 255, 255), -1)  # 노란색 점

            # 1. 벡터 계산 (중점 -> 끝점)
            dx = top_vertex[0] - centroid[0]
            dy = top_vertex[1] - centroid[1]

            # 2. 영상 상단(12시 방향)을 0도로 설정하여 각도 계산
            # dy에 -를 붙이는 이유는 영상 좌표계에서 위쪽이 -y 방향이기 때문입니다.
            angle_rad = math.atan2(dx, -dy)
            angle_deg = math.degrees(angle_rad)

            # 3. 각도 범위를 0 ~ 360도로 변환 (선택 사항)
            if angle_deg < 0:
                angle_deg += 360

            # 4. 시각화 (중점과 끝점 연결선 및 각도 텍스트)
            cv2.line(frame, centroid, top_vertex, (255, 255, 0), 2) # 하늘색 선
            cv2.putText(frame, f"Angle: {angle_deg:.2f} deg", (centroid[0] + 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)