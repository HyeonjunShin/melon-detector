import cv2
import numpy as np
from uon_camera.camera_kinect import KinectCamera


def init_window():
    def nothing(x):
        pass

    cv2.namedWindow(
        "window", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED
    )
    cv2.resizeWindow("window", 1280, 720)

    # 트랙바 설정 (기존과 동일)
    cv2.createTrackbar("H_Low", "window", 0, 255, nothing)
    cv2.createTrackbar("S_Low", "window", 0, 255, nothing)
    cv2.createTrackbar("V_Low", "window", 0, 255, nothing)
    cv2.createTrackbar("H_High", "window", 0, 255, nothing)
    cv2.createTrackbar("S_High", "window", 0, 255, nothing)
    cv2.createTrackbar("V_High", "window", 0, 255, nothing)

    # 초기값 설정
    init_low_hsv = [3, 50, 51]
    cv2.setTrackbarPos("H_Low", "window", init_low_hsv[0])
    cv2.setTrackbarPos("S_Low", "window", init_low_hsv[1])
    cv2.setTrackbarPos("V_Low", "window", init_low_hsv[2])

    init_high_hsv = [35, 255, 255]
    cv2.setTrackbarPos("H_High", "window", init_high_hsv[0])
    cv2.setTrackbarPos("S_High", "window", init_high_hsv[1])
    cv2.setTrackbarPos("V_High", "window", init_high_hsv[2])


if __name__ == "__main__":
    camera = KinectCamera()
    camera.start()
    init_window()

    while True:
        frame = camera.getFrame()
        if frame is None:
            continue
        image, depth, timestamp = frame

        # 원본 이미지 복사 (결과 표시용)
        result_img = image.copy()

        h_l = cv2.getTrackbarPos("H_Low", "window")
        s_l = cv2.getTrackbarPos("S_Low", "window")
        v_l = cv2.getTrackbarPos("V_Low", "window")
        h_h = cv2.getTrackbarPos("H_High", "window")
        s_h = cv2.getTrackbarPos("S_High", "window")
        v_h = cv2.getTrackbarPos("V_High", "window")

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([h_l, s_l, v_l]), np.array([h_h, s_h, v_h]))

        # 1. 노이즈 제거 (Opening)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # 2. 확실한 배경(Sure Background) 확보 - 팽창(Dilate)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # 3. 확실한 전경(Sure Foreground) 확보 - 거리 변환(Distance Transform)
        # 참외의 중심부일수록 값이 큼. 붙어있는 부분은 값이 작음.
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

        # 거리값의 50% 이상인 지점만 확실한 참외의 '핵'으로 간주 (이 수치를 조절하여 민감도 변경)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # 4. 불확실한 영역(Unknown Region) 확보
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 5. 마커(Marker) 생성
        # 서로 떨어진 '핵'들에 번호를 매김
        ret, markers = cv2.connectedComponents(sure_fg)

        # 배경은 0이 아니라 1로 설정 (Watershed 알고리즘 규칙)
        markers = markers + 1

        # 불확실한 영역은 0으로 설정
        markers[unknown == 255] = 0

        # 6. 워터쉐드 알고리즘 실행
        markers = cv2.watershed(image, markers)

        # 워터쉐드 결과 경계선은 -1로 표시됨 -> 붉은색으로 표시
        # image[markers == -1] = [0, 0, 255]

        # 7. 분리된 각 객체에 대해 루프 돌며 정보 추출
        # markers의 고유값들을 확인 (배경인 1과 경계선인 -1은 제외)
        unique_markers = np.unique(markers)

        for marker_id in unique_markers:
            if marker_id <= 1:  # 배경(1)이나 경계선(-1)은 건너뜀
                continue

            # 현재 마커 ID에 해당하는 영역만 마스크로 추출
            # dtype을 uint8로 변환해야 findContours 사용 가능
            temp_mask = np.zeros(markers.shape, dtype=np.uint8)
            temp_mask[markers == marker_id] = 255

            contours, _ = cv2.findContours(
                temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                cnt = contours[0]  # 마커별로 윤곽선은 하나씩 나옴

                # 면적 필터링 (너무 작은 조각 무시)
                if cv2.contourArea(cnt) < 500:
                    continue

                # --- (기존 로직) 중심점 및 타원 그리기 ---
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(result_img, (cx, cy), 5, (255, 0, 0), -1)

                if len(cnt) >= 5:
                    ellipse = cv2.fitEllipse(cnt)
                    cv2.ellipse(result_img, ellipse, (0, 255, 0), 2)

                    (center_e, axes, angle) = ellipse
                    major_axis_len = max(axes)
                    ang_rad = np.deg2rad(angle)

                    x1 = int(center_e[0] + (major_axis_len / 2) * np.sin(ang_rad))
                    y1 = int(center_e[1] - (major_axis_len / 2) * np.cos(ang_rad))
                    x2 = int(center_e[0] - (major_axis_len / 2) * np.sin(ang_rad))
                    y2 = int(center_e[1] + (major_axis_len / 2) * np.cos(ang_rad))

                    cv2.line(result_img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        cv2.imshow("window", result_img)
        # cv2.imshow("mask", sure_fg) # 디버깅용: 참외의 '핵'이 잘 잡히는지 확인할 때 주석 해제

        key = cv2.waitKeyEx(1)
        if key == ord("q") or key == 27:
            break

    camera.stop()
    cv2.destroyAllWindows()
