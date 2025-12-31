from camera import Camera
import cv2
import numpy as np

def nothing(x):
    pass


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



if __name__ == "__main__":
    camera = Camera()
    camera.start()

    cv2.namedWindow('Kinect camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Kinect camera', 3840//2, 2160//2)

    # cv2.createTrackbar('Exposure', 'Kinect camera', 0, 16670, lambda pos : camera.setExposure(pos))
    # cv2.setTrackbarPos('Exposure', 'Kinect camera', camera.getExposure())

    # cv2.createTrackbar('Brightness', 'Kinect camera', 0, 255, lambda pos : camera.setBrightness(pos))
    # cv2.setTrackbarPos('Brightness', 'Kinect camera', camera.getBrightness())

    # cv2.createTrackbar('Contrast', 'Kinect camera', 0, 255, lambda pos : camera.setContrast(pos))
    # cv2.setTrackbarPos('Contrast', 'Kinect camera', camera.getContrast())

    # cv2.createTrackbar('Saturation', 'Kinect camera', 0, 255, lambda pos : camera.setSaturation(pos))
    # cv2.setTrackbarPos('Saturation', 'Kinect camera', camera.getSaturation())

    # cv2.createTrackbar('WhiteBalance', 'Kinect camera', 0, 255, lambda pos : camera.setWhiteBalance(pos))
    # cv2.setTrackbarPos('WhiteBalance', 'Kinect camera', camera.getWhiteBalance())

    # cv2.createTrackbar('Gain', 'Kinect camera', 0, 255, lambda pos : camera.setGain(pos))
    # cv2.setTrackbarPos('Gain', 'Kinect camera', camera.getGain())

    # cv2.createTrackbar('Sharpness', 'Kinect camera', 0, 4, lambda pos : camera.setSharpness(pos))
    # cv2.setTrackbarPos('Sharpness', 'Kinect camera', camera.getSharpness())

    # cv2.createTrackbar('BlackLightCompensation', 'Kinect camera', 0, 1, lambda pos : camera.setBlackLightCompensation(pos))
    # cv2.setTrackbarPos('BlackLightCompensation', 'Kinect camera', camera.getBlackLightCompensation())

    # cv2.createTrackbar('PowerLineFrequency', 'Kinect camera', 1, 2, lambda pos : camera.setPowerLineFrequency(pos))
    # cv2.setTrackbarPos('PowerLineFrequency', 'Kinect camera', camera.getPowerLineFrequency())

    # HSV 색상 제어 트랙바 (노란색 기본값 세팅)
    cv2.createTrackbar('H_Low', 'Kinect camera', 15, 179, nothing)
    cv2.createTrackbar('S_Low', 'Kinect camera', 100, 255, nothing)
    cv2.createTrackbar('V_Low', 'Kinect camera', 100, 255, nothing)

    init_hsv = [3, 50, 51]
    cv2.setTrackbarPos('H_Low', 'Kinect camera', init_hsv[0])
    cv2.setTrackbarPos('S_Low', 'Kinect camera', init_hsv[1])
    cv2.setTrackbarPos('V_Low', 'Kinect camera', init_hsv[2])

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    marker_size = 0.025
    marker_margin = 0.013

    def get_x_aligned_corners(col_index):
        x_start = col_index * marker_margin
        # 좌상, 우상, 우하, 좌하 순서
        return np.array([
            [x_start,          marker_size, 0], 
            [x_start + marker_size, marker_size, 0], 
            [x_start + marker_size, 0,      0], 
            [x_start,          0,      0]
        ], dtype=np.float32)

    marker_ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    marker_points = [
        get_x_aligned_corners(0), # ID 0
        get_x_aligned_corners(1), # ID 1
        get_x_aligned_corners(2),  # ID 2
        get_x_aligned_corners(3),  # ID 2
        get_x_aligned_corners(4)  # ID 2
    ]


    board = cv2.aruco.Board(marker_points, aruco_dict, marker_ids)


    K_image = camera.K_image
    coff_image = camera.coff_image

    while True:
        frame = camera.getFrame()
        if frame is None:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)
        if ids is not None:
            # 검출된 마커 주변에 테두리와 ID 표시
            # cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            # print(f"검출된 마커 ID들: {ids.flatten()}")
            objPoints, imgPoints = board.matchImagePoints(corners, ids)
            if len(objPoints) > 0:
                # 6D Pose 계산 (solvePnP)
                # rvec: 회전(Rotation), tvec: 위치(Translation)
                success, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, K_image, coff_image)

                if success:
                    # 보드의 원점(첫 마커 좌상단)에 축 그리기 (10cm 길이)
                    cv2.drawFrameAxes(frame, K_image, coff_image, rvec, tvec, 0.1)
                    
                    # 좌표값 화면에 텍스트 표시
                    pos_str = f"X:{tvec[0][0]:.2f} Y:{tvec[1][0]:.2f} Z:{tvec[2][0]:.2f}"
                    cv2.putText(frame, pos_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


            # for i in range(len(ids)):
            #     # 각 마커마다 개별적으로 solvePnP 호출
            #     _, rvec, tvec = cv2.solvePnP(obj_points, corners[i], K_image, coff_image)
                
            #     cv2.drawFrameAxes(frame, K_image, coff_image, rvec, tvec, 0.03) 
            #     cv2.putText(frame, f"ID:{ids[i][0]}", tuple(corners[i][0][0].astype(int)), 
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # 2.7 1.3

        # h_l = cv2.getTrackbarPos('H_Low', 'Kinect camera')
        # s_l = cv2.getTrackbarPos('S_Low', 'Kinect camera')
        # v_l = cv2.getTrackbarPos('V_Low', 'Kinect camera')

        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # lower_yellow = np.array([h_l, s_l, v_l])
        # upper_yellow = np.array([35, 255, 255])
        
        # mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))


        # # 윤곽선 검출 및 그리기
        # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # if contours is None:
        #     continue

        # cnt = max(contours, key=cv2.contourArea)
        # # for cnt in contours:
        # if cv2.contourArea(cnt) > 500:
        #     cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

        # outer, inner = get_safe_triangles(cnt)
        # cv2.drawContours(frame, [outer], -1, (0, 0, 255), 2)
        # cv2.drawContours(frame, [inner], -1, (255, 0, 0), 2)

        # centroid = getConterPoints(inner)
        # cv2.circle(frame, centroid, 8, (0, 255, 255), -1)

        # edge_vectors = inner - np.roll(inner, -1, axis=0)
        # edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        # diffs = np.abs(edge_lengths - np.roll(edge_lengths, 1))
        # apex_idx = np.argmin(diffs)
        # apex = inner[apex_idx]
        # cv2.circle(frame, tuple(apex.astype(int)), 8, (255, 0, 255), -1)
        # cv2.line(frame, centroid, apex, (0, 0, 255), 2)

        # # 방향 벡터 (중심 -> 꼬다리)
        # dx, dy = apex - centroid

        # # np.arctan2(x, -y)를 사용하면 12시 방향이 0도, 시계방향으로 증가하는 각도가 나옵니다.
        # # 이미지 좌표계는 y가 아래로 증가하므로 -dy를 해줘야 수학적 위쪽 방향이 됩니다.
        # angle = np.degrees(np.arctan2(dx, -dy))

        # # -180~180 범위를 0~360 범위로 변환
        # angle = (angle + 360) % 360

        # print(f"꼬다리 좌표: {apex}")
        # print(f"최종 회전 각도: {angle:.2f}도")

        # text = f"Angle: {angle:.2f} deg"

        # # 폰트 설정 및 텍스트 그리기
        # # cv2.putText(이미지, 내용, 좌표, 폰트, 크기, 색상, 두께, 선타입)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(frame, text, (centroid[0] + 10, centroid[1] - 10), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # merged = np.hstack((mask, frame))
        # # 화면 표시

        # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        cv2.imshow("Kinect camera", frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
