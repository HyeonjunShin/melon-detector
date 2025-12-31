from pyk4a import PyK4APlayback, CalibrationType
import cv2
import numpy as np

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

def main():
    playback = PyK4APlayback("./recored_files/1080p.mkv")
    playback.open()

    K_img = playback.calibration.get_camera_matrix(CalibrationType.COLOR)
    coff_img = playback.calibration.get_distortion_coefficients(CalibrationType.COLOR)

    print(K_img)
    print(coff_img)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    obj_points = np.array([
        [0, 0, 0],               # 좌상 (Corner 0)
        [marker_size, 0, 0],   # 우상 (Corner 1)
        [marker_size, -marker_size, 0], # 우하 (Corner 2)
        [0, -marker_size, 0]   # 좌하 (Corner 3)
    ], dtype=np.float32)

    # marker_ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)

    # marker_points = [
    #     get_x_aligned_corners(0), # ID 0
    #     get_x_aligned_corners(1), # ID 1
    #     get_x_aligned_corners(2),  # ID 2
    #     get_x_aligned_corners(3),  # ID 2
    #     get_x_aligned_corners(4)  # ID 2
    # ]

    # board = cv2.aruco.Board(marker_points, aruco_dict, marker_ids)
    

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", (1920, 1080))
    while True:
        capture = playback.get_next_capture()
        frame = capture.color
        depth = capture.depth

        if depth is None:
            continue
        depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_VIRIDIS)


        if frame is None:
            continue
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for i in range(len(ids)):
                _, rvec, tvec = cv2.solvePnP(obj_points, corners[i], K_img, coff_img)
                cv2.drawFrameAxes(frame, K_img, coff_img, rvec, tvec, 0.03)
            #     cv2.drawFrameAxes(frame, K_image, coff_image, rvec, tvec, 0.03) 
            #     cv2.putText(frame, f"ID:{ids[i][0]}", tuple(corners[i][0][0].astype(int)), 
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # if ids is not None:
        #     objPoints, imgPoints = board.matchImagePoints(corners, ids)
        #     if len(objPoints) > 0:
        #         success, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, K_img, coff_img)
        #         if success:
        #             # 보드의 원점(첫 마커 좌상단)에 축 그리기 (10cm 길이)
        #             cv2.drawFrameAxes(frame, K_img, coff_img, rvec, tvec, 0.1)
                    
        #             # 좌표값 화면에 텍스트 표시
        #             pos_str = f"X:{tvec[0][0]:.2f} Y:{tvec[1][0]:.2f} Z:{tvec[2][0]:.2f}"
        #             cv2.putText(frame, pos_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        cv2.imshow("frame", frame)
        cv2.imshow("Depth", depth)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    playback.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
