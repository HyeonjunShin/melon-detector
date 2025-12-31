from pyk4a import PyK4APlayback, CalibrationType
import cv2
import numpy as np
from aruco_detector import ArUcoDetector


def main():
    playback = PyK4APlayback("./recored_files/1080p.mkv")
    playback.open()

    # The camera parameters
    WIDTH = 1920
    HEIGHT = 1080
    K = playback.calibration.get_camera_matrix(CalibrationType.COLOR)
    C = playback.calibration.get_distortion_coefficients(CalibrationType.COLOR)

    # Initalize the detectors
    aruco_detector = ArUcoDetector(K, C, 0.025)
    melon_detector = MelonDetector()
    
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(K, C, (WIDTH, HEIGHT), 0, (WIDTH, HEIGHT))
    mapx, mapy = cv2.initUndistortRectifyMap(K, C, None, new_mtx, (WIDTH, HEIGHT), cv2.CV_32FC1)

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
        frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

        aruco_detector.detect(frame)

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
