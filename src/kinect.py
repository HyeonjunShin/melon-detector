import cv2
import numpy as np
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS, ImageFormat, CalibrationType, ColorControlCommand, ColorControlMode
import matplotlib.pyplot as plt

def main():
    # 1. NFOV_UNBINNED 최적화 설정
    k4a = PyK4A(
        Config(
            color_resolution=ColorResolution.RES_1080P,
            color_format=ImageFormat.COLOR_BGRA32,
            depth_mode=DepthMode.WFOV_2X2BINNED,
            synchronized_images_only=False,
            camera_fps=FPS.FPS_30
        )
    )

    try:
        k4a.start()
        print("Kinect NFOV Viewer 시작 (종료: 'q')")
    except Exception as e:
        print(f"장치 오픈 실패: {e}")
        return
    
    for target in ColorControlCommand:
        # k4a._set_color_control(target, ColorControlMode.MANUAL)
        print(ColorControlCommand(target))
        print(k4a._get_color_control(target))

    k4a._set_color_control(ColorControlCommand.BRIGHTNESS, ColorControlMode.MANUAL)

    K = k4a.calibration.get_camera_matrix(CalibrationType.COLOR)
    distortion = k4a.calibration.get_distortion_coefficients(CalibrationType.COLOR)

    # 시각화할 거리 범위 설정 (단위: mm)
    # 500mm(50cm) ~ 4000mm(4m) 사이를 중점적으로 시각화
    MIN_DIST = 500 
    MAX_DIST = 4000 

    save_dir = "output/"
    img_count =0
    
    cv2.namedWindow("Real-time Detection", cv2.WINDOW_NORMAL)
    while True:
        capture = k4a.get_capture()

        # RGB와 Transformed Depth가 모두 있어야 함RES_720P
        if capture.color is not None and capture.transformed_depth is not None:
            
            # [A] RGB 프레임 처리 (1080p)
            color_frame = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2BGR)
            h, w = color_frame.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, distortion, (w, h), 1, (w, h))
            undistorted_frame = cv2.undistort(color_frame, K, distortion, None, new_camera_matrix)


            # [B] 깊이 프레임 처리 (RGB 크기에 맞춰진 1080p)
            # 1. 원본 데이터 복사
            # depth_data = capture.transformed_depth.astype(float)

            # 2. 특정 범위 밖의 데이터는 제외 (Clipping)
            # 측정 불가(0)나 너무 먼 데이터는 검은색으로 처리하기 위함
            # depth_data[depth_data == 0] = MAX_DIST # 측정 pyk4a --break-system-packages불가 지역 처리
            # depth_clipped = np.clip(depth_data, MIN_DIST, MAX_DIST)

            # 3. 0~255로 정규화 (선형 매핑)
            # depth_normalized = ((depth_clipped - MIN_DIST) / (MAX_DIST - MIN_DIST) * 255).astype(np.uint8)

            # 4. 컬러맵 적용 (가까운 곳이 붉은색이 되도록 반전)
            # depth_inverted = 255 - depth_normalized
            # depth_colored = cv2.applyColorMap(depth_inverted, cv2.COLORMAP_JET)

            # 5. 측정 불가(0) 영역은 검은색 마스킹 처리
            # depth_colored[capture.transformed_depth == 0] = 0
            # [C] 화면 표시 (너무 크면 0.5배로 축소)
            # display_rgb = cv2.resize(color_frame, None, fx=0.5, fy=0.5)
            # display_depth = cv2.resize(depth_colored, None, fx=0.5, fy=0.5)

            # frame_merged = np.hstack((color_frame, undistorted_frame))
            # cv2.imshow("Real-time Detection", frame_merged)
            cv2.imshow("Real-time Detection", color_frame)
            # cv2.imshow("Depth (Transformed to RGB)", depth_colored)


        # if capture.depth is not None:
        #     # 2. 데이터 전처리
        #     depth = capture.depth
        #     h, w = depth.shape
            
        #     # 성능을 위해 샘플링 (모든 점을 다 그리면 매우 느림)
        #     # 8픽셀마다 하나씩 선택
        #     step = 8
        #     x = np.arange(0, w, step)
        #     y = np.arange(0, h, step)
        #     X, Y = np.meshgrid(x, y)
            
        #     # 해당 좌표의 Depth(Z) 값 가져오기
        #     Z = depth[Y, X]

        #     # 0(측정불가) 데이터 필터링 (그래프 왜곡 방지)
        #     mask = Z > 0
        #     X_final = X[mask]
        #     Y_final = Y[mask]
        #     Z_final = Z[mask]

        #     # 3. 3D 그래프 그리기
        #     fig = plt.figure(figsize=(10, 7))
        #     ax = fig.add_subplot(111, projection='3d')

        #     # 산점도(Scatter) 그리기
        #     # c=Z_final로 거리에 따른 색상 부여
        #     img = ax.scatter(X_final, Y_final, Z_final, c=Z_final, cmap='viridis', s=2, alpha=0.5)

        #     # 4. 축 설정
        #     ax.set_xlabel('X (Pixels)')
        #     ax.set_ylabel('Y (Pixels)')
        #     ax.set_zlabel('Depth (mm)')
        #     ax.set_title('Azure Kinect 3D Depth Point Cloud')
            
        #     # 보기 편하게 시점 조정 (뒤집힘 방지)
        #     ax.invert_yaxis()
        #     ax.view_init(elev=20, azim=45) 

        #     fig.colorbar(img, ax=ax, label='Distance (mm)')
        #     print("그래프를 출력합니다.")
        #     plt.show()

        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            img_count += 1
            filename = f"{save_dir}/calib_{img_count:02d}.jpg"
            cv2.imwrite(filename, color_frame)
            print(f"[저장 성공] {filename}")

        elif key == ord('q'):
            break

    k4a.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()