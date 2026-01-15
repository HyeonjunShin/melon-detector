import pyk4a
from pyk4a import Config, PyK4A
import open3d as o3d
import numpy as np
import cv2

def main():
    # 1. Azure Kinect 설정 (스캔용 고화질 설정)
    config = Config(
        color_resolution=pyk4a.ColorResolution.RES_720P, # 실시간 성능을 위해 720p 권장
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        camera_fps=pyk4a.FPS.FPS_15,
        synchronized_images_only=True,
    )
    k4a = PyK4A(config)
    k4a.start()

    # 2. Open3D Visualizer 초기화
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Azure Kinect Real-time PCD", width=1280, height=720)

    # 3. 카메라 파라미터(Intrinsics) 가져오기
    # Open3D가 2D 이미지를 3D 공간으로 투영하려면 카메라 렌즈 정보가 필요함
    matrix = k4a.calibration.get_camera_matrix(pyk4a.CalibrationType.COLOR)
    # 해상도에 맞는 Intrinsic 객체 생성
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        1280, 720, # 720p 해상도 (1280x720)
        matrix[0,0], matrix[1,1], # fx, fy
        matrix[0,2], matrix[1,2]  # cx, cy
    )

    # 포인트 클라우드 객체 미리 생성 (빈 껍데기)
    pcd = o3d.geometry.PointCloud()
    is_first_frame = True

    try:
        print("Starting real-time visualization... Press 'Q' inside the window to exit.")
        
        while True:
            # --- [데이터 획득] ---
            capture = k4a.get_capture()
            if capture.color is None or capture.depth is None:
                continue

            # 1. Depth를 Color 시점에 맞춤 (필수)
            transformed_depth = capture.transformed_depth

            # 2. Open3D용 이미지로 변환
            # Color: BGRA -> RGB 변환
            color_frame = capture.color[:, :, :3]
            color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
            
            o3d_color = o3d.geometry.Image(color_frame)
            o3d_depth = o3d.geometry.Image(transformed_depth)

            # 3. RGBD 이미지 생성
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color, 
                o3d_depth, 
                depth_scale=1000.0, # Kinect는 mm 단위이므로 m로 변환
                depth_trunc=3.0,    # 3m 이상 거리는 무시 (노이즈 제거)
                convert_rgb_to_intensity=False
            )

            # 4. RGBD -> Point Cloud 변환
            temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, intrinsic
            )

            # --- [시각화 갱신] ---
            if is_first_frame:
                # 첫 프레임: 데이터를 채우고 시각화 창에 추가
                pcd.points = temp_pcd.points
                pcd.colors = temp_pcd.colors
                
                # 시점 보정 (Kinect는 상하반전되어 보일 수 있음 -> 뒤집기)
                # X축 180도 회전
                pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                
                vis.add_geometry(pcd)
                
                # 초기 시점 설정 (Reset View)
                vis.reset_view_point(True)
                is_first_frame = False
            else:
                # 이후 프레임: 점들의 위치와 색상만 업데이트 (객체를 지우고 다시 만들면 느림)
                pcd.points = temp_pcd.points
                pcd.colors = temp_pcd.colors
                
                # 똑같이 뒤집어줘야 함 (매 프레임 데이터가 새로 들어오므로)
                pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

                # 변경사항 반영
                vis.update_geometry(pcd)

            # 렌더링 업데이트 (이 함수들이 있어야 화면이 움직임)
            keep_running = vis.poll_events()
            vis.update_renderer()
            
            if not keep_running:
                break

    except KeyboardInterrupt:
        pass
    finally:
        k4a.stop()
        vis.destroy_window()
        print("Visualization stopped.")

if __name__ == "__main__":
    main()