import pyk4a

def loadStereoCamera():
    cam_left = pyk4a.PyK4A(
        pyk4a.Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            color_format=pyk4a.ImageFormat.COLOR_BGRA32,
            depth_mode=pyk4a.DepthMode.OFF,
            synchronized_images_only=False,
            camera_fps=pyk4a.FPS.FPS_30,
            wired_sync_mode=pyk4a.WiredSyncMode.MASTER
        ),
        device_id=0
    )

    cam_right = pyk4a.PyK4A(
        pyk4a.Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            color_format=pyk4a.ImageFormat.COLOR_BGRA32,
            depth_mode=pyk4a.DepthMode.OFF,
            synchronized_images_only=False,
            camera_fps=pyk4a.FPS.FPS_30,
            wired_sync_mode=pyk4a.WiredSyncMode.SUBORDINATE
        ),
        device_id=1
    )

    try:
        cam_right.start()
        cam_left.start()

        print("Kinect NFOV Viewer 시작 (종료: 'q')")
    except Exception as e:
        print(f"장치 오픈 실패: {e}")
        return

    K_left = cam_left.calibration.get_camera_matrix(pyk4a.CalibrationType.COLOR)
    K_right = cam_right.calibration.get_camera_matrix(pyk4a.CalibrationType.COLOR)
    D_left = cam_left.calibration.get_distortion_coefficients(pyk4a.CalibrationType.COLOR)
    D_right = cam_right.calibration.get_distortion_coefficients(pyk4a.CalibrationType.COLOR)

    # return cam_left, cam_right, K_left, K_right, D_left, D_right
    return cam_left, cam_right


