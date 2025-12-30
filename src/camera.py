from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS, ImageFormat, CalibrationType, ColorControlCommand, ColorControlMode
import cv2
import numpy as np

class Camera:
    def __init__(self):
        self.k4a = PyK4A(
            Config(
                color_resolution=ColorResolution.RES_1080P,
                color_format=ImageFormat.COLOR_BGRA32,
                depth_mode=DepthMode.WFOV_2X2BINNED,
                synchronized_images_only=False,
                camera_fps=FPS.FPS_30,
            )
        )


        if self.k4a._config.color_resolution == ColorResolution.RES_720P:
            self.w = 1280
            self.h = 720
        elif self.k4a._config.color_resolution == ColorResolution.RES_1080P:
            self.w = 1920
            self.h = 1080

    def start(self):
        try:
            self.k4a.start()
            print("Strat the Kinect camera.")
        except Exception as e:
            print(f"Error: {e}")
            return
        
        self.k4a._set_color_control(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE, 16670, ColorControlMode.MANUAL)
        self.k4a._set_color_control(ColorControlCommand.WHITEBALANCE, 4500, ColorControlMode.MANUAL)
        
        self.K_image = self.k4a.calibration.get_camera_matrix(CalibrationType.COLOR)
        self.coff_image = self.k4a.calibration.get_distortion_coefficients(CalibrationType.COLOR)

        self.new_mtx, roi = cv2.getOptimalNewCameraMatrix(self.K_image, self.coff_image, (self.w, self.h), 0, (self.w, self.h))
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.K_image, self.coff_image, None, self.new_mtx, (self.w, self.h), cv2.CV_32FC1)

        for target in ColorControlCommand:
            print(target, self.k4a._get_color_control(target))

    def getExposure(self):
        return self.k4a._get_color_control(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE)[0]
    def setExposure(self, value):
        print(self.getExposure())
        return self.k4a._set_color_control(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE, value, ColorControlMode.MANUAL)

    def getBrightness(self):
        return self.k4a._get_color_control(ColorControlCommand.BRIGHTNESS)[0]
    def setBrightness(self, value):
        print(self.getBrightness())
        return self.k4a._set_color_control(ColorControlCommand.BRIGHTNESS, value, ColorControlMode.MANUAL)

    def getContrast(self):
        return self.k4a._get_color_control(ColorControlCommand.CONTRAST)[0]
    def setContrast(self, value):
        print(self.getContrast())
        return self.k4a._set_color_control(ColorControlCommand.CONTRAST, value, ColorControlMode.MANUAL)
    
    def getSaturation(self):
        return self.k4a._get_color_control(ColorControlCommand.SATURATION)[0]
    def setSaturation(self, value):
        print(self.getSaturation())
        return self.k4a._set_color_control(ColorControlCommand.SATURATION, value, ColorControlMode.MANUAL)
        
    def getWhiteBalance(self):
        return self.k4a._get_color_control(ColorControlCommand.WHITEBALANCE)[0]
    def setWhiteBalance(self, value):
        print(self.getWhiteBalance())
        return self.k4a._set_color_control(ColorControlCommand.WHITEBALANCE, value, ColorControlMode.MANUAL)
            
    def getSharpness(self):
        return self.k4a._get_color_control(ColorControlCommand.SHARPNESS)[0]
    def setSharpness(self, value):
        print(self.getSharpness())
        return self.k4a._set_color_control(ColorControlCommand.SHARPNESS, value, ColorControlMode.MANUAL)
    
    def getBlackLightCompensation(self):
        return self.k4a._get_color_control(ColorControlCommand.BACKLIGHT_COMPENSATION)[0]
    def setBlackLightCompensation(self, value):
        print(self.getBlackLightCompensation())
        return self.k4a._set_color_control(ColorControlCommand.BACKLIGHT_COMPENSATION, value, ColorControlMode.MANUAL)
    
    def getPowerLineFrequency(self):
        return self.k4a._get_color_control(ColorControlCommand.POWERLINE_FREQUENCY)[0]
    def setPowerLineFrequency(self, value):
        print(self.getPowerLineFrequency())
        return self.k4a._set_color_control(ColorControlCommand.POWERLINE_FREQUENCY, value, ColorControlMode.MANUAL)
    
    def getGain(self):
        return self.k4a._get_color_control(ColorControlCommand.GAIN)[0]
    def setGain(self, value):
        print(self.getGain())
        return self.k4a._set_color_control(ColorControlCommand.GAIN, value, ColorControlMode.MANUAL)


    def stop(self):
        self.k4a.stop()
        print("Stop the Kinect camera.")
    
    def getFrame(self):
        frame = self.k4a.get_capture().color
        if frame is None:
            return None
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
        return frame

clicked_points = []
width, height = 800, 800
dst_pts = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append([x, y])
            print(f"점 추가: ({x}, {y}) - 현재 {len(clicked_points)}개")
        
        # 4개가 다 모이면 변환 프로세스 실행 (선택 사항)
        if len(clicked_points) == 4:
            print("4개의 점이 모두 선택되었습니다. 'Enter'를 누르세요.")

if __name__ == "__main__":
    camera = Camera()
    camera.start()
    # CHECKERBOARD = (5, 7)

    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("window", 3840//2, 2160//2)
    cv2.setMouseCallback("window", mouse_callback)
    while True:
        frame = camera.getFrame()
        if frame is None:
            continue

        for p in clicked_points:
                cv2.circle(frame, tuple(p), 5, (0, 0, 255), -1)
        cv2.imshow("window", frame)

        key = cv2.waitKey(1) & 0xFF
        # 4점을 다 찍고 Enter를 누르면 투영 변환 실행
        if len(clicked_points) == 4:
            src_pts = np.float32(clicked_points)
            # 변환 행렬 계산
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            # 원근 변환 적용
            warped = cv2.warpPerspective(frame, matrix, (width, height))
            
            cv2.imshow("Real-time Warped", warped)
                
        # 'r'을 누르면 좌표 초기화
        if key == ord('r'):
            clicked_points = []
            print("좌표가 초기화되었습니다.")

        # 'q'를 누르면 종료
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    #     # frame_color = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    #     # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    #     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR).astype(np.uint8)
    #     frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    #     found, corners = cv2.findChessboardCorners(frame_gray, (5,7), 
    #                                                cv2.CALIB_CB_ADAPTIVE_THRESH + 
    #                                                cv2.CALIB_CB_FAST_CHECK + 
    #                                                cv2.CALIB_CB_NORMALIZE_IMAGE)

    #     ret, corners = cv2.findChessboardCorners(
    #                 frame_gray, 
    #                 CHECKERBOARD, 
    #                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
    #             )

    #     if ret:
    #         # 코너 정밀화
    #         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #         corners2 = cv2.cornerSubPix(frame_gray, corners, (11, 11), (-1, -1), criteria)
    #         cv2.drawChessboardCorners(frame_bgr, CHECKERBOARD, corners2, ret)
    #         print("Found corners!")
    #     else:
    #         # 못 찾았을 때 현재 프레임 상태 확인용 메시지
    #         cv2.putText(frame_bgr, "Searching (5x7)...", (30, 30), 
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # cv2.imshow("window", frame)

        # # 'q' 키를 누르면 종료
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    camera.stop()
    # cv2.destroyAllWindows()

    # print("Program finished.")