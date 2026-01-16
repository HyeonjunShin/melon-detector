from pyk4a import PyK4APlayback, CalibrationType
from data_types import Frame
import cv2

class Playback:
    def __init__(self, filepath="./data/recored_files/1080p.mkv"):
        self.playback = PyK4APlayback(filepath)
        self.playback.open()
    
        # The camera parameters
        self.WIDTH = 1920
        self.HEIGHT = 1080
        self.K = self.playback.calibration.get_camera_matrix(CalibrationType.COLOR)
        self.D = self.playback.calibration.get_distortion_coefficients(CalibrationType.COLOR)

        # self.timestamps_image = []
        # self.timestamps_depth = []
        # while True:
        #     try:
        #         capture = self.playback.get_next_capture()
        #         if capture.color is not None:
        #             self.timestamps_image.append(capture._color_timestamp_usec)
        #         if capture.depth is not None:
        #             self.timestamps_depth.append(capture._depth_timestamp_usec)
        #     except EOFError as e:
        #         print(e)
        #         break

        # num_image = len(self.timestamps_image)
        # if num_image == 0:
        #     print("프레임을 찾을 수 없습니다.")
        #     return
        # else:
        #     print(num_image, "프레임을 찾었습니다.")


        # num_depth = len(self.timestamps_depth)
        # if num_depth == 0:
        #     print("프레임을 찾을 수 없습니다.")
        #     return
        # else:
        #     print(num_depth, "프레임을 찾었습니다.")

        # 타임스탬프를 모두 읽은 후, 파일 포인터를 다시 처음으로 되돌립니다.
        # self.playback.seek(0)

        # self.cur_idx = 0

        while True:
            capture = self.playback.get_next_capture()
            if capture.color is not None and capture.depth is not None:
                break
        
        self.frame = Frame(capture.color, capture.transformed_depth, capture.color_timestamp_usec)

    def getFrame(self):
        # # depth = capture.depth
        # # depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # # cv2.imshow("depth", depth_normalized)

        # # return frame, depth, timestamp_frame
        return self.frame
    
    def next(self):
        while True:
            capture = self.playback.get_next_capture()
            if capture.color is not None and capture.depth is not None:
                break
        self.frame = Frame(capture.color, capture.transformed_depth, capture.color_timestamp_usec)
    
    def prev(self):
        while True:
            capture = self.playback.get_previous_capture()
            if capture.color is not None and capture.depth is not None:
                break
        self.frame = Frame(capture.color, capture.transformed_depth, capture.color_timestamp_usec)
    
    def getImageAt(self, idx):
        self.playback.seek(self.timestamps_image[idx])
        capture = self.playback.get_next_capture()
        frame = capture.color
        return frame
    
    def close(self):
        self.playback.close()
    

if __name__ == "__main__":
    device = Playback("./data/recored_files/1080p.mkv")
    while True:
        res_frame = device.getImage()
        if res_frame is None: continue
        frame = res_frame.image
        depth = res_frame.depth

        depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

        alpha = 0.5  # 원본 컬러 이미지의 불투명도 (60%)
        beta = 0.5  # Depth 맵의 불투명도 (40%)
        
        overlay_result = cv2.addWeighted(frame, alpha, 
                                        depth, beta, 0)

        cv2.imshow("image", overlay_result)
        key = cv2.waitKeyEx(1)
        if key == ord('q') or key == 27:
            break

    device.close()
    cv2.destroyAllWindows()