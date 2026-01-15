from pyk4a import PyK4APlayback, CalibrationType
import cv2
from aruco_detector import ArUcoDetector
from melon_detector import MelonDetector

def main():
    playback = PyK4APlayback("./recored_files/1080p.mkv")
    playback.open()

    # The camera parameters
    WIDTH = 1920
    HEIGHT = 1080
    K = playback.calibration.get_camera_matrix(CalibrationType.COLOR)
    C = playback.calibration.get_distortion_coefficients(CalibrationType.COLOR)

    aruco_detector = ArUcoDetector(K, C, 0.025)
    melon_detector = MelonDetector()


    timestamps_image = []
    timestamps_depth = []
    while True:
        try:
            capture = playback.get_next_capture()
            if capture.color is not None:
                timestamps_image.append(capture._color_timestamp_usec)
            if capture.depth is not None:
                timestamps_depth.append(capture._depth_timestamp_usec)
        except EOFError as e:
            print(e)
            break

    total_frames = len(timestamps_image)
    if total_frames == 0:
        print("프레임을 찾을 수 없습니다.")
        return
    else:
        print(total_frames, "프레임을 찾었습니다.")

    cur_idx = 0
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", (1920, 1080))
    while True:
        print(cur_idx)
        playback.seek(timestamps_image[cur_idx])
        capture = playback.get_next_capture()
        frame = capture.color

        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        aruco_detector.detect(frame)
        melon_detector.detect(frame)

        cv2.imshow("frame", frame)

        key = cv2.waitKeyEx(33)  
        if key == ord('q') or key == 27:
            break
        elif key == 65363:
            cur_idx = cur_idx + 1
        elif key == 65361:
            cur_idx = cur_idx - 1


    playback.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()