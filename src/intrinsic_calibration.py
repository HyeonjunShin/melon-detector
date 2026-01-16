from device.kinect_camera import Camera
import cv2
import numpy as np

class Calibration:
    def __init__(self, camera):
        self.camera = camera
        self.board_size = (8, 6)
        self.square_size = 0.03 # 30mm
        self.min_distance = 100 # px
        self.save_dir = "output/"
        
    def start(self):
        last_pos = 0
        saved_count = 0
        while True:
            frame = self.camera.getFrame().color
            if frame is None:
                continue
            
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            found, corners = cv2.findChessboardCorners(frame_gray, self.board_size, None)

            if found:
                # 현재 체스판의 중심점 계산
                current_pos = corners.mean(axis=0)[0]
                
                # 1. 처음 캡처하거나, 이전 위치보다 일정 거리 이상 움직였을 때 캡처
                if last_pos is None or np.linalg.norm(current_pos - last_pos) > self.min_distance:
                    # 2. (옵션) 이미지 선명도 체크
                    variance = cv2.Laplacian(frame_gray, cv2.CV_64F).var()
                    
                    if variance > 100: # 선명도 기준값
                        saved_count += 1
                        last_pos = current_pos
                        cv2.imwrite(f'calib_{saved_count}.jpg', frame)
                        print(f"이미지 저장됨: {saved_count}")

                # 화면에 코너 표시
                cv2.drawChessboardCorners(frame, self.board_size, corners, found)

            cv2.imshow('Calibration', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cv2.destroyAllWindows()



if __name__ == "__main__":
    camera = Camera()
    camera.start()

    calibration = Calibration(camera)
    calibration.start()

    camera.stop()