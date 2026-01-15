import cv2
import numpy as np

class KFFIlter:
    def __init__(self, process_noise_cov=1.0, measurement_noise_cov=1.0) -> None:
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0], 
            [0, 1, 0, 1], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]
            ], np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0], 
            [0, 1, 0, 0]
            ], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise_cov
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise_cov

        self.prev_time = None

    def update(self, cx, cy, curtime):
        # First time initialization
        if self.prev_time is None:
            # Cannot initialize without a measurement
            if cx is None or cy is None:
                return None, None
            
            self.kf.statePost = np.array([
                [cx], [cy], [0], [0]
                ], dtype=np.float32)
            self.prev_time = curtime
            return int(cx), int(cy)
        
        # --- For subsequent updates ---
        
        # Calculate dt
        dt = (curtime - self.prev_time) / 1_000_000.0
        self.prev_time = curtime
        
        # Update transition matrix with dt
        self.kf.transitionMatrix[0, 2] = dt
        self.kf.transitionMatrix[1, 3] = dt

        # Always predict the next state
        predict = self.kf.predict()
        
        # Only correct if a measurement is available
        if cx is not None and cy is not None:
            measurement = np.array([[cx], [cy]], dtype=np.float32)
            self.kf.correct(measurement)
        else:
            # 측정값이 없으면, 예측된 상태가 우리의 최선의 추정치입니다.
            # statePost를 예측된 상태(statePre)로 수동 업데이트해야 합니다.
            self.kf.statePost = self.kf.statePre.copy()

        # 업데이트된 최신 추정 위치(statePost에서)를 반환합니다.
        return int(self.kf.statePost[0, 0]), int(self.kf.statePost[1, 0])
        
    def predict(self):
        predict = self.kf.predict()
        return predict[0].item(), predict[1].item()

    def reset(self):
        """필터를 초기 상태로 리셋합니다."""
        self.prev_time = None



class MelonPosFilter:
    def __init__(self) -> None:
        self.center = KFFIlter()
        self.tail = KFFIlter()

    def update(self, cx, cy, tx, ty, curtime):
        self.center.update(cx, cy, curtime)
        self.tail.update(tx, ty, curtime)

    def predict(self):
        cx, cy = self.center.predict()
        tx, ty = self.tail.predict()
        return cx, cy, tx, ty
    
    def reset(self):
        self.center.reset()
        self.tail.reset()





    