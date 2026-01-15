import cv2
import numpy as np
from kinect_camera import Camera
from kinect_playback import Playback
from melon_detector import detect_melon
from tracker import MelonPosFilter
import math
import time

class FPSMeter:
    def __init__(self) -> None:
        self.prev_time = 0
        self.fps = 0

    def draw(self, frame):
        curr_time = cv2.getTickCount()
        diff_time = (curr_time - self.prev_time) / cv2.getTickFrequency()
        if diff_time > 0:
            self.fps = 1.0 / diff_time
        self.prev_time = curr_time

        cv2.putText(frame, f"FPS: {self.fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
        return frame


# def detectArucoMarker(frame, K, D, marker_size, row_gap, col_gap, rows=2, cols=5, marker_type = cv2.aruco.DICT_4X4_50):
#     aruco_dict = cv2.aruco.getPredefinedDictionary(marker_type)
#     parameters = cv2.aruco.DetectorParameters()
#     detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

#     marker_points = []
#     total_markers = rows * cols
#     marker_ids = np.arange(total_markers, dtype=np.int32)

#     for row in range(rows): 
#         y_start = row * (marker_size + col_gap) 
#         for col in range(cols):
#             x_start = col * (marker_size + row_gap)
#             obj_points = np.array([
#                 [x_start, y_start, 0],
#                 [x_start + marker_size, y_start, 0],
#                 [x_start + marker_size, y_start + marker_size, 0],
#                 [x_start, y_start + marker_size, 0]
#             ], dtype=np.float32)
#             marker_points.append(obj_points)
    
#     board = cv2.aruco.Board(marker_points, aruco_dict, marker_ids)

#     if frame.shape[-1] == 3:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = frame

#     corners, ids, rejected = detector.detectMarkers(gray)
#     if ids is None:
#         return None

#     objP, imgP = board.matchImagePoints(corners, ids)
#     if len(objP) > 5:
#         valid, rvec, tvec = cv2.solvePnP(objP, imgP, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
#         if valid:
#             R_orig, _ = cv2.Rodrigues(rvec) # 기존 회전 벡터를 행렬로 변환
#             R_y180 = np.array([[-1, 0, 0],
#                                [ 0, 1, 0],
#                                [ 0, 0,-1]], dtype=np.float32)
#             R_new = R_orig @ R_y180 # 회전 결합
#             rvec_new, _ = cv2.Rodrigues(R_new)

#             t_shift = np.array([[-0.2], [0], [0]], dtype=np.float32)
#             tvec_adjustment = R_new @ t_shift
#             tvec_new = tvec + tvec_adjustment

#             return (rvec_new, tvec_new)
   
#     return None


def detectArucoMarker(frame, K, D, marker_size, row_gap, col_gap, rows=2, cols=5, marker_type = cv2.aruco.DICT_4X4_50):
    aruco_dict = cv2.aruco.getPredefinedDictionary(marker_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    marker_points = []
    total_markers = cols
    marker_ids = np.arange(total_markers, dtype=np.int32)

    # for row in range(rows): 
        # y_start = row * (marker_size + col_gap) 
    for col in range(cols):
        x_start = col * (marker_size + row_gap)
        obj_points = np.array([
            [x_start, 0, 0],
            [x_start + marker_size, 0, 0],
            [x_start + marker_size, 0 + marker_size, 0],
            [x_start, 0 + marker_size, 0]
        ], dtype=np.float32)
        marker_points.append(obj_points)
    
    board = cv2.aruco.Board(marker_points, aruco_dict, marker_ids)

    if frame.shape[-1] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    corners, ids, rejected = detector.detectMarkers(gray)
    if ids is None:
        return None

    objP, imgP = board.matchImagePoints(corners, ids)
    if len(objP) > 5:
        valid, rvec, tvec = cv2.solvePnP(objP, imgP, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if valid:
            R_orig, _ = cv2.Rodrigues(rvec) # 기존 회전 벡터를 행렬로 변환
            R_y180 = np.array([[-1, 0, 0],
                               [ 0, 1, 0],
                               [ 0, 0,-1]], dtype=np.float32)
            R_new = R_orig @ R_y180 # 회전 결합
            rvec_new, _ = cv2.Rodrigues(R_new)

            t_shift = np.array([[-0.2], [0], [0]], dtype=np.float32)
            tvec_adjustment = R_new @ t_shift
            tvec_new = tvec + tvec_adjustment

            return (rvec_new, tvec_new)
   
    return None


def drawMarker(frame, K, D, rvec, tvec):
    cv2.drawFrameAxes(frame, K, D, rvec, tvec, 0.05)


def img2world(u, v, K, D, rvec, tvec, height=0.0):
    # 1. 렌즈 왜곡 보정 (Undistort)
    points = np.array([[[u, v]]], dtype=np.float32)
    undistorted_points = cv2.undistortPoints(points, K, D, P=K)

    u_undist = undistorted_points[0][0][0]
    v_undist = undistorted_points[0][0][1]

    # 2. 좌표 변환 (cam to world coordinates)
    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    K_inv = np.linalg.inv(K)
    
    pixel_point = np.array([u_undist, v_undist, 1.0])
    point_camera = np.dot(K_inv, pixel_point)
    cam_world_center = -np.dot(R_inv, tvec)
    ray_world = np.dot(R_inv, point_camera)
    
    if abs(ray_world[2]) < 1e-6:
        return None 
        
    s = (height - cam_world_center[2, 0]) / ray_world[2]
    point_on_plane = cam_world_center + s * ray_world.reshape(3, 1)
    
    # [수정] numpy array가 아닌 순수 float 값으로 반환 (.item() 사용)
    # 이걸 안 하면 뒤에서 np.sqrt 계산할 때 에러가 발생함
    return point_on_plane[0, 0].item(), point_on_plane[1, 0].item(), point_on_plane[2, 0].item()


class Melon:
    id : int
    cx_entry : float
    cy_entry : float
    tx_entry : float
    ty_entry : float
    timestamp_entry : float

    angle : float
    velocity : float
    melon_pos_filter : MelonPosFilter

    def __init__(self) -> None:
        self.cx_entry = 0.0
        self.cy_entry = 0.0
        self.tx_entry = 0.0
        self.ty_entry = 0.0
        self.timestamp_entry = 0.0

        self.cx_last = 0.0
        self.cy_last = 0.0
        self.tx_last = 0.0
        self.ty_last = 0.0
        self.timestamp_last = 0.0
        self.timestamp_comp = 0.0

        self.angle = 0.0
        self.velocity = 0.0
        self.melon_pos_filter = MelonPosFilter()

    def setEntry(self, cx_entry, cy_entry, tx_entry, ty_entry, timestamp_entry):
        self.cx_entry = cx_entry
        self.cy_entry = cy_entry
        self.tx_entry = tx_entry
        self.ty_entry = ty_entry
        self.timestamp_entry = timestamp_entry

        self.melon_pos_filter.update(cx_entry, cy_entry, tx_entry, ty_entry, timestamp_entry)

    def setLast(self, cx_last, cy_last, tx_last, ty_last, timestamp_last):
        self.cx_last = cx_last
        self.cy_last = cy_last
        self.tx_last = tx_last
        self.ty_last = ty_last
        self.timestamp_last = timestamp_last
        self.timestamp_comp = time.time()

        self.melon_pos_filter.update(cx_last, cy_last, tx_last, ty_last, timestamp_last)


    def updata(self, cx, cy, tx, ty, timestamp):
        self.melon_pos_filter.update(cx, cy, tx, ty, timestamp)
        if self.timestamp_entry == 0.0:
            self.timestamp_entry = timestamp
    
    def calcVelocity(self):
        total_time_sec = (self.timestamp_last - self.timestamp_entry) / 1_000_000.0
        total_dist_m = math.sqrt((self.cx_last - self.cx_entry)**2 + (self.cy_last -  self.cy_entry)**2)

        if total_time_sec > 0:
            self.velocity = total_dist_m / total_time_sec

    def calcAngle(self):
        state_center = self.melon_pos_filter.center.kf.statePost
        state_tail = self.melon_pos_filter.tail.kf.statePost

        cx, cy = state_center[0, 0], state_center[1, 0]
        tx, ty = state_tail[0, 0], state_tail[1, 0]

        dx = tx - cx
        dy = ty - cy
        self.angle = math.degrees(math.atan2(dy, dx))
        if self.angle < 0:
            self.angle += 360
        
    def getPos(self, timestamp = None):
        cx = self.cx_last + self.velocity * (timestamp - self.timestamp_comp) 

        return (cx, self.cy_last, self.angle, self.velocity)


if __name__ == "__main__":
    camera = Playback("./data/recored_files/1080p.mkv")
    fps_meter = FPSMeter()
    K = camera.K
    D = camera.D

    # Real scale ROI
    roi_real = np.array([
        [0, 0, 0],
        [0.5, 0, 0]
    ], dtype=np.float32)

    is_calibaration = False
    is_inside_prev = False
    melon = None
    melon_list = []
    while True:
        frame = camera.getFrame()
        # if frame is None:
            # continue
        image = frame.image
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        depth = frame.depth
        timestamp_img = frame.timestamp_image

        if is_calibaration == False:
            results_aruco = detectArucoMarker(image, K, D, 0.027, 0.013, 0.405, 2, 5, cv2.aruco.DICT_4X4_50)
            if results_aruco:
                rvec, tvec = results_aruco
                roi_img, _ = cv2.projectPoints(roi_real, rvec, tvec, K, D)
                roi_Xmax, roi_Xmin = roi_img[..., 0].astype(int)
                print(roi_Xmin, roi_Xmax)
                # roi_Xmin = roi_img[0, 0, 0]
                # roi_Xmax = roi_img[1, 0, 0]
                roi_img = roi_img.astype(int).reshape(-1, 2)
                is_calibaration = True
                print("Calibration Success!")
                print(cv2.Rodrigues(rvec))
                print(tvec)
            else:
                continue

        melon_results = detect_melon(image, image)
        if melon_results is not None:
            cx_img, cy_img, tx_img, ty_img = melon_results
            is_inside_now = (cx_img > roi_Xmin and tx_img > roi_Xmin and cx_img < roi_Xmax and tx_img < roi_Xmax)
            if is_inside_now:
                cx_mm, cy_mm, _ = img2world(cx_img, cy_img, K, D, rvec, tvec, height=0.0)
                tx_mm, ty_mm, _ = img2world(tx_img, ty_img, K, D, rvec, tvec, height=0.0)
                print(cx_mm, cy_mm)
            if is_inside_now and not is_inside_prev: # enter
                is_inside_prev = True
                melon = Melon()
                melon.setEntry(cx_mm, cy_mm, tx_mm, ty_mm, timestamp_img)
            elif is_inside_now and is_inside_prev: # stay
                melon.updata(cx_mm, cy_mm, tx_mm, ty_mm, timestamp_img)
            elif not is_inside_now and is_inside_prev: # out
                is_inside_prev = False
                melon.setLast(cx_mm, cy_mm, tx_mm, ty_mm, timestamp_img)
                melon.calcAngle()
                melon.calcVelocity()
                print(melon.cx_last, melon.cy_last, melon.angle, melon.velocity)
                melon_list.append(melon)
                melon = None

        if len(melon_list) > 0:
            print(melon_list[0].getPos(time.time()))
        

        cv2.polylines(image, [roi_img], True, (128, 128, 128), 10)
        drawMarker(image, K, D, rvec, tvec)
        fps_meter.draw(image)
        cv2.imshow("window", image)
        # cv2.imshow("depth", depth)

        key = cv2.waitKeyEx(1)
        if key == ord('q') or key == 27:
            break
        elif key == 65363:
            camera.next()
        elif key == 65361:
            camera.prev()

    camera.close()
    cv2.destroyAllWindows()