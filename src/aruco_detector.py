import numpy as np
import cv2


# marker_margin = 0.013
# def get_x_aligned_corners(col_index):
#     x_start = col_index * marker_margin
#     # 좌상, 우상, 우하, 좌하 순서
#     return np.array([
#         [x_start,          marker_size, 0], 
#         [x_start + marker_size, marker_size, 0], 
#         [x_start + marker_size, 0,      0], 
#         [x_start,          0,      0]
#     ], dtype=np.float32)


class ArUcoDetector:
    def __init__(self, K, C, marker_size, marker_type = cv2.aruco.DICT_4X4_50) -> None:

        self.K = K
        self.C = C

        self.marker_size = marker_size
        # self.obj_points = np.array([[-self.marker_size/2,  self.marker_size/2, 0],
        #             [ self.marker_size/2,  self.marker_size/2, 0],
        #             [ self.marker_size/2, -self.marker_size/2, 0],
        #             [-self.marker_size/2, -self.marker_size/2, 0]], dtype=np.float32)
        # marker_ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)

        # marker_points = [
        #     get_x_aligned_corners(0), # ID 0
        #     get_x_aligned_corners(1), # ID 1
        #     get_x_aligned_corners(2),  # ID 2
        #     get_x_aligned_corners(3),  # ID 2
        #     get_x_aligned_corners(4)  # ID 2
        # ]

        # board = cv2.aruco.Board(marker_points, aruco_dict, marker_ids)
        self.obj_points = np.array([
                                    [0, 0, 0],        
                                    [marker_size, 0, 0],
                                    [marker_size, -marker_size, 0],
                                    [0, -marker_size, 0]
        ], dtype=np.float32)

        aruco_dict = cv2.aruco.getPredefinedDictionary(marker_type)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


    def detect(self, frame):
        if frame.shape[-1] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        corners, ids, rejected = self.detector.detectMarkers(gray)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for id, corner in zip(ids, corners):
                if id > 9:
                    continue
                _, rvec, tvec = cv2.solvePnP(self.obj_points, corner, self.K, self.C, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                cv2.drawFrameAxes(frame, self.K, self.C, rvec, tvec, 0.03)