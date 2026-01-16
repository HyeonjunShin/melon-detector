import numpy as np
import cv2

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
    if objP is not None:
        valid, rvec, tvec = cv2.solvePnP(objP, imgP, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if valid:
            # R_orig, _ = cv2.Rodrigues(rvec) # 기존 회전 벡터를 행렬로 변환
            # R_y180 = np.array([[-1, 0, 0],
                            #    [ 0, 1, 0],
                            #    [ 0, 0,-1]], dtype=np.float32)
            # R_new = R_orig @ R_y180 # 회전 결합
            # rvec_new, _ = cv2.Rodrigues(R_new)

            # t_shift = np.array([[-0.2], [0], [0]], dtype=np.float32)
            # tvec_adjustment = R_new @ t_shift
            # tvec_new = tvec + tvec_adjustment

            # return (rvec_new, tvec_new)
            return (rvec, tvec)
   
    return None


def drawMarker(frame, K, D, rvec, tvec):
    cv2.drawFrameAxes(frame, K, D, rvec, tvec, 0.05)

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

