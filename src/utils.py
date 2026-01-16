import numpy as np
import cv2 

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
