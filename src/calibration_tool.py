import cv2
import os
import numpy as np
from device.kinect_camera import Camera

def getChessboardCorners(gray, CHECKERBOARD):
    return cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

def checkValidCorners(corners, CHECKERBOARD):
    if corners is None:
        return False
    return len(corners) == CHECKERBOARD[0] * CHECKERBOARD[1]
    
if __name__ == "__main__":
    CHECKERBOARD = (8, 6)
    DIR_PATH = "output/"
    save_count = 0
    current_pos = None
    last_pos = None
    min_distance = 100
    min_variance = 200
    distance = 0

    camera = Camera()
    camera.start()

    cv2.namedWindow("Calibration tool", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration tool", 3840//2, 2160//2)
    
    while True:
        frame = camera.getFrame().color
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame_view = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = getChessboardCorners(gray, CHECKERBOARD)
        cv2.drawChessboardCorners(frame_view, CHECKERBOARD, corners, ret)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        if ret:
            current_pos = corners.mean(axis=0)[0]
        
        if last_pos is not None and current_pos is not None:
            distance = np.linalg.norm(current_pos - last_pos)
            cv2.putText(frame_view, f"Distance: {distance}", (10, 90), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            
        cv2.putText(frame_view, f"Variance: {variance}", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Calibration tool", frame_view)
    
        key = cv2.waitKey(1)
        if key == 32:
            if save_count == 0:
                if variance < min_variance:
                    print("이미지 선명도 이상")
                    continue

                save_count += 1
                cv2.imwrite(os.path.join(DIR_PATH, f"calib_{save_count}.jpg"), frame)
                last_pos = current_pos
                print(f"이미지 저장됨: {save_count}")

            else:
                if variance < min_variance:
                    print("이미지 선명도 이상")
                    continue

                if  distance < min_distance:
                    print("이전 이미지와의 거리 이상")
                    continue

                save_count += 1
                cv2.imwrite(os.path.join(DIR_PATH, f"calib_{save_count}.jpg"), frame)
                last_pos = current_pos
                print(f"이미지 저장됨: {save_count}")


        if key == ord('q'):
            break


    cv2.destroyAllWindows()

    img_list = [os.path.join(DIR_PATH, path) for path in os.listdir(DIR_PATH)]
    print(img_list)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    objpoints = [] # 실제 세계의 3D 점
    imgpoints = [] # 이미지 상의 2D 점

    valid_img_count = 0
    for fname in img_list:
        print(fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 코너 찾기
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            objpoints.append(objp)
            # 서브픽셀 정밀도 향상
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            valid_img_count += 1
            print(f"[성공] {os.path.basename(fname)}")
        else:
            print(f"[실패] {os.path.basename(fname)} - 코너를 찾을 수 없음")

    if valid_img_count < 5:
        print("성공한 이미지가 너무 적습니다. 더 많은 사진이 필요합니다.")
        
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # --- 결과 출력 ---
    print("\n" + "="*30)
    print("      Calibration Result")
    print("="*30)
    print(f"재투영 오차 (RMS Error): {ret:.4f}")
    
    print("\n[내부 파라미터 행렬 (Camera Matrix)]")
    print(f"fx (초점거리 x): {mtx[0,0]:.2f}")
    print(f"fy (초점거리 y): {mtx[1,1]:.2f}")
    print(f"cx (주점 x): {mtx[0,2]:.2f}")
    print(f"cy (주점 y): {mtx[1,2]:.2f}")
    
    print("\n[왜곡 계수 (Distortion Coefficients)]")
    # k1, k2, p1, p2, k3
    dist_labels = ["k1", "k2", "p1", "p2", "k3"]
    for label, val in zip(dist_labels, dist[0]):
        print(f"{label}: {val:.6f}")
    print("="*30)

    print(dist[0])
    print(mtx)

    test_img = cv2.imread(img_list[0])
    h, w = test_img.shape[:2]

    # 왜곡 보정 실행
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, (w, h), cv2.CV_32FC1)

    # dst = cv2.undistort(test_img, mtx, dist, None, new_mtx)
    while True:
        frame = camera.getFrame().color
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        
        # ROI를 이용하여 검은 외곽 잘라내기 (선택 사항)
        # x, y, w_roi, h_roi = roi
        # dst = dst[y:y+h_roi, x:x+w_roi]

        combined = np.hstack((frame, dst))
        
        cv2.imshow("Calibration", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 결과 비교 출력
    # combined = np.hstack((test_img, dst))
    # cv2.imshow("Calibration", cv2.resize(combined, (w, h//2)))
    # cv2.waitKey(0)


    cv2.destroyAllWindows()
    camera.stop()
