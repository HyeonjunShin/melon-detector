from kinect_camera import Camera, FPSMeter
from aruco_marker import detectArucoMarker, drawMarker
import cv2
from melon_detector import detect_melon
from utils import img2world
import numpy as np

def main():
    camera = Camera()
    camera.start()

    fps_meter = FPSMeter()
    K = camera.K
    D = camera.D

    while True:
        frame = camera.getFrame()
        if frame is None:
            continue
        image = frame.image
        # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        depth = frame.depth
        timestamp_img = frame.timestamp_image


        results_aruco = detectArucoMarker(image, K, D, 0.027, 0.013, 0.405, 2, 5, cv2.aruco.DICT_4X4_50)
        # if ids is not None:
        #     for i in range(len(ids)):
        #         # corners[i][0]이 해당 마커의 4개 모서리 2D 좌표
        #         _, rvec, tvec = cv2.solvePnP(marker_points, corners[i][0], K, D)
        #         rvecs.append(rvec)
        #         tvecs.append(tvec)
                
        #         # (선택 사항) 축 그리기 - OpenCV 버전에 따라 함수가 다를 수 있음
        #         cv2.drawFrameAxes(image, K, D, rvec, tvec, 0.03)

        # if corners is not None:
        #     rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, K, D)            
        #     for i, rvec in enumerate(rvecs):
        #         tvec = tvecs[i]
        #         cv2.drawFrameAxes(image, K, D, rvec, tvec, 0.05)
        if results_aruco:
            rvec, tvec = results_aruco
            drawMarker(image, K, D, rvec, tvec)

        # if is_calibaration == False:
        #     results_aruco = detectArucoMarker(image, K, D, 0.027, 0.013, 0.405, 2, 5, cv2.aruco.DICT_4X4_50)
        #     if results_aruco:
        #         rvec, tvec = results_aruco
        #         roi_img, _ = cv2.projectPoints(roi_real, rvec, tvec, K, D)
        #         roi_Xmax, roi_Xmin = roi_img[..., 0].astype(int)
        #         print(roi_Xmin, roi_Xmax)
        #         # roi_Xmin = roi_img[0, 0, 0]
        #         # roi_Xmax = roi_img[1, 0, 0]
        #         roi_img = roi_img.astype(int).reshape(-1, 2)
        #         is_calibaration = True
        #         print("Calibration Success!")
        #         print(cv2.Rodrigues(rvec))
        #         print(tvec)
        #     else:
        #         continue

        melon_results = detect_melon(image, image)
        if melon_results is not None:
            cx_img, cy_img, tx_img, ty_img, mask = melon_results
            # is_inside_now = (cx_img > roi_Xmin and tx_img > roi_Xmin and cx_img < roi_Xmax and tx_img < roi_Xmax)
            # if is_inside_now:
            cx_mm, cy_mm, _ = img2world(cx_img, cy_img, K, D, rvec, tvec, height=0.0)
            tx_mm, ty_mm, _ = img2world(tx_img, ty_img, K, D, rvec, tvec, height=0.0)
            print(cx_mm, cy_mm)
        #     if is_inside_now and not is_inside_prev: # enter
        #         is_inside_prev = True
        #         melon = Melon()
        #         melon.setEntry(cx_mm, cy_mm, tx_mm, ty_mm, timestamp_img)
        #     elif is_inside_now and is_inside_prev: # stay
        #         melon.updata(cx_mm, cy_mm, tx_mm, ty_mm, timestamp_img)
        #         # depth = depth[mask]
        #         # depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #         # depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        #         # cv2.imshow("depth", depth)
        #         print(melon.cx_last, melon.cy_last, melon.z, melon.angle, melon.velocity)

        #     elif not is_inside_now and is_inside_prev: # out
        #         is_inside_prev = False
        #         melon.setLast(cx_mm, cy_mm, tx_mm, ty_mm, timestamp_img)
        #         melon.calcAngle()
        #         melon.calcVelocity()

        #         z = ((tvec[2] * 1000) - np.median(depth[mask==255]) ).item()
        #         melon.setZ(z)
        #         print(melon.cx_last, melon.cy_last, melon.z, melon.angle, melon.velocity)

        #         melon_list.append(melon)
        #         melon = None

        # if len(melon_list) > 0:
        #     print(melon_list[0].getPos(time.time()))
        

        # cv2.polylines(image, [roi_img], True, (128, 128, 128), 10)
        fps_meter.draw(image)
        cv2.imshow("window", image)
        # cv2.imshow("depth", depth)

        key = cv2.waitKeyEx(1)
        if key == ord('q') or key == 27:
            break



if __name__ == "__main__":
    main()