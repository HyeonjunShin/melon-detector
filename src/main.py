import cv2
import numpy as np
from kinect_camera import Camera
from kinect_playback import Playback
from melon_detector import detect_melon
import time
from aruco_marker import detectArucoMarker, drawMarker
from data_types import Melon






if __name__ == "__main__":
    # camera = Playback("./data/recored_files/1080p.mkv")
    camera = Camera()
    camera.start()

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
        # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
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
            cx_img, cy_img, tx_img, ty_img, mask = melon_results
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
                # depth = depth[mask]
                # depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                # depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
                # cv2.imshow("depth", depth)
                print(melon.cx_last, melon.cy_last, melon.z, melon.angle, melon.velocity)

            elif not is_inside_now and is_inside_prev: # out
                is_inside_prev = False
                melon.setLast(cx_mm, cy_mm, tx_mm, ty_mm, timestamp_img)
                melon.calcAngle()
                melon.calcVelocity()

                z = ((tvec[2] * 1000) - np.median(depth[mask==255]) ).item()
                melon.setZ(z)
                print(melon.cx_last, melon.cy_last, melon.z, melon.angle, melon.velocity)

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