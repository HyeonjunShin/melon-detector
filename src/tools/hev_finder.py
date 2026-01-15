import cv2
import sys
import os
from kinect_playback import Playback
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def nothing(x):
    pass

cv2.namedWindow('Control Panel', cv2.WINDOW_NORMAL)

cv2.createTrackbar('H_Low', 'Control Panel', 15, 179, nothing)
cv2.createTrackbar('S_Low', 'Control Panel', 100, 255, nothing)
cv2.createTrackbar('V_Low', 'Control Panel', 100, 255, nothing)

cv2.createTrackbar('H_High', 'Control Panel', 35, 179, nothing)
cv2.createTrackbar('S_High', 'Control Panel', 255, 255, nothing)
cv2.createTrackbar('V_High', 'Control Panel', 255, 255, nothing)

cv2.setTrackbarPos('H_Low', 'Control Panel', 0)
cv2.setTrackbarPos('S_Low', 'Control Panel', 52)
cv2.setTrackbarPos('V_Low', 'Control Panel', 0)

cv2.setTrackbarPos('H_High', 'Control Panel', 25)
cv2.setTrackbarPos('S_High', 'Control Panel', 222)
cv2.setTrackbarPos('V_High', 'Control Panel', 255)


if __name__ == "__main__":
    device = Playback("./data/recored_files/distance2500.mkv")

    while True:
        frame = device.getImageAt(device.cur_idx)
        if frame is None:
            continue
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        h_l = cv2.getTrackbarPos('H_Low', 'Control Panel')
        s_l = cv2.getTrackbarPos('S_Low', 'Control Panel')
        v_l = cv2.getTrackbarPos('V_Low', 'Control Panel')

        h_h = cv2.getTrackbarPos('H_High', 'Control Panel')
        s_h = cv2.getTrackbarPos('S_High', 'Control Panel')
        v_h = cv2.getTrackbarPos('V_High', 'Control Panel')


        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([h_l, s_l, v_l])
        upper_yellow = np.array([h_h, s_h, v_h])
        
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

        result = cv2.bitwise_and(frame, frame, mask=mask)

        merge = np.hstack([frame, result])
        cv2.imshow('Control Panel', merge)

        key = cv2.waitKeyEx(1)
        print(key)
        if key == ord('q') or key == 27:
            break
        elif key == 93:
            device.cur_idx = device.cur_idx + 1
        elif key == 91:
            device.cur_idx = device.cur_idx - 1
        # save only the values to the txt
        elif key == ord('s'):
            with open('./output/hsv.txt', 'w') as f:
                f.write(f'{h_l} {s_l} {v_l}\n')
                f.write(f'{h_h} {s_h} {v_h}')
                break


    device.close()
    cv2.destroyAllWindows()
