import cv2
from camera_devices.kinect_wrapper import KinectCamera

mouse_x, mouse_y = 0, 0


def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y


def main():
    camera = KinectCamera()
    camera.start()
    K = camera.K
    D = camera.D

    cv2.namedWindow("img")
    cv2.setMouseCallback("img", mouse_callback)

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        color, depth, timestamp = frame

        print(depth)

        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # color = cv2.addWeighted(color, 0.6, depth_color, 0.4, 0, 0)

        text = f"X: {mouse_x}, Y: {mouse_y}, depth: {depth[mouse_y][mouse_x]}"
        cv2.putText(
            color,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(color, (mouse_x, mouse_y), 4, (0, 255, 0), -1, cv2.LINE_AA)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        cv2.imshow("img", color)
        cv2.imshow("depth", depth_color)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    camera.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
