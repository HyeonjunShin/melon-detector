import cv2
import numpy as np

# 0번은 보통 내장 웹캠, USB 카메라는 1, 2 등의 인덱스를 가집니다.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Focus
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
focus_value = 20 
cap.set(cv2.CAP_PROP_FOCUS, focus_value)

# Exposure
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
exposure_value = -100
cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 2. 성능 향상을 위해 가우시안 블러 적용 (노이즈 제거)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    # 3. BGR -> HSV 변환
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 4. 노란색 범위 설정
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    # 5. 마스크 생성 및 모폴로지 연산 (작은 구멍 메우기)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 6. 윤곽선 찾기
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 7. 윤곽선 그리기 및 중심점 표시
    if len(contours) > 0:
        # 가장 큰 윤곽선 찾기
        c = max(contours, key=cv2.contourArea)
        
        # 최소한의 크기 조건 (너무 작은 노이즈 무시)
        if cv2.contourArea(c) > 500:
            # 윤곽선 그리기
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
            
            # 물체의 중심점(Moments) 계산
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # 중심에 점 찍기
                cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
                cv2.putText(frame, "Yellow Object", (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 결과 화면 표시
    cv2.imshow("Real-time Detection", frame)
    cv2.imshow("Mask", mask)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()