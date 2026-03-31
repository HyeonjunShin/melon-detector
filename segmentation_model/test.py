import cv2
import numpy as np

# 1. 학습에 사용했던 원본 RGB 이미지와 마스크 파일을 그대로 불러옵니다.
# (경로를 실제 파일 이름으로 변경해 주세요)
img_path = "/home/hyeonjun/Desktop/melon_dataset-v3/rgb_0062.jpg" 
mask_path = "/home/hyeonjun/Desktop/melon_dataset-v3/segmentation_0062.png"

img = cv2.imread(img_path)
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

# 2. 마스크가 칠해진 영역을 빨간색으로 덮어씌웁니다.
overlay = img.copy()
overlay[mask > 0] = (0, 0, 255) # BGR 포맷이므로 빨간색

# 3. 원본 이미지와 마스크를 50%씩 섞어서 화면에 띄웁니다.
blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

cv2.imshow("Check Raw Ground Truth", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()