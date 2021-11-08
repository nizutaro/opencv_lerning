
# opencv使ってカメラからの画像取得
import cv2

capture = cv2.VideoCapture(0)
capture.set(3, 320)  # 320 320 640 720　横の長さ
capture.set(4, 240)  # 180 240  360 405　縦の長さ

while True:
    ret, img = capture.read()
    img = cv2.flip(img, 1)  # 鏡表示にするため．
    cv2.imshow('img', img)
    # key Operation
    key = cv2.waitKey(10)
    if key == 27 or key == ord('q'):  # escまたはqキーで終了
        break
capture.release()
cv2.destroyAllWindows()
print("Exit")
