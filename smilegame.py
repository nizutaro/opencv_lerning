import cv2
import numpy as np
import time
import pygame.mixer

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 180 240  360 405
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)  # 320 320 640 720
cv2.namedWindow('img', cv2.WND_PROP_FULLSCREEN)
print(capture.get(cv2.CAP_PROP_FPS))
print(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
print(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

LV_GAIN = 5/100  # Smile_neighborからIntensityに変換するときのGain
TH_SMILE_NUM = 8  # 所定の笑顔認識数　今は8人に設定
TH_SMILE_TIME = 5  # 維持しないといけない秒数

# 笑顔強度に応じたRGBの色を返す


def Intensity2RGB(intensityZeroOne):
    if intensityZeroOne < 0.1:
        return (255, 0, 0)
    elif intensityZeroOne < 0.2:
        return (255, 127, 0)
    elif intensityZeroOne < 0.3:
        return (255, 255, 0)
    elif intensityZeroOne < 0.4:
        return (127, 255, 0)
    elif intensityZeroOne < 0.5:
        return (0, 255, 0)
    elif intensityZeroOne < 0.6:
        return (0, 255, 127)
    elif intensityZeroOne < 0.7:
        return (0, 255, 255)
    elif intensityZeroOne < 0.8:
        return (0, 127, 255)
    else:
        return (0, 0, 255)


face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('./haarcascade_smile.xml')

# 時間計測のための変数
f_timecount = False
t_starttime = 0

# ゲームクリアしたかどうか
f_clear = False

# 効果音を鳴らすための処理
pygame.mixer.init()
pygame.mixer.music.load("chaim.mp3")

while True:
    if f_clear == False:
        ret, img = capture.read()
        img = cv2.flip(img, 1)  # 鏡表示にするため．
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

        # 笑顔カウンタをリセット
        smilecount = 0

        for (x, y, w, h) in faces:
            # 顔領域切り出し
            roi_gray = gray[y:y+h, x:x+w]

            # サイズを規格化
            roi_gray = cv2.resize(roi_gray, (100, 100))

    #                # 輝度で規格化
    #                lmin = roi_gray.min() #輝度の最小値
    #                lmax = roi_gray.max() #輝度の最大値
    #                for index1, item1 in enumerate(roi_gray):
    #                        for index2, item2 in enumerate(item1) :
    #                                roi_gray[index1][index2] = int((item2 - lmin)/(lmax-lmin) * item2)

            # 笑顔認識
            smiles = smile_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=0, minSize=(20, 20))

            # 笑顔強度の算出
            smile_neighbors = len(smiles)
            intensityZeroOne = smile_neighbors * LV_GAIN
            #if intensityZeroOne > 1.0: intensityZeroOne = 1.0

            # 顔領域に矩形描画（色は強度に応じて）
            cv2.rectangle(img, (x, y), (x+w, y+h),
                          Intensity2RGB(intensityZeroOne), 2)  # blue

            # 笑顔強度が0.8以上の場合，笑顔としてカウント
            if intensityZeroOne >= 0.8:
                smilecount += 1

        print("笑顔数=", smilecount)  # 認識した笑顔数を表示

        # 時間計測
        if smilecount >= TH_SMILE_NUM:  # もし笑顔数が閾値超えてたら
            if f_timecount == False:  # もしまだカウントダウンが始まってなかったら
                f_timecount = True
                t_start = time.time()
            else:  # カウントダウンが始まってたら
                if time.time() - t_start < TH_SMILE_TIME:  # もしカウントダウンが閾値超えてないなら
                    tremain = TH_SMILE_TIME - \
                        (time.time() - t_start)  # 残り時間の算出
                    cv2.putText(img, str(np.ceil(tremain)), (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
                else:  # もしカウントダウンが閾値超えたら
                    cv2.putText(img, "OK!!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))  # 成功の記載
                    # ゲームクリアのフラグ
                    f_clear = True

        else:  # 笑顔数が閾値を下回ったら
            f_timecount = False  # 時間計測をストップ
            t_start = 0.0

        cv2.imshow('img', img)

        if f_clear == True:  # ゲームクリアのフラグが立ってたら
            # 音を鳴らすコード
            pygame.mixer.music.play(3)
            time.sleep(1)
            pygame.mixer.music.stop()

    # key Operation
    key = cv2.waitKey(5)
    if key == 27 or key == ord('q'):  # escまたはeキーで終了
        break
    elif key == ord('r'):  # ゲームをリスタート
        f_clear = False

capture.release()
cv2.destroyAllWindows()

print("Exit")
