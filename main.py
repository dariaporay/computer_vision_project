import cv2
import mediapipe as mp
import numpy as np
import math

#создаем детектор
handsDetector = mp.solutions.hands.Hands(static_image_mode=False,
                   max_num_hands=1,
                   min_detection_confidence=0.7,
                   min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w, _ = frame.shape
print(h, w)
helped_picture = np.zeros((h, w, 3), dtype=np.uint8)
picture_no_draw_zone = np.zeros((60, w, 3), dtype=np.uint8)
picture_no_draw_zone[:4, :] = (219, 215, 210)
fl_figure = 0
fl = False
x_const = 0
y_const = 0
color = (0, 255, 0)
eps = 20
size = 3
k = 1
while cap.isOpened():
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    flipped = np.fliplr(frame)
    # переводим его в формат RGB для распознавания
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    # Распознаем
    results = handsDetector.process(flippedRGB)
    # Рисуем распознанное, если распозналось
    if results.multi_hand_landmarks is not None:

        hand = results.multi_handedness[0].classification[0].label
        if hand == "Left":
            k = -1
        else:
            k = 1

        mp.solutions.drawing_utils.draw_landmarks(flippedRGB, results.multi_hand_landmarks[0])

        # нас интересует только подушечка указательного пальца (индекс 8)
        # нужно умножить координаты а размеры картинки
        x_index = int(results.multi_hand_landmarks[0].landmark[8].x *
                flippedRGB.shape[1])
        y_index = int(results.multi_hand_landmarks[0].landmark[8].y *
                flippedRGB.shape[0])
        x_little = int(results.multi_hand_landmarks[0].landmark[20].x *
                      flippedRGB.shape[1])
        y_little = int(results.multi_hand_landmarks[0].landmark[20].y *
                      flippedRGB.shape[0])
        x_big = int(results.multi_hand_landmarks[0].landmark[4].x *
                       flippedRGB.shape[1])
        y_big = int(results.multi_hand_landmarks[0].landmark[4].y *
                       flippedRGB.shape[0])
        x_big3 = int(results.multi_hand_landmarks[0].landmark[3].x *
                    flippedRGB.shape[1])
        y_big3 = int(results.multi_hand_landmarks[0].landmark[3].y *
                    flippedRGB.shape[0])
        x_middle = int(results.multi_hand_landmarks[0].landmark[12].x *
                    flippedRGB.shape[1])
        y_middle = int(results.multi_hand_landmarks[0].landmark[12].y *
                    flippedRGB.shape[0])
        x_noname = int(results.multi_hand_landmarks[0].landmark[16].x *
                       flippedRGB.shape[1])
        y_noname = int(results.multi_hand_landmarks[0].landmark[16].y *
                       flippedRGB.shape[0])
        x_ring = int(results.multi_hand_landmarks[0].landmark[5].x *
                       flippedRGB.shape[1])
        y_ring = int(results.multi_hand_landmarks[0].landmark[5].y *
                       flippedRGB.shape[0])
        x_17 = int(results.multi_hand_landmarks[0].landmark[17].x *
                     flippedRGB.shape[1])
        x_13 = int(results.multi_hand_landmarks[0].landmark[13].x *
                       flippedRGB.shape[1])
        y_17 = int(results.multi_hand_landmarks[0].landmark[17].y *
                       flippedRGB.shape[0])
        y_0 = int(results.multi_hand_landmarks[0].landmark[0].y *
                       flippedRGB.shape[0])
        y_1 = int(results.multi_hand_landmarks[0].landmark[1].y *
                       flippedRGB.shape[0])
        if y_17 - y_0 > 0 or k * (x_13 - x_17) > 5 or (y_1 - y_0) > 8:
            fl = False
        else:
            if abs(x_big - x_index) < eps and abs(y_big - y_index) < eps:
                if fl:
                    cv2.circle(flippedRGB, (x_const, y_const), int(math.hypot(x_const - x_index, y_const - y_index)),
                               (0, 0, 255), size)
                else:
                    fl_figure = 2
                    fl = True
                    x_const, y_const = x_index, y_index
                    cv2.circle(flippedRGB, (x_index, y_index), size, (0, 0, 255), -1)

            elif abs(x_noname - x_index) < eps and abs(y_noname - y_index) < eps:
                if fl:
                    cv2.line(flippedRGB, (x_const, y_const), (x_index, y_index), (255, 0, 0), size)
                else:
                    fl_figure = 1
                    fl = True
                    x_const, y_const = x_index, y_index
                    cv2.circle(flippedRGB, (x_index, y_index), size, (255, 0, 0), -1)

            elif abs(x_big - x_middle) < eps and abs(y_big - y_middle) < eps:
                cv2.circle(helped_picture, (x_index, y_index), 20, (0, 0, 0), -1)

            elif abs(x_big - x_little) < eps and abs(y_big - y_little) < eps:
                cv2.circle(helped_picture, (x_index, y_index), size, (0, 255, 0), -1)

            else:
                if fl:
                    if fl_figure == 1:
                        cv2.line(helped_picture, (x_const, y_const), (x_index, y_index), (255, 0, 0), size)
                    elif fl_figure == 2:
                        cv2.circle(helped_picture, (x_const, y_const),
                                   int(math.hypot(x_const - x_index, y_const - y_index)), (0, 0, 255), size)
                fl = False
                fl_figure = 0

    # переводим в BGR и показываем результат
    helped_picture[h - 60:, :] = picture_no_draw_zone
    helped_mask = cv2.cvtColor(helped_picture, cv2.COLOR_BGR2GRAY)
    ret_inv, tr_inv = cv2.threshold(helped_mask, 10, 255, cv2.THRESH_BINARY_INV)
    flipped_BGR = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    res_image = cv2.bitwise_and(flipped_BGR, flipped_BGR, mask=tr_inv)
    cv2.imshow("picture_no_draw_zone", picture_no_draw_zone)
    cv2.imshow("Hands", res_image + helped_picture)

# освобождаем ресурсы
handsDetector.close()