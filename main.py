import cv2
import mediapipe as mp
import numpy as np

#создаем детектор
handsDetector = mp.solutions.hands.Hands(static_image_mode=False,
                   max_num_hands=1,
                   min_detection_confidence=0.7,
                   min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w, _ = frame.shape
helped_picture = np.ones((h, w, 3), dtype=np.uint8)
while(cap.isOpened()):
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
        x_noname = int(results.multi_hand_landmarks[0].landmark[12].x *
                    flippedRGB.shape[1])
        y_noname = int(results.multi_hand_landmarks[0].landmark[12].y *
                    flippedRGB.shape[0])
        if abs(x_big - x_noname) < 15 and abs(y_big - y_noname) < 15:
            cv2.circle(helped_picture, (x_index, y_index), 20, (0, 0, 0), -1)
        elif abs(x_big - x_little) < 15 and abs(y_big - y_little) < 15:
            cv2.circle(helped_picture, (x_index, y_index), 5, (0, 255, 0), -1)
    # переводим в BGR и показываем результат
    helped_mask = cv2.cvtColor(helped_picture, cv2.COLOR_BGR2GRAY)
    ret_inv, tr_inv = cv2.threshold(helped_mask, 10, 255, cv2.THRESH_BINARY_INV)
    flipped_BGR = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    res_image = cv2.bitwise_and(flipped_BGR, flipped_BGR, mask=tr_inv)
    cv2.imshow("Hands", res_image)

# освобождаем ресурсы
handsDetector.close()