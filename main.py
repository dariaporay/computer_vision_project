import cv2
import mediapipe as mp
import numpy as np
import math

#создаем детектор
handsDetector = mp.solutions.hands.Hands(static_image_mode=False,
                   max_num_hands=1,
                   min_detection_confidence=0.9,
                   min_tracking_confidence=0.9)
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
h, w, _ = frame.shape
helped_picture = np.zeros((h, w, 3), dtype=np.uint8)
color_helped_picture = np.zeros((h, w, 3), dtype=np.uint8)
print(color_helped_picture)
bin = cv2.imread("images/RecycleBin.png")
change_color = cv2.imread("images/change_color.jpg")
color_line = cv2.imread("images/color_line.png")
color_line = cv2.resize(color_line, dsize=(30, h // 2))
border_color = [30, 30]
fl_color = 0
picture_no_draw_zone = np.zeros((60, w, 3), dtype=np.uint8)
color = (0, 0, 0)
x_index_pred = -1
y_index_pred = -1
# линия между хонами рисования и не присования
picture_no_draw_zone[:4, :] = (219, 215, 210)
picture_no_draw_zone[30:, w - 30:] = bin
picture_no_draw_zone[30:, :30] = bin
fl_figure = 0
fl = False
x_const = 0
y_const = 0
# максимальное расстояние между пальцами для рисования, найдено подбором
eps = 0.03 * w
size = 3
k = 1
# распознование указательного пальца взято из канваса
while cap.isOpened():
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    flipped = np.fliplr(frame)
    # переводим его в формат RGB для распознавания
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    # Распознаем
    results = handsDetector.process(flippedRGB)
    # анализируем изображение, если распозналось
    if results.multi_hand_landmarks is not None:

        # понимаем какая рука: левая или правая
        hand = results.multi_handedness[0].classification[0].label
        if hand == "Left":
            k = -1
        else:
            k = 1

        # находим координате тех, точек которые нам нужны
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
        x_13 = int(results.multi_hand_landmarks[0].landmark[10].x *
                       flippedRGB.shape[1])
        y_17 = int(results.multi_hand_landmarks[0].landmark[17].y *
                       flippedRGB.shape[0])
        y_0 = int(results.multi_hand_landmarks[0].landmark[0].y *
                       flippedRGB.shape[0])
        y_1 = int(results.multi_hand_landmarks[0].landmark[1].y *
                       flippedRGB.shape[0])

        # рисование точки, в которой находится указательный палец
        cv2.circle(flippedRGB, (x_index, y_index), size, color, -1)

        # очистка холста
        if (0 < h - y_index < 30 and 0 < w - x_index < 30) or (0 < h - y_index < 30 and 0 < x_index < 30):
            helped_picture = np.zeros((h, w, 3), dtype=np.uint8)
        # ставим ограничения на положение руки во время рисования
        # числа подобраны ручками
        if y_17 - y_0 > eps or k * (x_index - x_little) > -eps / 4 or (y_1 - y_0) > eps or y_index - y_0 > -eps / 4:
            fl = False
        # ограничения пройдены, можно рисовать
        else:
            # проверка близости указательного и большого пальца для рисования окружности на данном кадре
            if abs(x_big - x_index) < eps and abs(y_big - y_index) < eps:
                if fl:
                    cv2.circle(flippedRGB, (x_const, y_const), int(math.hypot(x_const - x_index, y_const - y_index)),
                               color, size)
                else:
                    fl_figure = 2
                    fl = True
                    x_const, y_const = x_index, y_index
                    cv2.circle(flippedRGB, (x_index, y_index), size, color, -1)

            # проверка близости безымянного и указательного пальца для рисования отрезка на данном кадре
            elif abs(x_noname - x_index) < eps and abs(y_noname - y_index) < eps:
                if fl:
                    cv2.line(flippedRGB, (x_const, y_const), (x_index, y_index), color, size)
                else:
                    fl_figure = 1
                    fl = True
                    x_const, y_const = x_index, y_index
                    cv2.circle(flippedRGB, (x_index, y_index), size, color, -1)

            # проверка близости среднего и большого пальца для стирания уже нарисованного на вспомогательном холсте
            elif abs(x_big - x_middle) < eps and abs(y_big - y_middle) < eps:
                cv2.circle(helped_picture, (x_index, y_index), 20, [0, 0, 0], -1)

            # проверка близости мизинца и большого пальца для обычного рисования на вспомогателном холсте
            elif abs(x_big - x_little) < eps and abs(y_big - y_little) < eps:
                if x_index_pred == -1:
                    cv2.circle(helped_picture, (x_index, y_index), size, (color[2], color[1], color[0]), -1)
                else:
                    cv2.line(helped_picture, (x_index_pred, y_index_pred), (x_index, y_index),
                             (color[2], color[1], color[0]), size)

            # фиксирование конечной версии отрезка или окружности на вспомогательный холст
            else:
                if fl:
                    if not (h - 60 <= y_index or h - 60 <= y_const):
                        if fl_figure == 1:
                            cv2.line(helped_picture, (x_const, y_const), (x_index, y_index),
                                     (color[2], color[1], color[0]), size)
                        elif fl_figure == 2:
                            cv2.circle(helped_picture, (x_const, y_const),
                                       int(math.hypot(x_const - x_index, y_const - y_index)),
                                       (color[2], color[1], color[0]), size)
                fl = False
                fl_figure = 0

        x_index_pred, y_index_pred = x_index, y_index

        if border_color[0] - y_index >= 0 and border_color[1] - x_index >= 0:
            if fl_color == 0:
                fl_color = 1
                border_color[0] = h // 2
            else:
                color = [int(i) for i in color_helped_picture[y_index][x_index][::-1]]
        elif fl_color == 1:
            fl_color = 0
            border_color[0] = 30

    # переводим в BGR, накладываем нарисованный материал и показываем результат
    helped_picture[h - 60:, :] = picture_no_draw_zone
    helped_mask = cv2.cvtColor(helped_picture, cv2.COLOR_BGR2GRAY)
    ret_inv, tr_inv = cv2.threshold(helped_mask, 10, 255, cv2.THRESH_BINARY_INV)
    flipped_BGR = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    res_image = cv2.bitwise_and(flipped_BGR, flipped_BGR, mask=tr_inv) + helped_picture

    if fl_color == 0:
        color_helped_picture[30:, :] = [0, 0, 0]
        color_helped_picture[:30, :30] = change_color
        res_image[:30, :30] = [0, 0, 0]
    else:
        color_helped_picture[:h//2, :30] = color_line
        res_image[:h//2, :30] = [0, 0, 0]

    cv2.imshow("Hands", res_image + color_helped_picture)

# освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
handsDetector.close()
























# import cv2
# import threading
# import requests
# import mediapipe as mp
# import numpy as np
#
# # Глобальная переменная для хранения ввода
# steer = 90
# drive = 75
#
# ESP32_IP = "192.168.4.1"  # Замените на IP ESP32
#
#
# def set_servo(servo_num, angle):
#     url = f"http://{ESP32_IP}/servo{servo_num}?angle={angle}"
#     try:
#         response = requests.get(url)
#         # print(f"Серво {servo_num}: {angle}° | Ответ: {response.text}")
#     except Exception as e:
#         print(f"Ошибка: {e}")
#
# # Функция для ввода в отдельном потоке
# def input_thread():
#     while True:
#         global steer
#         global drive
#         steer, drive = map(int, input().split())
#
#
# # # Запускаем поток ввода
# input_processor = threading.Thread(target=input_thread, daemon=True)
# input_processor.start()
#
#
# #создаем детектор
# handsDetector = mp.solutions.hands.Hands(static_image_mode=False,
#                    max_num_hands=1,
#                    min_detection_confidence=0.9,
#                    min_tracking_confidence=0.9)
#
# # Основной поток — обработка видео
# camera_url = f"http://{ESP32_IP}:81"
# cap = cv2.VideoCapture(camera_url)
#
# ret, frame = cap.read()
# h, w, _ = frame.shape
# fl_figure = 0
# fl = False
# x_const = 0
# y_const = 0
# # максимальное расстояние между пальцами для рисования, найдено подбором
# eps = 0.03 * w
# size = 3
# k = 1
# delta_s = 5
# delta_d = 10
#
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Выход по нажатию клавиши 'q' в OpenCV (альтернативный способ)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#     flipped = np.fliplr(frame)
#     # переводим его в формат RGB для распознавания
#     flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
#     # Распознаем
#     results = handsDetector.process(flippedRGB)
#     # анализируем изображение, если распозналось
#     if results.multi_hand_landmarks is not None:
#
#         # понимаем какая рука: левая или правая
#         hand = results.multi_handedness[0].classification[0].label
#         if hand == "Left":
#             k = -1
#         else:
#             k = 1
#
#         # находим координате тех, точек которые нам нужны
#         # нужно умножить координаты а размеры картинки
#         x_index = int(results.multi_hand_landmarks[0].landmark[8].x *
#                       flippedRGB.shape[1])
#         y_index = int(results.multi_hand_landmarks[0].landmark[8].y *
#                       flippedRGB.shape[0])
#         x_little = int(results.multi_hand_landmarks[0].landmark[20].x *
#                        flippedRGB.shape[1])
#         y_little = int(results.multi_hand_landmarks[0].landmark[20].y *
#                        flippedRGB.shape[0])
#         x_big = int(results.multi_hand_landmarks[0].landmark[4].x *
#                     flippedRGB.shape[1])
#         y_big = int(results.multi_hand_landmarks[0].landmark[4].y *
#                     flippedRGB.shape[0])
#         x_big3 = int(results.multi_hand_landmarks[0].landmark[3].x *
#                      flippedRGB.shape[1])
#         y_big3 = int(results.multi_hand_landmarks[0].landmark[3].y *
#                      flippedRGB.shape[0])
#         x_middle = int(results.multi_hand_landmarks[0].landmark[12].x *
#                        flippedRGB.shape[1])
#         y_middle = int(results.multi_hand_landmarks[0].landmark[12].y *
#                        flippedRGB.shape[0])
#         x_noname = int(results.multi_hand_landmarks[0].landmark[16].x *
#                        flippedRGB.shape[1])
#         y_noname = int(results.multi_hand_landmarks[0].landmark[16].y *
#                        flippedRGB.shape[0])
#         x_ring = int(results.multi_hand_landmarks[0].landmark[5].x *
#                      flippedRGB.shape[1])
#         y_ring = int(results.multi_hand_landmarks[0].landmark[5].y *
#                      flippedRGB.shape[0])
#         x_17 = int(results.multi_hand_landmarks[0].landmark[17].x *
#                    flippedRGB.shape[1])
#         x_13 = int(results.multi_hand_landmarks[0].landmark[10].x *
#                    flippedRGB.shape[1])
#         y_17 = int(results.multi_hand_landmarks[0].landmark[17].y *
#                    flippedRGB.shape[0])
#         y_0 = int(results.multi_hand_landmarks[0].landmark[0].y *
#                   flippedRGB.shape[0])
#         y_1 = int(results.multi_hand_landmarks[0].landmark[1].y *
#                   flippedRGB.shape[0])
#
#         # ставим ограничения на положение руки во время рисования
#         # числа подобраны ручками
#         if y_17 - y_0 > eps or k * (x_index - x_little) > -eps / 4 or (y_1 - y_0) > eps or y_index - y_0 > -eps / 4:
#             fl = False
#             steer = 90
#             drive = 75
#         # ограничения пройдены, можно рисовать
#         else:
#             # проверка близости указательного и большого пальца для рисования окружности на данном кадре
#             if abs(x_big - x_index) < eps and abs(y_big - y_index) < eps:
#                 drive = min(100, drive + delta_s)
#
#             # проверка близости безымянного и указательного пальца для рисования отрезка на данном кадре
#             elif abs(x_noname - x_index) < eps and abs(y_noname - y_index) < eps:
#                 drive = max(50, drive - delta_s)
#
#             # проверка близости среднего и большого пальца для стирания уже нарисованного на вспомогательном холсте
#             elif abs(x_big - x_middle) < eps and abs(y_big - y_middle) < eps:
#                 steer = min(180, steer + delta_d)
#
#             # проверка близости мизинца и большого пальца для обычного рисования на вспомогателном холсте
#             elif abs(x_big - x_little) < eps and abs(y_big - y_little) < eps:
#                 steer = max(0, steer - delta_d)
#
#
#     # переводим в BGR, накладываем нарисованный материал и показываем результат
#     flipped_BGR = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
#
#     # Отображаем кадр
#     cv2.imshow("ESP32 Camera", flipped_BGR)
#
#     set_servo(1, steer)
#     set_servo(2, drive)
#
# cap.release()
# cv2.destroyAllWindows()
# handsDetector.close()
