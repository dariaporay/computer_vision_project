import cv2
import mediapipe as mp
import numpy as np
import math
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
from tkinter import filedialog, Tk
from collections import deque


# Функция для поиска доступной камеры
def find_available_camera():
    for i in range(0, -1, -1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"Используется камера {i}")
                return cap, i
        cap.release()
    return None, -1


# Создаем детектор
handsDetector = mp.solutions.hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.9,
                                         min_tracking_confidence=0.9)

# Поиск камеры
cap, camera_id = find_available_camera()
if cap is None:
    print("Не удалось найти камеру! Используем стандартную камеру 0")
    cap = cv2.VideoCapture(0)

ret, frame = cap.read()
ph, pw, pc = frame.shape
h, w = 720, int(pw * 720 / ph)

if not ret:
    print("Ошибка: не удалось получить кадр с камеры")
    frame = np.zeros((h, w, 3), dtype=np.uint8)

else:
    frame = cv2.resize(frame, (w, h))

# Константа eps для определения попадания (0.03 от ширины экрана)
eps = 0.03 * w

helped_picture = np.zeros((h, w, 3), dtype=np.uint8)
color_helped_picture = np.zeros((h, w, 3), dtype=np.uint8)

# Загрузка изображений с проверкой
bin_img = cv2.imread("images/RecycleBin.png")
if bin_img is None:
    bin_img = np.zeros((30, 30, 3), dtype=np.uint8)
    cv2.putText(bin_img, "Bin", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

change_color = cv2.imread("images/change_color.jpg")
if change_color is None:
    change_color = np.zeros((30, 30, 3), dtype=np.uint8)
    cv2.rectangle(change_color, (5, 5), (25, 25), (0, 255, 0), -1)

color_line = cv2.imread("images/color_line.png")
if color_line is None:
    color_line = np.zeros((h // 2, 30, 3), dtype=np.uint8)
    for i in range(h // 2):
        color_line[i, :] = (i * 255 // (h // 2), 0, 255 - i * 255 // (h // 2))
else:
    color_line = cv2.resize(color_line, dsize=(30, h // 2))

border_color = [30, 30]
fl_color = 0
picture_no_draw_zone = np.zeros((60, w, 3), dtype=np.uint8)
color = (0, 0, 0)
x_index_pred = -1
y_index_pred = -1

# Параметры для лазера
LASER_FADE_FRAMES = 30  # Количество кадров, через которое исчезает линия
laser_trails = deque(maxlen=100)  # Хранит точки лазера с временем жизни


# Функция для парсинга GeoGebra файла
def parse_geogebra_file(file_path, width, height):
    """Парсит .ggb файл и возвращает изображение с отрисованным чертежом"""
    temp_image = np.zeros((height, width, 3), dtype=np.uint8)

    try:
        print(f"Открываем файл: {file_path}")

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            print("Файл успешно открыт как ZIP архив")
            print("Содержимое:", zip_ref.namelist())

            xml_content = None
            for xml_name in ['geogebra.xml', 'construction.xml', 'ggb.xml']:
                if xml_name in zip_ref.namelist():
                    with zip_ref.open(xml_name) as xml_file:
                        xml_content = xml_file.read()
                    print(f"Найден XML файл: {xml_name}")
                    break

            if xml_content is None:
                print("Не найден XML файл в архиве")
                return None

            tree = ET.parse(BytesIO(xml_content))
            root = tree.getroot()

        points = {}
        segments = []
        circles = []
        rects = []

        def to_screen(x, y):
            scale_x = width / 20
            scale_y = height / 20
            screen_x = int((x + 10) * scale_x)
            screen_y = int((10 - y) * scale_y)
            screen_x = max(0, min(width - 1, screen_x))
            screen_y = max(0, min(height - 1, screen_y))
            return screen_x, screen_y

        # Поиск всех элементов
        all_elements = []

        # Ищем в construction
        construction = root.find('.//construction')
        if construction is None:
            construction = root.find('.//{*}construction')

        if construction is not None:
            all_elements = construction.findall('.//element')
            if not all_elements:
                all_elements = construction.findall('.//{*}element')

        # Если не нашли, ищем в корне
        if not all_elements:
            all_elements = root.findall('.//element')
            if not all_elements:
                all_elements = root.findall('.//{*}element')

        print(f"Найдено элементов: {len(all_elements)}")

        # Сбор точек
        for element in all_elements:
            element_type = element.get('type', '')
            label = element.get('label', '')

            if element_type == 'point':
                coords = element.find('.//coords')
                if coords is None:
                    coords = element.find('.//{*}coords')

                if coords is not None:
                    x = float(coords.get('x', 0))
                    y = float(coords.get('y', 0))
                    z = float(coords.get('z', 1))

                    if z != 1 and z != 0:
                        x = x / z
                        y = y / z

                    screen_x, screen_y = to_screen(x, y)
                    points[label] = (screen_x, screen_y)

                    cv2.circle(temp_image, (screen_x, screen_y), 5, (255, 0, 0), -1)
                    cv2.putText(temp_image, label, (screen_x + 5, screen_y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                    print(f"Точка {label}: ({screen_x}, {screen_y})")

        # Сбор отрезков
        for element in all_elements:
            element_type = element.get('type', '')
            label = element.get('label', '')

            if element_type == 'segment':
                start_label = None
                end_label = None

                # Ищем в дочерних элементах
                for child in element:
                    tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    if tag in ['startPoint', 'point'] and start_label is None:
                        start_label = child.get('label', '')
                    elif tag in ['endPoint', 'point'] and start_label is not None:
                        end_label = child.get('label', '')

                if start_label in points and end_label in points:
                    segments.append((points[start_label], points[end_label]))
                    cv2.line(temp_image, points[start_label], points[end_label], (0, 0, 255), 3)
                    print(f"ОТРЕЗОК {label}: {start_label} -> {end_label}")

        # Поиск через команды
        commands = root.findall('.//command')
        if not commands:
            commands = root.findall('.//{*}command')

        for cmd in commands:
            cmd_name = cmd.get('name', '')
            if cmd_name == 'Segment':
                inputs = cmd.findall('.//input')
                if not inputs:
                    inputs = cmd.findall('.//{*}input')

                if len(inputs) >= 2:
                    start_label = inputs[0].get('label', '')
                    end_label = inputs[1].get('label', '')

                    if start_label in points and end_label in points:
                        already_added = False
                        for seg in segments:
                            if (seg[0] == points[start_label] and seg[1] == points[end_label]) or \
                                    (seg[0] == points[end_label] and seg[1] == points[start_label]):
                                already_added = True
                                break

                        if not already_added:
                            segments.append((points[start_label], points[end_label]))
                            cv2.line(temp_image, points[start_label], points[end_label], (0, 0, 255), 3)
                            print(f"ОТРЕЗОК ИЗ КОМАНДЫ: {start_label} -> {end_label}")

        print(f"ИТОГО: {len(points)} точек, {len(segments)} отрезков, {len(circles)} окружностей")

        # Принудительно рисуем все отрезки
        for seg in segments:
            cv2.line(temp_image, seg[0], seg[1], (0, 0, 255), 3)

        if len(points) == 0 and len(segments) == 0:
            print("Не найдено ни одного объекта")
            return None

        return temp_image

    except Exception as e:
        print(f"Ошибка при парсинге GeoGebra файла: {e}")
        import traceback
        traceback.print_exc()
        return None


# Функция для загрузки изображения
def load_image_file(file_path, width, height):
    img = cv2.imread(file_path)
    if img is None:
        print(f"Не удалось загрузить изображение: {file_path}")
        return None

    img_height, img_width = img.shape[:2]
    max_size = min(width, height) * 0.6
    scale = min(max_size / img_width, max_size / img_height)
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)

    if new_width > 0 and new_height > 0:
        img = cv2.resize(img, (new_width, new_height))

    return img


# Класс для управления трансформацией импортированного объекта
class TransformControl:
    def __init__(self, x, y, width, height, image):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.image = image
        self.resize_handle_size = 15
        self.last_finger_x = 0
        self.last_finger_y = 0
        self.is_interacting = False

    def draw(self, frame):
        if 0 <= self.y < frame.shape[0] and 0 <= self.x < frame.shape[1]:
            scaled_image = cv2.resize(self.image, (self.width, self.height))
            h_img, w_img = scaled_image.shape[:2]

            y_end = min(self.y + h_img, frame.shape[0])
            x_end = min(self.x + w_img, frame.shape[1])
            h_crop = y_end - self.y
            w_crop = x_end - self.x

            if h_crop > 0 and w_crop > 0:
                # Создаем маску для непрозрачных пикселей импортированного изображения
                if len(scaled_image.shape) == 3:
                    # Если изображение цветное, создаем маску по черному цвету (фон)
                    gray = cv2.cvtColor(scaled_image[:h_crop, :w_crop], cv2.COLOR_BGR2GRAY)
                    mask = gray > 10  # Не черные пиксели

                    # Накладываем только непрозрачные пиксели
                    for c in range(3):
                        frame[self.y:y_end, self.x:x_end, c][mask] = scaled_image[:h_crop, :w_crop, c][mask]
                else:
                    frame[self.y:y_end, self.x:x_end] = scaled_image[:h_crop, :w_crop]

        # Рисуем рамку
        cv2.rectangle(frame, (self.x, self.y),
                      (self.x + self.width, self.y + self.height),
                      (0, 255, 0), 2)

        handle_size = self.resize_handle_size
        corners = [
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x, self.y + self.height),
            (self.x + self.width, self.y + self.height)
        ]

        for cx, cy in corners:
            cv2.rectangle(frame, (cx - handle_size // 2, cy - handle_size // 2),
                          (cx + handle_size // 2, cy + handle_size // 2),
                          (0, 255, 0), -1)

    def check_interaction(self, finger_x, finger_y):
        handle_size = self.resize_handle_size

        # Проверяем углы (приоритет у углов)
        if (abs(finger_x - (self.x + self.width)) < handle_size + eps and
                abs(finger_y - (self.y + self.height)) < handle_size + eps):
            return "resize_br"
        elif (abs(finger_x - (self.x + self.width)) < handle_size + eps and
              abs(finger_y - self.y) < handle_size + eps):
            return "resize_tr"
        elif (abs(finger_x - self.x) < handle_size + eps and
              abs(finger_y - (self.y + self.height)) < handle_size + eps):
            return "resize_bl"
        elif (abs(finger_x - self.x) < handle_size + eps and
              abs(finger_y - self.y) < handle_size + eps):
            return "resize_tl"
        # Проверяем область перемещения
        elif (self.x - eps <= finger_x <= self.x + self.width + eps and
              self.y - eps <= finger_y <= self.y + self.height + eps):
            return "move"
        return None

    def resize(self, finger_x, finger_y, handle_type):
        old_x, old_y = self.x, self.y
        old_w, old_h = self.width, self.height

        if handle_type == "resize_br":
            self.width = max(30, finger_x - self.x)
            self.height = max(30, finger_y - self.y)
        elif handle_type == "resize_bl":
            self.width = max(30, old_x + old_w - finger_x)
            self.height = max(30, finger_y - self.y)
            self.x = finger_x
        elif handle_type == "resize_tr":
            self.width = max(30, finger_x - self.x)
            self.height = max(30, old_y + old_h - finger_y)
            self.y = finger_y
        elif handle_type == "resize_tl":
            self.width = max(30, old_x + old_w - finger_x)
            self.height = max(30, old_y + old_h - finger_y)
            self.x = finger_x
            self.y = finger_y

        # Ограничиваем границами
        self.x = max(0, min(self.x, w - self.width))
        self.y = max(0, min(self.y, h - self.height))
        self.width = max(30, min(self.width, w - self.x))
        self.height = max(30, min(self.height, h - self.y))

    def move(self, finger_x, finger_y):
        new_x = finger_x - self.width // 2
        new_y = finger_y - self.height // 2
        self.x = max(0, min(new_x, w - self.width))
        self.y = max(0, min(new_y, h - self.height))

    def finalize(self, drawing_layer):
        # Сохраняем только непрозрачные пиксели (не черные)
        if 0 <= self.y < drawing_layer.shape[0] and 0 <= self.x < drawing_layer.shape[1]:
            scaled_image = cv2.resize(self.image, (self.width, self.height))
            h_img, w_img = scaled_image.shape[:2]

            y_end = min(self.y + h_img, drawing_layer.shape[0])
            x_end = min(self.x + w_img, drawing_layer.shape[1])
            h_crop = y_end - self.y
            w_crop = x_end - self.x

            if h_crop > 0 and w_crop > 0:
                # Создаем маску для непрозрачных пикселей
                gray = cv2.cvtColor(scaled_image[:h_crop, :w_crop], cv2.COLOR_BGR2GRAY)
                mask = gray > 10

                # Сохраняем только непрозрачные пиксели
                for c in range(3):
                    drawing_layer[self.y:y_end, self.x:x_end, c][mask] = scaled_image[:h_crop, :w_crop, c][mask]


# Функция для корректной загрузки иконок
def load_icon(path, size):
    icon = cv2.imread(path)
    if icon is None:
        icon = np.zeros((size, size, 3), dtype=np.uint8)
        if "круг" in path:
            cv2.circle(icon, (size // 2, size // 2), size // 2 - 3, (0, 255, 0), -1)
        elif "отрезок" in path:
            cv2.line(icon, (5, size - 5), (size - 5, 5), (0, 0, 255), 3)
        elif "ластик" in path:
            cv2.rectangle(icon, (5, 5), (size - 5, size - 5), (255, 255, 255), -1)
        elif "кривая" in path:
            points = np.array([[5, size - 5], [size // 3, size // 2], [2 * size // 3, size // 3], [size - 5, 5]],
                              np.int32)
            cv2.polylines(icon, [points], False, (255, 255, 0), 2)
        elif "прямоугольник" in path:
            cv2.rectangle(icon, (5, 5), (size - 5, size - 5), (0, 255, 255), -1)
        elif "лазер" in path:
            # Иконка лазера - красная линия и точка
            cv2.line(icon, (5, size - 5), (size - 5, 5), (0, 0, 255), 2)
            cv2.circle(icon, (size - 5, 5), 5, (0, 0, 255), -1)
        elif "импорт" in path:
            cv2.arrowedLine(icon, (size // 2, 5), (size // 2, size - 5), (0, 255, 255), 2)
            cv2.rectangle(icon, (size // 4, size - 5), (3 * size // 4, size - 5), (0, 255, 255), 2)
    else:
        if len(icon.shape) == 3 and icon.shape[2] == 4:
            icon = cv2.cvtColor(icon, cv2.COLOR_BGRA2BGR)
        icon = cv2.resize(icon, (size, size))
    return icon


# Загрузка иконок для режимов
icon_size = 30
icon_margin = 10
main_icon_x = w - icon_size - icon_margin
main_icon_y = icon_margin

circle_icon = load_icon("images/круг.png", icon_size)
line_icon = load_icon("images/отрезок.png", icon_size)
eraser_icon = load_icon("images/ластик.png", icon_size)
curve_icon = load_icon("images/кривая.png", icon_size)
rect_icon = load_icon("images/прямоугольник.png", icon_size)
laser_icon = load_icon("images/лазер.png", icon_size)
import_icon = load_icon("images/импорт.png", icon_size)

# Режимы рисования
MODE_CURVE = 0
MODE_LINE = 1
MODE_CIRCLE = 2
MODE_ERASER = 3
MODE_RECTANGLE = 4
MODE_LASER = 5
MODE_IMPORT = 6

current_mode = MODE_CURVE
menu_expanded = False

menu_icons = [
    {"mode": MODE_CURVE, "icon": curve_icon, "name": "кривая"},
    {"mode": MODE_LINE, "icon": line_icon, "name": "отрезок"},
    {"mode": MODE_CIRCLE, "icon": circle_icon, "name": "круг"},
    {"mode": MODE_RECTANGLE, "icon": rect_icon, "name": "прямоугольник"},
    {"mode": MODE_LASER, "icon": laser_icon, "name": "лазер"},
    {"mode": MODE_ERASER, "icon": eraser_icon, "name": "ластик"},
    {"mode": MODE_IMPORT, "icon": import_icon, "name": "импорт"}
]


def draw_interface(frame, current_mode, menu_expanded):
    if current_mode == MODE_CURVE:
        main_icon = curve_icon
    elif current_mode == MODE_LINE:
        main_icon = line_icon
    elif current_mode == MODE_CIRCLE:
        main_icon = circle_icon
    elif current_mode == MODE_RECTANGLE:
        main_icon = rect_icon
    elif current_mode == MODE_LASER:
        main_icon = laser_icon
    elif current_mode == MODE_ERASER:
        main_icon = eraser_icon
    else:
        main_icon = import_icon

    # Рисуем фон для основной иконки
    cv2.rectangle(frame, (main_icon_x - 2, main_icon_y - 2),
                  (main_icon_x + icon_size + 2, main_icon_y + icon_size + 2),
                  (200, 200, 200), -1)
    cv2.rectangle(frame, (main_icon_x - 2, main_icon_y - 2),
                  (main_icon_x + icon_size + 2, main_icon_y + icon_size + 2),
                  (0, 0, 0), 1)

    frame[main_icon_y:main_icon_y + icon_size, main_icon_x:main_icon_x + icon_size] = main_icon

    if menu_expanded:
        menu_y_start = main_icon_y + icon_size + 3
        menu_height = len(menu_icons) * (icon_size + 3) + 5

        cv2.rectangle(frame, (main_icon_x - 5, menu_y_start - 5),
                      (main_icon_x + icon_size + 5, menu_y_start + menu_height),
                      (240, 240, 240), -1)
        cv2.rectangle(frame, (main_icon_x - 5, menu_y_start - 5),
                      (main_icon_x + icon_size + 5, menu_y_start + menu_height),
                      (0, 0, 0), 1)

        for i, item in enumerate(menu_icons):
            icon_y = menu_y_start + i * (icon_size + 3)

            cv2.rectangle(frame, (main_icon_x, icon_y),
                          (main_icon_x + icon_size, icon_y + icon_size),
                          (0, 0, 0), 1)

            frame[icon_y:icon_y + icon_size, main_icon_x:main_icon_x + icon_size] = item["icon"]

    return frame


# Функция проверки попадания пальца на иконку
def is_finger_on_icon(x, y, icon_x, icon_y, size):
    return (icon_x - eps <= x <= icon_x + size + eps) and (icon_y <= y <= icon_y + size + eps)


picture_no_draw_zone[30:, w - 30:] = bin_img
picture_no_draw_zone[30:, :30] = bin_img

# Переменные для рисования
is_drawing = False
start_x, start_y = 0, 0
prev_x, prev_y = -1, -1
drawing_size = 3
k = 1
frame_counter = 0

# Переменные для импорта
app_state = "drawing"
imported_image = None
transform_control = None
interaction_type = None
was_interacting = False

# Создаем слои
drawing_layer = np.zeros((h, w, 3), dtype=np.uint8)
preview_layer = np.zeros((h, w, 3), dtype=np.uint8)  # Слой для предпросмотра фигур
finger_layer = np.zeros((h, w, 3), dtype=np.uint8)  # Отдельный слой для точки на пальце

print(f"Программа запущена. Размер экрана: {w}x{h}")
print(f"eps = {eps}")
print("Управление: наведите на иконку в правом верхнем углу для выбора режима")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ошибка захвата кадра")
        break
    else:
        frame = cv2.resize(frame, (w, h))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_counter += 1

    # Обновляем время жизни лазерных линий
    new_trails = []
    for trail in laser_trails:
        if trail[2] > 0:
            new_trails.append((trail[0], trail[1], trail[2] - 1))
        else:
            # Удаляем истекшие линии
            pass
    laser_trails.clear()
    laser_trails.extend(new_trails)

    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)

    # Очищаем слои предпросмотра в начале каждого кадра
    preview_layer.fill(0)
    finger_layer.fill(0)

    if results.multi_hand_landmarks is not None:
        hand = results.multi_handedness[0].classification[0].label
        if hand == "Left":
            k = -1
        else:
            k = 1

        x_index = int(results.multi_hand_landmarks[0].landmark[8].x * flippedRGB.shape[1])
        y_index = int(results.multi_hand_landmarks[0].landmark[8].y * flippedRGB.shape[0])
        x_little = int(results.multi_hand_landmarks[0].landmark[20].x * flippedRGB.shape[1])
        y_little = int(results.multi_hand_landmarks[0].landmark[20].y * flippedRGB.shape[0])
        x_big = int(results.multi_hand_landmarks[0].landmark[4].x * flippedRGB.shape[1])
        y_big = int(results.multi_hand_landmarks[0].landmark[4].y * flippedRGB.shape[0])
        x_middle = int(results.multi_hand_landmarks[0].landmark[12].x * flippedRGB.shape[1])
        y_middle = int(results.multi_hand_landmarks[0].landmark[12].y * flippedRGB.shape[0])
        x_noname = int(results.multi_hand_landmarks[0].landmark[16].x * flippedRGB.shape[1])
        y_noname = int(results.multi_hand_landmarks[0].landmark[16].y * flippedRGB.shape[0])
        x_17 = int(results.multi_hand_landmarks[0].landmark[17].x * flippedRGB.shape[1])
        y_17 = int(results.multi_hand_landmarks[0].landmark[17].y * flippedRGB.shape[0])
        y_0 = int(results.multi_hand_landmarks[0].landmark[0].y * flippedRGB.shape[0])
        y_1 = int(results.multi_hand_landmarks[0].landmark[1].y * flippedRGB.shape[0])

        # Рисуем точку на кончике указательного пальца на отдельном слое finger_layer
        # Используем яркий цвет для точки (красный), чтобы она всегда была видна
        cv2.circle(finger_layer, (x_index, y_index), drawing_size + 2, color, -1)
        # Добавляем обводку белым для контраста
        cv2.circle(finger_layer, (x_index, y_index), drawing_size + 4, (255, 255, 255), 1)

        # Проверка взаимодействия с интерфейсом
        finger_on_main_icon = is_finger_on_icon(x_index, y_index, main_icon_x, main_icon_y, icon_size)

        # Логика меню - мгновенное переключение режимов
        if finger_on_main_icon and not menu_expanded:
            menu_expanded = True
        elif menu_expanded:
            menu_y_start = main_icon_y + 3
            mode_changed = False

            for i, item in enumerate(menu_icons):
                icon_y = menu_y_start + i * (icon_size + 3)
                if is_finger_on_icon(x_index, y_index, main_icon_x, icon_y, icon_size):
                    if current_mode != item["mode"]:
                        current_mode = item["mode"]
                        color = (255, 255, 255)
                        mode_changed = True
                        is_drawing = False
                        prev_x, prev_y = -1, -1
                        preview_layer.fill(0)

                        # Импорт файла
                        if current_mode == MODE_IMPORT and app_state == "drawing":
                            print("Открываем диалог выбора файла...")

                            root = Tk()
                            root.withdraw()
                            root.attributes('-topmost', True)

                            file_path = filedialog.askopenfilename(
                                title="Выберите чертеж",
                                filetypes=[
                                    ("GeoGebra файлы", "*.ggb"),
                                    ("Изображения", "*.png *.jpg *.jpeg *.bmp"),
                                    ("Все файлы", "*.*")
                                ]
                            )
                            root.destroy()

                            if file_path:
                                print(f"Выбран файл: {file_path}")

                                if file_path.lower().endswith('.ggb'):
                                    print("Парсим GeoGebra файл...")
                                    imported_image = parse_geogebra_file(file_path, w, h)
                                    if imported_image is None:
                                        print("Не удалось распарсить GGB, пробуем как изображение")
                                        imported_image = load_image_file(file_path, w, h)
                                else:
                                    imported_image = load_image_file(file_path, w, h)

                                if imported_image is not None:
                                    app_state = "positioning"
                                    transform_control = TransformControl(
                                        w // 4, h // 4,
                                        imported_image.shape[1],
                                        imported_image.shape[0],
                                        imported_image
                                    )
                                    print("Чертеж загружен! Режим позиционирования.")
                                    current_mode = MODE_CURVE
                                else:
                                    print("Не удалось загрузить файл")
                            else:
                                print("Файл не выбран")
                            current_mode = MODE_CURVE
                    break

            if not finger_on_main_icon and not mode_changed:
                finger_on_any_menu_icon = False
                for i, item in enumerate(menu_icons):
                    icon_y = menu_y_start + i * (icon_size + 3)
                    if is_finger_on_icon(x_index, y_index, main_icon_x, icon_y, icon_size):
                        finger_on_any_menu_icon = True
                        break

                if not finger_on_any_menu_icon:
                    menu_expanded = False

        # Режим позиционирования
        if app_state == "positioning" and transform_control is not None:
            fingers_touching = abs(x_big - x_index) < eps and abs(y_big - y_index) < eps

            if fingers_touching:
                if not was_interacting:
                    interaction_type = transform_control.check_interaction(x_index, y_index)
                    was_interacting = True
            else:
                if was_interacting:
                    was_interacting = False
                    interaction_type = None

            if was_interacting and interaction_type:
                if interaction_type == "move":
                    transform_control.move(x_index, y_index)
                elif interaction_type.startswith("resize"):
                    transform_control.resize(x_index, y_index, interaction_type)

            fingers_pinching = abs(x_big - x_middle) < eps and abs(y_big - y_middle) < eps
            if fingers_pinching and not was_interacting:
                print("Фиксация чертежа")
                transform_control.finalize(drawing_layer)
                app_state = "drawing"
                transform_control = None
                imported_image = None
                interaction_type = None
                was_interacting = False
        else:
            # Рисование
            hand_in_position = not (y_17 - y_0 > eps or k * (x_index - x_little) > -eps / 4 or (
                    y_1 - y_0) > eps or y_index - y_0 > -eps / 4)
            fingers_touching = abs(x_big - x_index) < eps and abs(y_big - y_index) < eps

            # Очистка корзиной
            bin_zone = 30
            if (0 < h - y_index < bin_zone and 0 < w - x_index < bin_zone) or (
                    0 < h - y_index < bin_zone and 0 < x_index < bin_zone):
                drawing_layer = np.zeros((h, w, 3), dtype=np.uint8)
                helped_picture = np.zeros((h, w, 3), dtype=np.uint8)
                laser_trails.clear()
                preview_layer.fill(0)
                is_drawing = False
                prev_x, prev_y = -1, -1

            if hand_in_position and fingers_touching and not menu_expanded and current_mode != MODE_IMPORT:
                if not is_drawing:
                    is_drawing = True
                    start_x, start_y = x_index, y_index
                    prev_x, prev_y = x_index, y_index

                    if current_mode == MODE_CURVE:
                        cv2.circle(drawing_layer, (x_index, y_index), drawing_size, color, -1)
                    elif current_mode == MODE_LASER:
                        # Для лазера добавляем точку с временем жизни
                        laser_trails.append(((x_index, y_index), (x_index, y_index), LASER_FADE_FRAMES))
                    elif current_mode == MODE_ERASER:
                        cv2.circle(drawing_layer, (x_index, y_index), 20, (0, 0, 0), -1)
                else:
                    if current_mode == MODE_CURVE:
                        if prev_x != -1 and prev_y != -1:
                            cv2.line(drawing_layer, (prev_x, prev_y), (x_index, y_index), color, drawing_size)
                        prev_x, prev_y = x_index, y_index
                    elif current_mode == MODE_LASER:
                        # Добавляем линию лазера с временем жизни
                        laser_trails.append(((prev_x, prev_y), (x_index, y_index), LASER_FADE_FRAMES))
                        prev_x, prev_y = x_index, y_index
                    elif current_mode == MODE_LINE:
                        # Рисуем предпросмотр на отдельном слое
                        cv2.line(preview_layer, (start_x, start_y), (x_index, y_index), color, drawing_size)
                        prev_x, prev_y = x_index, y_index
                    elif current_mode == MODE_CIRCLE:
                        # Рисуем предпросмотр на отдельном слое
                        radius = int(math.hypot(start_x - x_index, start_y - y_index))
                        cv2.circle(preview_layer, (start_x, start_y), radius, color, drawing_size)
                        prev_x, prev_y = x_index, y_index
                    elif current_mode == MODE_RECTANGLE:
                        # Рисуем предпросмотр на отдельном слое
                        cv2.rectangle(preview_layer, (start_x, start_y), (x_index, y_index), color, drawing_size)
                        prev_x, prev_y = x_index, y_index
                    elif current_mode == MODE_ERASER:
                        cv2.circle(drawing_layer, (x_index, y_index), 20, (0, 0, 0), -1)
                        prev_x, prev_y = x_index, y_index
            else:
                if is_drawing:
                    if current_mode == MODE_LINE:
                        cv2.line(drawing_layer, (start_x, start_y), (prev_x, prev_y), color, drawing_size)
                    elif current_mode == MODE_CIRCLE:
                        radius = int(math.hypot(start_x - prev_x, start_y - prev_y))
                        cv2.circle(drawing_layer, (start_x, start_y), radius, color, drawing_size)
                    elif current_mode == MODE_RECTANGLE:
                        cv2.rectangle(drawing_layer, (start_x, start_y), (prev_x, prev_y), color, -1)
                is_drawing = False
                prev_x, prev_y = -1, -1
                preview_layer.fill(0)  # Очищаем предпросмотр после фиксации

            # Выбор цвета
            if border_color[0] - y_index >= 0 and border_color[1] - x_index >= 0:
                if fl_color == 0:
                    fl_color = 1
                    border_color[0] = h // 2
                else:
                    selected_color = color_helped_picture[y_index][0]
                    color = (int(selected_color[0]), int(selected_color[1]), int(selected_color[2]))
                    if current_mode == MODE_ERASER:
                        color = (255, 255, 255)
            elif fl_color == 1:
                fl_color = 0
                border_color[0] = 30

    # Наложение слоев
    helped_picture[h - 60:, :] = picture_no_draw_zone
    flipped_BGR = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)

    # Сначала накладываем drawing_layer
    drawing_mask = cv2.cvtColor(drawing_layer, cv2.COLOR_BGR2GRAY)
    _, drawing_mask = cv2.threshold(drawing_mask, 1, 255, cv2.THRESH_BINARY)
    res_image = flipped_BGR.copy()
    res_image[drawing_mask == 255] = drawing_layer[drawing_mask == 255]

    # Затем накладываем helped_picture
    helped_mask = cv2.cvtColor(helped_picture, cv2.COLOR_BGR2GRAY)
    _, helped_mask = cv2.threshold(helped_mask, 1, 255, cv2.THRESH_BINARY)
    res_image[helped_mask == 255] = helped_picture[helped_mask == 255]

    # Рисуем лазерные линии на preview_layer
    laser_layer = np.zeros((h, w, 3), dtype=np.uint8)
    for trail in laser_trails:
        start_point, end_point, life = trail
        # Чем меньше life, тем тусклее линия
        alpha = life / LASER_FADE_FRAMES
        laser_color = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
        cv2.line(laser_layer, start_point, end_point, laser_color, drawing_size + 1)

    # Накладываем слой лазера
    laser_mask = cv2.cvtColor(laser_layer, cv2.COLOR_BGR2GRAY)
    _, laser_mask = cv2.threshold(laser_mask, 1, 255, cv2.THRESH_BINARY)
    res_image[laser_mask == 255] = laser_layer[laser_mask == 255]

    # Накладываем слой предпросмотра фигур
    preview_mask = cv2.cvtColor(preview_layer, cv2.COLOR_BGR2GRAY)
    _, preview_mask = cv2.threshold(preview_mask, 1, 255, cv2.THRESH_BINARY)
    res_image[preview_mask == 255] = preview_layer[preview_mask == 255]

    # Накладываем слой с точкой на пальце ПОВЕРХ всего
    finger_mask = cv2.cvtColor(finger_layer, cv2.COLOR_BGR2GRAY)
    _, finger_mask = cv2.threshold(finger_mask, 1, 255, cv2.THRESH_BINARY)
    res_image[finger_mask == 255] = finger_layer[finger_mask == 255]

    if app_state == "positioning" and transform_control is not None:
        transform_control.draw(res_image)
        cv2.putText(res_image, "", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Интерфейс выбора цвета
    if fl_color == 0:
        color_helped_picture[30:, :] = [0, 0, 0]
        color_helped_picture[:30, :30] = change_color
        res_image[:30, :30] = [0, 0, 0]
    else:
        color_helped_picture[:h // 2, :30] = color_line
        res_image[:h // 2, :30] = [0, 0, 0]

    # Отрисовка меню режимов
    res_image = draw_interface(res_image, current_mode, menu_expanded)

    # Добавляем палитру цветов
    res_image = res_image + color_helped_picture

    cv2.imshow("Hands", res_image)

cap.release()
cv2.destroyAllWindows()
handsDetector.close()