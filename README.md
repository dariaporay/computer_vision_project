# геометрия в воздухе

#### Описание проекта:
Используя эту программу, можно рисовать на изображении во время съемки видео.
Сложив пальцы руки определенным способом, программа будет рисовать кривую (набор точек), окружность или отрезок, 
опираясь на положение указательного пальца. 
Также присутствует возможность стирать нарисованное (целиком или частично).
#### Установка проекта:
Загрузите все библиотеки из equirements.txt, картинку RecycleBin.png и основной код программы main.py.
#### Использование:
1. Во время рисования надо держать руку в плоскости, параллельной плоскости, снимающего устройства, 
также пальцы должны быть напрвлены вверх и ладонь в сторону камеры (в других положениях есть ограничения, 
рисовать не получится).
2. В зоне под серой линией рисовать не получится, эта функция заблокирована.
3. Рекомендуемое рассотяение между камерой и рукой 0.75 - 1,5 метра.
4. Освещение во время съемки должно быть не засвеченным, близким к белому по оттенку. Руку должно быть хороши видно. 
5. Надо помнить, что все рисуется от указательного пальца.
6. 4 положения руки для рисования (аналогичные для левой руки):
    * соединение большого палца и мизинца - рисование кривых (надор точек)
   ![рисование кривых](/images/hand_curve.jpg)
    * соединение большого и среднего пальцев - стирание
   ![стирание](/images/hand_rubber.jpg)
    * соединение большого и указательного пальцев - рисование окружностей (место соединения - центр окружности; 
   разъединения - точка, лежащая на оркужности)
   ![рисование окружностей](/images/hand_curcle.jpg)
    * соединение указательного и безымянного пальцев - рисование отрезков (место соединения - начало отрезка; 
   разъединения - конец отрезка)
   ![рисование отрезков](/images/hand_segment.jpg)
7. Можно стереть все, дотронувшись до картинки мусорки кончиком указательного пальца.
8. Для подключение сторонней камеры, замените 11 строчку кода на "cap = cv2.VideoCapture(1)".
#### Примеры:
1. Рисование кривых
   ![кривая](/images/curve.png)
2. Стирание
   ![стирание](/images/rubber.png)
3. Рисование окружностей
   ![окружность](/images/curcle.png)
4. Рисование отрезков
   ![отрезок](/images/segment.png)
5. примеры геометричейского чертежа
   ![пример](/images/example_1.png)
   ![пример](/images/example_2.png)
#### Статус проекта:
проект завершен