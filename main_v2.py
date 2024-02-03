import cv2
import dlib

# Создаем объект захвата видео (0 - индекс камеры, если у вас только одна камера)
cap = cv2.VideoCapture(0)

# Загружаем предобученный детектор лиц из dlib
detector = dlib.get_frontal_face_detector()

while True:
    # Захватываем кадр с камеры
    ret, frame = cap.read()

    # Преобразуем кадр в оттенки серого (для улучшения производительности)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Используем детектор лиц для поиска лиц на кадре
    faces = detector(gray)

    # Рисуем прямоугольники вокруг обнаруженных лиц
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Отображаем результат
    cv2.imshow('Face Detection', frame)

    # Выход из цикла по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
