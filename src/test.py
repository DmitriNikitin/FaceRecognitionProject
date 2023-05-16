import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

# Определение корневого каталога
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Загрузка обученной модели
model_path = os.path.join(base_dir, 'src', 'saved_model.h5')
model = tf.keras.models.load_model(model_path)

# Путь к папке с тестовыми изображениями
test_data_dir = os.path.join(base_dir, 'data', 'test')

# Создание генератора изображений для тестовых данных
test_data_generator = ImageDataGenerator(rescale=1. / 255)

test_data = test_data_generator.flow_from_directory(
    directory=test_data_dir,
    target_size=(250, 250),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Предсказание классов для тестовых изображений
predictions = model.predict(test_data)

# Расшифровка предсказаний
predicted_labels = ['me' if np.argmax(prediction) == 0 else 'not me' for prediction in predictions]

# Вывод результатов
for filename, label, prediction in zip(test_data.filenames, predicted_labels, predictions):
    # Полный путь к изображению
    image_path = os.path.join(test_data_dir, filename)

    # Загрузка изображения с помощью OpenCV
    image = cv2.imread(image_path)

    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Загрузка классификатора для обнаружения лиц
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Обнаружение лиц на изображении
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Отрисовка рамок вокруг обнаруженных лиц
    for (x, y, w, h) in faces:
        # Обводка лица в квадратную рамку
        if label == 'me':
            color = (0, 255, 0)  # Зеленая рамка для вас
        else:
            color = (0, 0, 255)  # Красная рамка для неизвестного лица

        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=2)

        # Добавление подписи над рамкой
        if label == 'me':
            text = 'Dmitri'
        else:
            text = 'unknonw'

        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness=2)

    # Отображение изображения с рамками
    cv2.imshow(filename, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
