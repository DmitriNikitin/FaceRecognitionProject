import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.model import create_model

# Получение текущего рабочего каталога
current_dir = os.getcwd()

# Формирование пути к папке с данными для обучения
train_data_dir = os.path.join(current_dir, '..', 'data', 'train')

# Создание генератора изображений
image_generator = ImageDataGenerator(rescale=1./255)

# Загрузка изображений из папки с автоматической разметкой
train_data = image_generator.flow_from_directory(
    train_data_dir,
    target_size=(250, 250),
    batch_size=32,
    class_mode='categorical'
)

# Считывание модели из файла
model = create_model()

# Компиляция модели
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Обучение модели
model.fit(train_data, epochs=10)

# Сохранение модели
model.save('D:/Documents/Study/Apps/FaceRecognitionProject/src/saved_model.h5')

