import tensorflow as tf
from tensorflow.keras import layers

def create_model():
    # Определение модели CNN
    model = tf.keras.Sequential()

    # Слои свертки и пулинга
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(250, 250, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Преобразование данных в плоский вектор
    model.add(layers.Flatten())

    # Полносвязные слои
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    return model
