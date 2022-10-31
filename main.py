import tensorflow as tf
from matplotlib import pyplot as plt

from tensorflow import keras
from keras import layers
from keras.models import Sequential
import pathlib


print(tf.__version__)
data_dir = "cats"
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*/*.jpg')))
print(image_count)

batch_size = 32 # image_batch представляет собой
                # тензор формы (32, 32, 32, 3).
                # Это пакет из 32 изображений размером 32x32x3
                # (последний размер относится к цветовым каналам RGB)
img_height = 32
img_width = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    pathlib.Path('cats' + '/training_set'),  #для обучения
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    pathlib.Path('cats' + '/test_set'),  #для тестирования
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)


class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE  #Dataset.cache хранит изображения в памяти после их загрузки с диска в течение первой эпохи.
#Dataset.prefetch перекрывает предварительную обработку данных и выполнение модели во время обучения.

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


#Применения преобразования изображений (Data Augmentation)
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                          input_shape=(img_height,
                                       img_width,
                                       3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

num_classes = len(class_names)

#Используем последовательную модель, каждый слой имеет ровно один тензор входной и один выходной тензор
model = Sequential([
#    data_augmentation,
    layers.Rescaling(1. / 255), # Слои масштабирования
    layers.Conv2D(16, 3, padding='same', activation='sigmoid'), #Добавления операции свертки^ на выходе 16 признаков изображения, 3 размер ядра свертки(3х3)
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='sigmoid'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='sigmoid'),
    layers.MaxPooling2D(), #всего пулинг — это извлечение максимума/среднего элементов,
                           # попадающих под окно пулинга (область фиксированноо размера, например, окно 2х2).
    layers.Dropout(0.2), # Отключит нейрон с вероятностью 0.2
    layers.Flatten(), #Flatten, преобразует формат изображений из двумерного массива в одномерный массив.
    layers.Dense(128, activation='relu'), #полносвязный слой
                                        # ( tf.keras.layers.Dense ) со 128 единицами поверх него,
                                        # который активируется функцией активации ReLU ( 'relu' )
    layers.Dense(num_classes, name="outputs")
])

model.compile(optimizer='adam', #оптимизатор
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #функция потерь
              metrics=['accuracy']) #просмотр точности обучения и проверки для каждой эпохи обучения

#model.summary() # просмотр всех слоев

#Обучение модели, анализ точности и потери
epochs = 100 # количество эпох
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

model.save('my_model') # Сохранения текущей модели

# Визуализация результатов тренировок
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
