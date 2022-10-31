import tensorflow as tf
from tensorflow import keras
import numpy as np

#print(tf.__version__)
#print(tf.config.list_physical_devices('GPU'))

model = keras.models.load_model('my_model'
                                '')

doggy_path = "cat1.jpg"


class_names = ['cats', 'dogs']
img_height = 32
img_width = 32

img = tf.keras.utils.load_img(
    doggy_path, target_size=(img_height, img_width)

)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "Это изображение, скорее всего, принадлежит к {} с достоверностью {:.2f} процентов."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
