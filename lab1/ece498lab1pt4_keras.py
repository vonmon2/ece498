import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Conv2D(3, (5,5), input_shape=(28,28,1), strides=1, padding='valid', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=2, strides=2),
    keras.layers.Conv2D(3, (3, 3), input_shape=(12,12,3), strides=1, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=2, strides=2),
    keras.layers.Flatten(input_shape=(6,6)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

new_test = test_images.reshape(10000,28,28,1)
new_train = train_images.reshape(60000,28,28,1)

model.save('keras_model.h5') 

keep = model.fit(new_train, train_labels, epochs=15)

test_loss, test_acc = model.evaluate(new_test,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(new_test)

# Grab an image from the test dataset.
img = new_test[1]

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

predictions_single = probability_model.predict(img)

np.argmax(predictions_single[0])

#grab loss value for graph
tloss = keep.history['loss']

plt.plot(range(1, len(tloss)+1), tloss)
plt.ylabel('loss value')
plt.xlabel('epoch')
plt.show()