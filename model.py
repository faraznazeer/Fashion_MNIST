
#importing required modules
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


#importing fashion mnist dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# Defining classes
class_name = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#Scaling the data
train_images = train_images / 255.0
test_iamges = test_images / 255.0


#build model
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)),
	keras.layers.Dense(128, activation=tf.nn.relu),
	keras.layers.Dense(10, activation=tf.nn.softmax)
	])


#compile model
model.compile( optimizer = 'adam',
			   loss = 'sparse_categorical_crossentropy',
			   metrics = ['accuracy'])

#train model
model.fit( train_images, train_labels, epochs = 5 , verbose = 1 )

#evaluate model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('\n Test accuracy:', test_accuracy)

#save the model
model.save('keras_trained_model.hdf5')
print("\n Model saved to trained_model.hdf5")