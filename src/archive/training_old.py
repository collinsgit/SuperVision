# -*- coding: utf-8 -*-
"""

Trains the neural network

"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Reshape, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import requests
import os
import os.path
import collections
import json
from bs4 import BeautifulSoup as Soup
import shutil
from random import shuffle
import urllib.parse
from PIL import Image
import imghdr

file_dir = os.path.abspath(os.path.split(__file__)[0])
data_dir = "data"
cache_dir = "cached_data"
checkpoint_dir = "checkpoint"
def path_for(*p):
	full_path = os.path.join(file_dir, *p)
	full_path_dir = os.path.split(full_path)[0]
	os.makedirs(full_path_dir, exist_ok=True)
	return full_path



class PlotLosses(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []
		self.loss_fig = plt.figure(1)
		self.plot = True

	def on_batch_end(self, batch, logs={}):
		if batch > 1:
			self.losses.append(logs.get('loss'))

	def on_epoch_end(self, batch, logs={}):
		if self.plot:
			self.loss_fig.clear()
			plt.figure(1)
			plt.plot(self.losses)
			plt.title('Training Loss')
			plt.pause(0.0001)


def debug():
	loader = ImageNetLoader()
	structure = SynSet.populate_structure()
	loader.clear_cache()
	loader.cache_synsets_of_depth(5,10,20)


def is_image_name(name):
	return '.' in name and name.split('.')[-1].lower() in ('jpg', 'jpeg', 'png')


def load_images(image_shape, final_shape):
	images = []

	synsets = set(os.listdir('cached_data'))
	for synset in synsets:
		path = os.path.join('cached_data', synset)
		image_names = set(filter(is_image_name, os.listdir(path)))
		for image_name in image_names:
			image = load_img(os.path.join(path, image_name))
			images.append((image.resize(image_shape),
						synset,
						image.resize(final_shape)))

	shuffle(images)

	one_hot = np.zeros((len(images), len(synsets)))
	index = 0
	one_hot_mapping = {}
	for synset in synsets:
		one_hot_mapping[synset] = index
		index += 1

	for i in range(len(images)):
		one_hot[i][one_hot_mapping[images[i][1]]] = 1

	print(one_hot_mapping)

	return np.array([img_to_array(image[0]) for image in images]), \
		one_hot, \
		np.array([img_to_array(image[2]) for image in images])



def main():
	image_size = (16, 16, 3)
	activation = 'relu'
	final_shape = (128, 128, 3)
	num_classes = 18

	image_input = Input(shape=image_size)
	label_input = Input(shape=(num_classes,))

	x = Conv2D(32, 3, activation=activation)(image_input)

	x = Conv2D(64, 3, activation=activation)(x)

	x = Conv2D(128, 3, activation=activation)(x)

	x = Dropout(0.4)(x)

	x = Flatten()(x)
	x = Dense(16*16*3, activation=activation)(
		concatenate([x, label_input]))

	x = Dense(np.prod(final_shape), activation=activation)(
		concatenate([x, Flatten()(image_input)]))

	x = Reshape(final_shape)(x)

	output = x
				
	model = keras.models.Model(inputs=[image_input, label_input], outputs=output)

	print(model.summary())
				
	adam = Adam(lr=1e-4)

	plot_losses = PlotLosses()

	model.compile(loss=keras.losses.mean_squared_error,
				optimizer=adam,
				metrics=['accuracy'])

	train_images, train_labels, train_original_images = load_images(image_size[:-1], final_shape[:-1])

	history = model.fit([train_images, train_labels], train_original_images,
						validation_split=0.3,
						epochs=100,
						batch_size=10)

	max_val_acc = max(history.history['val_acc'])
	print(max_val_acc)

	model.save('my_model.h5')


def test_model():
	model = keras.models.load_model('my_model.h5')

	image = load_img('cached_data/n04227787/5.jpg')
	# image = load_img('bridge.JPG')
	image.save('original_image.jpg')

	smaller_image = image.resize((128, 128))
	smaller_image.save('smaller_image.jpg')

	image = image.resize((16, 16))
	image.save('little_image.jpg')
	label = np.zeros((1, 18))
	label[0, 15] = 1
	print(label)
	image = model.predict([keras.backend.stack([image]), label], steps=1)
	array_to_img(image[0]).save('predicted_image.jpg')


if __name__ == '__main__':
	main()
