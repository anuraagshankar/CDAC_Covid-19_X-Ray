# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:07:19 2020

@author: Yash Sonar
"""

import keras
from keras import models
from keras.models import model_from_json
from keras import layers
from keras.applications import InceptionResNetV2
from keras import optimizers
from keras.layers.core import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

#json_file = open('./model3.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#model = model_from_json(loaded_model_json)
# load weights into new model
#model.load_weights('./weights00000003.h5')
#print("Loaded model from disk")

"""conv_base = InceptionResNetV2(weights = 'imagenet', include_top = False, input_shape=(299, 299, 3))
conv_base.trainable = False

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(96, activation='relu'))
model.add(layers.Dense(1, activation= 'sigmoid'))"""
json_file = open('./model3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.compile(loss = 'binary_crossentropy', optimizer = optimizers.Adam(lr = 0.0001), metrics = ['accuracy'])
model.load_weights('./weights00000003.h5')

test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory('../xr_inv',
                                 target_size=(299, 299),
                                 batch_size=1,
                                 class_mode = 'binary', shuffle=False
                                 )
res = model.predict_generator(test_set)

print(res)