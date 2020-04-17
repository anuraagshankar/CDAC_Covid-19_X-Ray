from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import model_from_json
import numpy as np
import os
import argparse

model_path = 'model.json'
model_weights = 'model.h5'
path = 'test'

covid_count = len(os.listdir(path+'/covid'))
non_covid_count = len(os.listdir(path+'/non_covid'))
total = covid_count + non_covid_count

def loss(preds):
    l = 0
    for i in range(covid_count):
        l += np.log10(1- preds[i][0])
    for i in range(covid_count, total):
        l += np.log10(preds[i][0])
    return -l

def recall(preds):
    true_positive = covid_count
    false_negative = 0
    for i in range(covid_count):
        if preds[i] == 1: false_negative += 1
    return true_positive,true_positive+false_negative

def accuracy(preds):
    false = 0
    for i in range(covid_count):
        if preds[i] == 1: false += 1
    for i in range(non_covid_count, total):
        if preds[i] == 0: false += 1
    return total-false, total

def predict():
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    model.load_weights(model_weights)

    data_gen = ImageDataGenerator(rescale=1./255)
    test_it = data_gen.flow_from_directory(path,
                                 target_size=(299, 299),
                                 batch_size=1,
                                 class_mode = 'binary', shuffle=False
                                 )
    model.compile(loss = 'binary_crossentropy', optimizer = optimizers.Adam(lr = 0.0001), metrics = ['accuracy'])    
    
    return model.predict_generator(test_it)

def evaluate():
    res = predict()
    print(res)
    around_res = np.around(res)

    l = loss(res)
    acc = accuracy(around_res)
    rec = recall(around_res)
    
    return l, acc, rec

l, acc, rec = evaluate()
print(f'Loss: {l}')
print(f'Accuracy: {acc[0]}/{acc[1]}:- {acc[0]/acc[1]}')
print(f'Recall: {rec[0]}/{rec[1]}:- {rec[0]/rec[1]}')