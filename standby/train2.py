import os
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import argparse

image_shape = 299 # Original size of the InceptionResnetV2 model

def model_architecture():
    model = Sequential()
    resnet_layer = InceptionResNetV2(include_top = False, weights = 'imagenet', input_shape = (image_shape,image_shape,3))
    model.add(resnet_layer)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(96, activation='relu'))
    model.add(Dense(1, activation= 'sigmoid'))
    model.layers[0].trainable = False
    return model

def train(args):
    data_path = args.data
    epochs = args.epochs
    early_stop = args.early_stop
    batch_size = args.batch_size
    weights = args.weights

    model = model_architecture()
    model.summary()

    data_gen = ImageDataGenerator(rescale=1./255)
    train_it = data_gen.flow_from_directory(data_path + '/train', 
        target_size = (image_shape,image_shape), batch_size = batch_size, class_mode = 'binary')
    val_it = data_gen.flow_from_directory(data_path + '/val', 
        target_size = (image_shape,image_shape), batch_size = 20, class_mode = 'binary')
    test_it = data_gen.flow_from_directory(data_path + '/test', 
        target_size = (image_shape,image_shape), batch_size = 1, class_mode = 'binary', shuffle=False)

    model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 0.0001), metrics = ['accuracy'])

    if weights is not None:
        model.load_weights(weights)

    try: os.mkdir('logs')
    except: pass
    filepath="logs/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, period=2)
    callbacks = [checkpoint]
    if early_stop:
        early_stopping = EarlyStopping(patience = 2, restore_best_weights = True)
        callbacks.append(early_stopping)

    steps = 375//batch_size

    model.fit_generator(train_it, steps_per_epoch=steps, epochs=epochs, callbacks=callbacks, validation_data=val_it, validation_steps=3)

    os.mkdir('model')
    model_json = model.to_json()
    with open("model/model_final.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights('model/model_final.hdf5')

    print("MODEL TRAINED")

    test_loss, test_acc = model.evaluate_generator(test_it)
    print("Test Results: ")
    print("Loss: " + str(test_loss))
    print("Test: " + str(test_acc))
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data',
                        help='Path of directory containing the dataset.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to run the training for.')
    parser.add_argument('--early_stop', type=bool, default=True,
                        help='Whether to stop the model from further training if val_loss starts to increase(Bool).')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to train the data with.')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to existing weights to train model further on pretrained weights.')
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()