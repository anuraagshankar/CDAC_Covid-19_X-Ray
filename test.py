import os
import pandas as pd
import numpy as np
import argparse
from keras.models import model_from_json
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.preprocessing import image

def convert(x):
    if x == 0: return 'Positive'
    else: return 'Negative'

def getImages(image_list, path):
    images = []
    count = 0
    for img in image_list:
        im = image.load_img(path+'/'+img, target_size=(299,299))
        im = image.img_to_array(im)
        im = preprocess_input(im)
        images.append(im)
        count += 1
        if count == 10:
            yield np.array(images)
            count = 0
            images = []
            
    yield np.array(images)
    
def predict(args):
    data_path = args.img_path
    model_path = args.model_path
    model_weights = args.weights
    
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_weights)

    image_list = sorted(os.listdir(data_path))
    images = getImages(image_list, data_path)
    preds = []
    for batch in images:
        pred = model.predict(batch)
        preds.extend(pred)
    preds = np.argmax(preds, axis=1)
    image_preds = [(image_list[i], convert(preds[i])) for i in range(len(image_list))]
    return image_preds

def preds_to_csv(preds):
    df = pd.DataFrame(preds)
    df.rename(columns = {0:'Image Name', 1: 'Prediction'}, inplace=True)
    df['Sr. No.'] = np.arange(1, len(df) + 1)
    df.set_index('Sr. No.', inplace=True)
    df.to_csv('predictions.csv')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='data',
                        help='Path of the directory containing the images.')
    parser.add_argument('--model_path', type=str, default='model.json',
                        help='Path of model JSON file.')
    parser.add_argument('--weights', type=str, default='model.hdf5',
                        help='Path of model weights file(HDF5 or H5).')
    args = parser.parse_args()
    res = predict(args)
    preds_to_csv(res)
    print('PREDICTIONS STORED IN predictions.csv')

if __name__ == "__main__":
    main()