"""
    Prepare dataset from scratch from complete dataset of covid and pnuemonia chest x-rays.
    Covid dataset from GitHub contains both CT and X-Ray scans(some of which are unclear).
    If manual deletion of unclear data is not done, it wil lead to misleading training and classification.
    Therefore it is advised to use the dataset directly. This code is provided for reference.
    All images used in the dataset will be stored in the folders 'covid_images' and 'non_covid_images'.
    Ideal splits:
        Covid - Total:164 => Train: 125, Val: 19, Test: 20
        Non-Covid - Total:350 => Train: 250, Val: 40, Test: 60
"""
import os
import pandas as pd
from shutil import copyfile
import argparse
import numpy as np

def generateCovidDataset(covid_path):
    
    metadata = pd.read_csv(covid_path + '/metadata.csv')
    covid = metadata[metadata['finding'] == 'COVID-19']
    covid_images = list(covid['filename'])
    np.random.shuffle(covid_images)
    count = 0
    src_path = covid_path + '/images/'
    os.mkdir('covid_images')
    dst_path = 'covid_images/'
    for image in covid_images:
        try:
            src = src_path + image
            dst = dst_path + str(count) + '.' + image.split('.')[-1]
            copyfile(src, dst)
            count += 1
        except:
            pass
    # The images directory contains X-Rays(some unclear because of markings in the scan), CT scans.
    # Manual removal of CT-scans and marked X-rays is required, which reduces size from 216 to 164.
    # If only clear images are kept, it leads to a split of 125,19,20 for train,val,test.
    image_names = os.listdir('covid_images')
    
    test_images = image_names[:20]
    val_images = image_names[20:39]
    train_images = image_names[39:]
    
    os.mkdir('data')
    os.mkdir('data/train')
    os.mkdir('data/test')
    os.mkdir('data/val')
    os.mkdir('data/train/covid')
    os.mkdir('data/test/covid')
    os.mkdir('data/val/covid')

    src = 'covid_images/'
    dst = 'data/'
    for image in train_images:
        copyfile(src + image, dst + 'train/covid/' + image)
    for image in test_images:
        copyfile(src + image, dst + 'test/covid/' + image)
    for image in val_images:
        copyfile(src + image, dst + 'val/covid/' + image)

    print('COVID DATA GENERATED.')

def generateNonCovidDataset(non_covid_path):

    # Taking data only from training set because it size of > 1000 which is more than enough for training.
    non_covid_images = [non_covid_path + '/train/NORMAL/' + image for image in os.listdir(non_covid_path + '/train/NORMAL')]
    non_covid_images += [non_covid_path + '/train/PNEUMONIA/' + image for image in os.listdir(non_covid_path + '/train/PNEUMONIA')]
    np.random.shuffle(non_covid_images)
    non_covid = non_covid_images[:350]
    os.mkdir('non_covid_images')

    count = 0
    for src in non_covid:
        dst = 'non_covid_images/' + str(count) +'.' + src.split('.')[-1]
        copyfile(src, dst)
        count += 1
    
    os.mkdir('data/train/non_covid')
    os.mkdir('data/test/non_covid')
    os.mkdir('data/val/non_covid')

    non_covid_data = os.listdir('non_covid_images')
    train_images = non_covid_data[:250]
    test_images = non_covid_data[250:310]
    val_images = non_covid_data[310:]

    src = 'non_covid_images/'
    dst = 'data/'
    for image in train_images:
        copyfile(src + image, dst + 'train/non_covid/' + image)
    for image in test_images:
        copyfile(src + image, dst + 'test/non_covid/' + image)
    for image in val_images:
        copyfile(src + image, dst + 'val/non_covid/' + image)

    print('NON COVID DATA GENERATED.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--covid', type=str, default='covid',
                        help='Path of the GitHub repository having COVID-19 data.')
    parser.add_argument('--non_covid', type=str, default='non_covid',
                        help='Path of the Kaggle pneumonia competition dataset.')
    args = parser.parse_args()
    covid_path = args.covid
    non_covid_path = args.non_covid
    generateCovidDataset(covid_path)
    generateNonCovidDataset(non_covid_path)

if __name__ == '__main__':
    main()