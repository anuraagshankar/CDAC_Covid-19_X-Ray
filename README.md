# Covid-19_X-Ray_Detection
## Model to predict if a person is infected with COVID-19 with the help of chest X-Rays.

The model is built using Transfer Learning and classifies 107 test images with **100% Accuracy**.

Requirements(Current Specifications):
1. Pandas - 0.25.1
2. Numpy  - 1.17.2
3. Keras  - 2.3.1
4. Tensorflow - 2.1.0

### Model Architecture:
The base pretrained model used for the Transfer Learning approach is the VGG-19 model.

The model consists of:

    VGG-19 Convolutional Layers -> FC1 (512 Activations) -> FC2 (128 Activations) -> FC3 (64 Activations) -> Softmax Classifier (0-Covid, 1-Non-Covid)
   
The model architecure (JSON File) and trained weights are available [here](https://drive.google.com/open?id=1r0kgUTdvZKuqQx9qNPoUPK8pzAOHIZ14).

### Dataset:
The dataset consists of X-Rays of multiple patients which is classified as given below:
1. COVID-19
2. Non-Covid:
   * Normal
   * Bacterial Pneumonia
   * Viral Pneumonia
   
Since the problem mainly concerns with COVID-19 patients, the latter 3 subcategories have been clubbed into 1 i.e. Non-Covid.

Examples:
1. COVID-19 X-Rays:

<img src="standby/covid1.jpeg" height=250 width=250>   <img src="standby/covid2.png" height=250 width=250>


2. Normal X-Rays:

<img src="standby/non_covid1.jpeg" height=250 width=250>   <img src="standby/non_covid2.jpeg" height=250 width=250>

The dataset has been split in the following fashion:
1. covid     - 219 images : 125 train, 47 validation, 47 test
2. non-covid - 367 images : 250 train, 57 validation, 60 test

The dataset is available [here](https://drive.google.com/open?id=1T5KHnOoKvsWXHbMlcTe8lcVWtdIqg6Ig).

The datasets have been obtained from the following resources:
- [IEEE 8023 Chest X-Ray](https://github.com/ieee8023/covid-chestxray-dataset)
- [Kaggle Pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

The dataset can be generated from scratch using the file *data.py* which takes two arguments:

    --covid     Path to the Covid GitHub directory.
    --non_covid Path to the Kaggle directory.
    
Directly using this dataset will lead to misclassification of images because the COVID-19 dataset consists of images such as CT scans, side view chest x-rays and marked images. Thus, it is suggested to directly use the dataset referenced above.

### Training:
The trained weights for the model have been referenced in the Model Architecture segment.
If you wish to train the model, the file *train.py* can be used which takes the following arguments:
    
    --data    Path of directory containing the dataset.
    --epochs  Number of epochs to run the training for.
    --early_stop Whether to stop the model from further training if val_loss starts to increase.
    --batch_size Batch size to train the data with.
    --weights Path to existing weights to train model further on pretrained weights.
    
### Evaluating:
The model gets a **100% Accuracy** on the 107 test images in the given dataset.

If you wish to evaluate a dataset, create two sub-directories:
* covid     - containing all COVID-19 scans
* non_covid - containing all other scans

The dataset can be evaluated using the file *evaluate.py* can be used, which produces the following metrics as an output:

    loss      The binary loss of the model's predictions with the actual data.
    accuracy  Percentage of images classified correctly.
    recall    Ratio of true positives to the sum of true positives and false negatives

### Predicting:
To make predictions on a dataset, first place all the images you want to test in a folder.

Then, the file *test.py* can be used to make predictions with the following arguments:

    --img_path   Path of the directory containing the images.
    --model_path Path of model JSON file.
    --weights    Path of model weights file.

The predictions will be stored in a csv file named *predictions.csv*
