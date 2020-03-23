# ********************************************************************
# Title: Classifying Images of the ASL Alphabet using Keras
# Author: Dan Rasband
# Date: August 16, 2018
# Code Version: 17
# Availability: https://www.kaggle.com/danrasband/classifying-images-of-the-asl-alphabet-using-keras
# *******************************************************************

# Imports for Deep Learning
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator

# Ensure consistency across runs
from numpy.random import seed
import random
seed(2)
#from tensorflow import set_random_seed
# set_random_seed(2)

# Imports to view data
import cv2
from glob import glob

# Metrics
from sklearn.metrics import classification_report, confusion_matrix

# Visualization
from keras.utils import print_summary
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

# Utils
from pathlib import Path
import pandas as pd
import numpy as np
from os import getenv
import time
import itertools

# Image Preprocessing
from skimage.filters import sobel, scharr


# In[2]:


# Set global variables
TRAIN_DIR = '/home/kchonka/Documents/SignText/data_split/train'
TEST_DIR = '/home/kchonka/Documents/SignText/data_split/val'
CUSTOM_TEST_DIR = '../input/asl-alphabet-test/asl-alphabet-test'
CLASSES = [folder[len(TRAIN_DIR) + 1:] for folder in glob(TRAIN_DIR + '/*')]
CLASSES.sort()

TARGET_SIZE = (64, 64)
TARGET_DIMS = (64, 64, 3) # add channel for RGB
N_CLASSES = 29
VALIDATION_SPLIT = 0.1
BATCH_SIZE = 64

# Model saving for easier local iterations
MODEL_DIR = '/home/kchonka/Documents/SignText'
MODEL_PATH = MODEL_DIR + '/cnn-model.h5'
MODEL_WEIGHTS_PATH = MODEL_DIR + '/cnn-model.weights.h5'
MODEL_SAVE_TO_DISK = getenv('KAGGLE_WORKING_DIR') != '/kaggle/working'

print('Save model to disk? {}'.format('Yes' if MODEL_SAVE_TO_DISK else 'No'))


# In[6]:


def load_model_from_disk():
    '''A convenience method for re-running certain parts of the
    analysis locally without refitting all the data.'''
    model_file = Path(MODEL_PATH)
    model_weights_file = Path(MODEL_WEIGHTS_PATH)

    if model_file.is_file() and model_weights_file.is_file():
        print('Retrieving model from disk...')
        model = load_model(model_file.__str__())

        print('Loading CNN model weights from disk...')
        model.load_weights(model_weights_file)
        return model

    return None

CNN_MODEL = load_model_from_disk()
REPROCESS_MODEL = (CNN_MODEL is None)

print('Need to reprocess? {}'.format(REPROCESS_MODEL))


# In[7]:


print_summary(CNN_MODEL)


# In[54]:


X = X.flatten()
X = X[:(64*64*3)]
print(X.shape)
X = np.reshape(X, (-1, 64, 64, 3))


# In[55]:


CNN_MODEL.predict(X)


# In[21]:


cls = CLASSES[0]
img_path = TRAIN_DIR + '/' + cls + '/**'
path_contents = glob(img_path)

imgs = random.sample(path_contents, 1)

X = cv2.imread(imgs[0])

print(X)


# In[22]:


CNN_MODEL(X)


# In[23]:


def preprocess_image(image):
    '''Function that will be implied on each input. The function
    will run after the image is resized and augmented.
    The function should take one argument: one image (Numpy tensor
    with rank 3), and should output a Numpy tensor with the same
    shape.'''
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    return sobely


# In[28]:


new_x = preprocess_image(X)


# In[36]:


new_x = ImageDataGenerator(new_x)
# new_x = next(new_x)
new_x[0]


# In[30]:


CNN_MODEL(new_x)
