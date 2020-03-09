import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img

from keras.models import Model

# Code to split the dataset 9/1 for test/validation

import split_folders

# Replace with the path of your local 'data_raw' directory
# For the 'data_raw' folder, use just the "train" folder from the original dataset.
input_folder = '/home/kchonka/Documents/SignText/data_raw'

# Split with a ratio:
# We're only spliting into training/validation 9/1, so ratio=(.9, .1).
# Output is going to go into a folder named "data_split"
# This split distribution can be recreated by using the same seed #
# Using the default seed # of 1337
split_folders.ratio(input_folder, output="data_split", seed=1337, ratio=(.9, .1))

# create an image generator
# allows us to load images gradually, only images needed immediately will be loaded

datagen = ImageDataGenerator()

# Test Path
train_path = '/Users/kelvin/Desktop/asl-alphabet/Train'

# Validate pathway
val_path = '/Users/kelvin/Desktop/asl-alphabet/Validate'

# Function that serves as in iterator for our separate image classes (train & test)

# load and iterate training dataset
test_it = datagen.flow_from_directory(train_path, target_size=(40,40),batch_size=1)
# load and iterate test dataset
val_it = datagen.flow_from_directory(val_path, target_size=(40,40), batch_size=9)

img = load_img('/Users/kelvin/Desktop/asl-alphabet/Validate/asl_alphabet_train/A/A1.jpg',grayscale=True)

# report details about the image
print(type(img))
print(img.format)
print(img.mode)
print(img.size)
# show the image
img.show()
