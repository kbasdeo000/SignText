import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img

from keras.models import Model

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