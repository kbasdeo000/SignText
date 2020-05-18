# Getting a Prediction from trained model:
# (Python 3)

### STEP 0 - IMPORTS
# *** Using kera.models import load_model yields an error:
# AttributeError: '_thread._local' object has no attribute 'value'
# Use tensorflow import instead:
import os
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import cv2
from matplotlib import pyplot as plt
import numpy as np
import glob

def get_prediction(img_path):

    ### Step 0.1
    # this is the dictionary created at training time that associates a class with an in the output vector of the model
    class_indices = {'A': 0,
     'B': 1,
     'C': 2,
     'D': 3,
     'E': 4,
     'F': 5,
     'G': 6,
     'H': 7,
     'I': 8,
     'J': 9,
     'K': 10,
     'L': 11,
     'M': 12,
     'N': 13,
     'O': 14,
     'P': 15,
     'Q': 16,
     'R': 17,
     'S': 18,
     'T': 19,
     'U': 20,
     'V': 21,
     'W': 22,
     'X': 23,
     'Y': 24,
     'Z': 25,
     'del': 26,
     'nothing': 27,
     'space': 28}

    # this dictionary inverts the one above: ie gives you the class from the index
    # used to get the letter class from the argmax of a prediction vector.
    ind_to_class = {v: k for k, v in class_indices.items()}

    # approximate data mean and variance, computed on subset of training data
    (R_MEAN, G_MEAN, B_MEAN) = (132.54925778832765,127.47777489388343,131.40493902829613)
    (R_VAR, G_VAR, B_VAR) = (57.8223931610762,64.89711912659384,66.70657380138726)

    # STEP 1 - LOAD THE MODEL AND ITS WEIGHTS
    # path to model definition (replace with your own PATH)
    model_archi_path = os.getcwd() + '/asl-alphabet/slim-cnn-model_1589504755.7570796.archi.h5'

    # load the model architecture:
    model = load_model(model_archi_path)
    # check model summary
    # print(model.summary())

    # path to model weights (relies on the naming convention. ie that the file ends with 'archi.h5')
    # if you want to have different naming convention, you need to change the path to weights
    model_weights_path = model_archi_path[:-len('archi.h5')]+'weights.h5'
    # load weights
    model.load_weights(model_weights_path)

    # STEP 1.5 - LOAD IMAGE: (the file path is the path that gets fed to this function)
    # load image to python object
    img = image.load_img(img_path, target_size=(64,64))
    img = image.img_to_array(img, dtype='int')
    # print(img)
    # plt.imshow(img)
    # plt.show()

    ### STEP 2 - PREPROCESS IMAGE
    # at this stage, img should be a numpy array of shape (64,64,3)
    # with values between 0 and 255

    # normalize to mean 0 variance 1
    img = (img-np.mean(img))/np.std(img)
    # plt.imshow(img)
    # plt.show()

    # apply edge detection transform transform
    # img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    # img = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5)
    img = cv2.GaussianBlur(img, (7, 7), 0)
    img = cv2.Laplacian(img, cv2.CV_64F, ksize=5)

    # plt.imshow(img)
    # plt.show()

    # # renormalize?
    # img = (img-np.mean(img))/np.std(img)
    # plt.imshow(img)
    # plt.show()

    # check the shape of img
    # it should be (64, 64, 3) (64*64 pixels with 3 colors)
    print('img shape: {}'.format(np.shape(img)))
    # expand to have a batchsize of 1
    img = np.reshape(img, (1,64,64,3))

    ### STEP 3 - GET A PREDICTION
    pred_vector = model.predict(img)
    pred_class = ind_to_class[np.argmax(pred_vector)]

    # get class of prediction [A,B,C,...,Z,del, nothing]
    print('prediction vector: {}'.format(np.round(pred_vector, decimals=2)))
    print('file name: {}'.format(img_path.split('/')[-1]))
    print('prediction clsas: {}'.format(ind_to_class[np.argmax(pred_vector)]))


    if pred_class == 'nothing':
        return ""
    elif pred_class == 'space':
        return " "
    else:
        return pred_class
