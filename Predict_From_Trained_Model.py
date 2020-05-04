# Getting a Prediction from trained model:
# (Python 3)

### STEP 0 - IMPORTS
import os
from keras.models import load_model
from keras.preprocessing import image
import cv2
from matplotlib import pyplot as plt
import numpy as np

def preprocess_image(image):
    '''Function that will be implied on each input. The function
    will run after the image is resized and augmented.
    The function should take one argument: one image (Numpy tensor
    with rank 3), and should output a Numpy tensor with the same
    shape.'''
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    return sobely

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

    # STEP 1 - LOAD THE MODEL AND ITS WEIGHTS
    # path to model definition (replace with your own PATH)
    MODEL_DIR = os.getcwd() # + '/asl-alphabet'
    model_def_path = MODEL_DIR + '/slim-cnn-model.h5'

    # load the model
    model = load_model(model_def_path)

    # path to model weights
    model_weights_path = MODEL_DIR + '/slim-cnn-model.weights.h5'
    # load weights
    model.load_weights(model_weights_path)


    # STEP 1.5 - LOAD IMAGE: (Take this from webcam)
    # load image to python object
    img = image.load_img(img_path, target_size=(64,64))
    img = image.img_to_array(img, dtype='int')
    # print(img)
    # plt.imshow(img)
    # plt.show()

    ### STEP 2 - PREPROCESS IMAGE
    # at this stage, img should be a numpy array of shape (64,64,3)
    # with values between 0 and 255
    # your task is to get this img from the webcam


    # normalize to mean 0 variance 1
    img = (img-np.mean(img))/np.std(img)
    # plt.imshow(img)
    #plt.show()

    img = preprocess_image(img)
    # plt.imshow(img)
    # plt.show()
    img = (img-np.mean(img))/np.std(img)

    # check the shape of img
    # it should be (64, 64, 3) (64*64 pixels with 3 colors)
    # print('img shape: {}'.format(np.shape(img)))
    # plt.imshow(img)
    # plt.show()

    # expand to have a batchsize of 1
    img = np.reshape(img, (1,64,64,3))

    ### STEP 3 - GET A PREDICTION
    pred_vector = model.predict(img)
    pred_class = ind_to_class[np.argmax(pred_vector)]

    # get class of prediction [A,B,C,...,Z,del, nothing]
    print('prediction vector: {}'.format(np.round(pred_vector, decimals=2)))
    print('file name: {}'.format(img_path.split('/')[-1]))
    print('prediction clsas: {}'.format(ind_to_class[np.argmax(pred_vector)]))

    return pred_class
