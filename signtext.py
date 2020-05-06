# Flask API
# Back-end code

import os
import cv2
from flask import Flask, request, jsonify, redirect
from Predict_From_Trained_Model import *
from flask_cors import CORS, cross_origin # Allows Cross Origin Resource Sharing
from PIL import Image

# initialize a flask object
app = Flask(__name__)
CORS(app, support_credentials=True)
#app.config["DEBUG"] = True
app.config['CORS_HEADERS'] = 'Content-Type'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create a directory in a known location to save image files to:
uploads_dir = os.path.join(app.instance_path, 'uploads')
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

# Home page route:
@app.route('/')
def home():
    return "home"   # dummy example


# Translate page route:
@app.route('/translate', methods=['POST'])
@cross_origin(origin = '*')
def get_image():

    prediction = None

    #if request.method == 'POST':
    #image = request.get_data()
    #pil_image = open(image, 'wb')
    file = request.form['image']
    print(type(file))
    print(file)
    #img = cv2.imread(file)

    # Get image path:
    curr_path = os.getcwd()
    print(curr_path)
    image_path = curr_path + 'image.jpg'
    #prediction = get_prediction(image_path)

    return jsonify("yes")

    '''
    if image:
        pil_image = Image.open(image)
        return jsonify("yes")
    '''    '''
        image.save(os.path.join(uploads_dir, secure_filename(image.filename)))
        # Get image path:
        curr_path = os.getcwd()
        image_path = curr_path + secure_filename

        prediction = get_prediction(image_path)
        return jsonify(prediction)
        '''

# app.run()
if __name__ == '__main__':
    app.run()
