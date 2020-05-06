# Flask API
# Back-end code

import os
import cv2
from flask import Flask, request, jsonify, redirect
from Predict_From_Trained_Model import *
from flask_cors import CORS, cross_origin # Allows Cross Origin Resource Sharing
from PIL import Image
import base64

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
def recieve_image():

    prediction = None

    image_data = request.form['image']
    # print(type(image_data)) --> <str>
    # Form: <str>: 'data:image/png;base64, base64 encoding itself'
    # img = base64.b64decode(image_data) ----> doesn't work

    # Methodology:
    # Decode image
    # Save & get path
    # Pass the img path to Predict_From_Trained_Model.get_prediction
    # Return prediction

    return jsonify(prediction)

# app.run()
if __name__ == '__main__':
    app.run()
