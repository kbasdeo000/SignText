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

# Translate page route:
@app.route('/translate', methods=['POST'])
@cross_origin(origin = 'https://signtext.ue.r.appspot.com')
def recieve_image():

    data_string = request.form['image']
    # data_string form: <str>: 'data:image/png;base64,base64 encoding itself'
    # Get substring, disregarding the first 21 positions
    img_string = data_string[22:]
    img = base64.b64decode(img_string)

    # Save the image & its path name
    img_file_name = "image.png"
    with open(img_file_name, 'wb') as f:
        f.write(img)
    cur_path = os.getcwd()
    img_path = cur_path + "/image.png"

    # Pass the path to the predictor function & return a prediction
    prediction = get_prediction(img_path)
    return jsonify(prediction)

# app.run()
if __name__ == '__main__':
    app.run()
