# Flask API
# Back-end code

import os
from flask import Flask, request, jsonify
from Predict_From_Trained_Model import *

# initialize a flask object

app = Flask(__name__)
# app.config["DEBUG"] = True


# Home page route:
@app.route('/')
def home():
    return "home"   # dummy example


# Translate page route:
@app.route('/translate', methods=['POST'])
def get_image():
    image = request.files['file']

    # Get image path:
    image_path = os.getcwd()

    prediction = get_prediction(image_path)
    return jsonify(prediction)

# app.run()
if __name__ == '__main__':
    app.run()
