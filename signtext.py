# Flask API
# Back-end code

from flask import Flask, request, jsonify
from Keras_Code.Predict_From_Trained_Model import *

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
    # the image should be saved and have a path ?
    prediction = get_prediction(image)
    return jsonify(prediction)


# app.run()
if __name__ == '__main__':
    app.run()
