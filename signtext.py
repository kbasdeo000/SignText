# Flask API
# Back-end code

import os
from flask import Flask, request, jsonify, redirect
from Predict_From_Trained_Model import *

# initialize a flask object
app = Flask(__name__)
app.config["DEBUG"] = True
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
@app.route('/translate', methods=['GET','POST'])
def get_image():

    prediction = ''

    if request.method == 'POST':
        image = request.files['image']

        # if user does not select file, browser also
        # submit an empty part without filename
        if image.filename == '':
            return redirect(request.url)

        if image and allowed_file(image.filename):
            image.save(os.path.join(uploads_dir, secure_filename(image.filename)))
            # Get image path:
            curr_path = os.getcwd()
            image_path = curr_path + secure_filename

            prediction = get_prediction(image_path)
            return jsonify(prediction)

    return jsonify(prediction)  # Default GET return

# app.run()
if __name__ == '__main__':
    app.run()
