from flask import Flask, request, render_template, redirect
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from flask import send_from_directory

app = Flask(__name__)

# Cache the model to avoid reloading on each request
model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model("trained_model.h5")
    return model

def model_prediction(test_image):
    model = load_model()
    image = Image.open(io.BytesIO(test_image))
    image = image.resize((64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            image_bytes = file.read()
            result_index = model_prediction(image_bytes)
            with open("labels.txt") as f:
                labels = f.read().splitlines()
            prediction = labels[result_index] if labels and 0 <= result_index < len(labels) else "Unknown"
            return render_template('result.html', prediction=prediction)
    return render_template('upload.html')

@app.route('/about_us')
def about_us():
    return render_template('aboutus.html')

@app.route('/vision')
def vision():
    return render_template('vision.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run()