from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

def model_prediction(test_image, model):
    image = Image.open(io.BytesIO(test_image))
    image = image.resize((64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
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
            model = tf.keras.models.load_model("trained_model.h5")
            result_index = model_prediction(image_bytes, model)
            with open("labels.txt") as f:
                labels = f.read().splitlines()
            prediction = labels[result_index] if labels and 0 <= result_index < len(labels) else "Unknown"
            return render_template('result.html', prediction=prediction)
    return render_template('upload.html')


@app.route('/about_us')  # Define the route for the about_us page
def about_us():
    return render_template('aboutus.html')

@app.route('/vision')
def vision():
    return render_template('vision.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(debug=True)
