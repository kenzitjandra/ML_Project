from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io


app = Flask(__name__)
model = load_model('model.h5')

from flask_cors import CORS
CORS(app)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # make sure it's RGB
    img = img.resize((150, 150))  # same size as during training
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)  # shape: (1, 224, 224, 3)


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img_bytes = file.read()
    img_tensor = preprocess_image(img_bytes)
    prediction = model.predict(img_tensor)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return jsonify({'class': int(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)
