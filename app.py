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
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((150, 150))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img_bytes = file.read()
    img_tensor = preprocess_image(img_bytes)

    raw_output = model.predict(img_tensor)[0]  # shape: (num_classes,)
    probabilities = tf.nn.softmax(raw_output).numpy()

    top_indices = np.argsort(probabilities)[::-1][:3]
    class_map = {
        0: "Buildings",
        1: "Forests",
        2: "Mountains",
        3: "Glacier",
        4: "Street",
        5: "Sea"
    }

    top_predictions = [
        {
            "label": class_map[i],
            "confidence": round(probabilities[i] * 100, 2)
        }
        for i in top_indices
    ]

    return jsonify({"top_predictions": top_predictions})

if __name__ == '__main__':
    app.run(debug=True)
