from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model = tf.keras.models.load_model("plant_disease_model.h5")

@app.route("/")
def home():
    return "Plant Disease Detection API Running"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read())).resize((224,224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    return jsonify({
        "class_index": int(class_index),
        "confidence": float(np.max(prediction))
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)