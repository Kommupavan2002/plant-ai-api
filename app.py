from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route("/")
def home():
    return "Plant Disease Detection API Running (TFLite)"

def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img = preprocess_image(img)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return jsonify({
        "class_index": class_index,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)