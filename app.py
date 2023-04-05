from flask import Flask, request, jsonify
from keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)

# Load the model and labels
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

@app.route("/predict", methods=["POST"])
def predict():
    # Get the image file from the request
    image_file = request.files["image"]

    # Read the image file
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Resize the image to (224, 224)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Convert the image to a numpy array
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image
    image = (image / 127.5) - 1

    # Make a prediction using the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = float(prediction[0][index])

    # Return the prediction and confidence score as a JSON response
    return jsonify({"class": class_name, "confidence": confidence_score})

if __name__ == "__main__":
    app.run(debug=True)


