from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Ensure the 'uploads' directory exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Load the pre-trained model
model = tf.keras.models.load_model("car_classification_model.h5")

# Load class names from CSV file
names_path = "C:\\Users\\a883714\\.cache\\kagglehub\\datasets\\jutrera\\stanford-car-dataset-by-classes-folder\\versions\\2\\names.csv"
class_names = pd.read_csv(names_path, header=None)[0].tolist()

def prepare_image(image_path):
    """Preprocess the image to match model input size and scale."""
    img = Image.open(image_path).resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the file from post request
        file = request.files["file"]
        if file:
            # Save the file locally for prediction
            image_path = os.path.join("uploads", file.filename)
            file.save(image_path)

            # Preprocess and predict
            img = prepare_image(image_path)
            predictions = model.predict(img)
            class_id = np.argmax(predictions)
            class_name = class_names[class_id]

            # Render the result on the HTML page
            return render_template("index.html", class_name=class_name)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
