import numpy as np
import os
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import cv2


flask_app = Flask(__name__)

#loading model
model = pickle.load(open("main_pickle_save.pkl", "rb")) #rb = read (in binary)

#whenver user reaches this route, it will render index.html
@flask_app.route("/")
def Home():
    return render_template("index.html")

#route for predict
@flask_app.route("/predict", methods=["POST"]) #post is better to send large amount of data securely
def predict():
    file = request.files["Image"]

    try:
        file_byte = np.frombuffer(file.read(), np.uint8) #convert file to bytes
        img = cv2.imdecode(file_byte, cv2.IMREAD_GRAYSCALE) #grayscale conversion

        img = cv2.bitwise_not(img) #since the mnist images are black bg and white digit, and our input is inverse of that

        img = cv2.resize(img, (28, 28))

        img_array = img.astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))

        preds = model.predict(img_array)
        digit = int(np.argmax(preds, axis=1)[0])

        # return jsonify({"prediction": digit})
        return render_template("index.html", prediction_text=f"The predicted number is: {digit}")

    except:
        print("Failed")
        return render_template("index.html", prediction_text="Prediction failed.")

if __name__ == "__main__":
    # flask_app.run(debug=True) #use when local
    port = int(os.environ.get("PORT", 5000)) #this is for deployment
    flask_app.run(host="0.0.0.0", port=port) #this as well