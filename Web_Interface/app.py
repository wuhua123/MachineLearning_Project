from flask import Flask, render_template, request, jsonify

import os
import base64
import numpy as np
import io
from PIL import Image

from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import (
    Xception, 
    preprocess_input, 
    decode_predictions
)

app = Flask(__name__)

@app.route("/")
def home():
    print("Server received request for 'Home' page...")
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict_image():
    #file = request.files['image']
    #filename = os.path.join('selected', file.filename)

    #file.save(filename)

    # Load the VGG19 model
    # https://keras.io/applications/#VGG19
    model = Xception(include_top=True, weights='imagenet')

    # Define default image size for VGG19
    image_size = (299, 299)

    # initialize the response dictionary that will be returned
    message = request.get_json(force = True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    img  = Image.open(io.BytesIO(decoded))

    # if the image mode is not RGB, convert it
    if img.mode != "RGB":
        img = img.convert("RGB")

    # resize the input image and preprocess it
    img = img.resize(image_size)
    
    # Preprocess image for model prediction
    # This step handles scaling and normalization for VGG19
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make predictions
    predictions = model.predict(x)
    prediction = decode_predictions(predictions, top=1)
    print('Predicted:', decode_predictions(predictions, top=3))

    response = {
        'predictions':{
            'label': np.array(prediction[0][0][1]).tolist(),
            'probability': np.array(prediction[0][0][2]).tolist()     
            }
        }

    print(response)
    
    # return the response dictionary as a JSON response
    return jsonify(response)
 
if __name__ == '__main__':
    app.run(debug=True)