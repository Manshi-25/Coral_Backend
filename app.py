import os
from flask import Flask, request, jsonify
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from PIL import Image
import numpy as np
import logging
from flask_cors import CORS
from werkzeug.utils import secure_filename

import io
from flask_cors import CORS
from waitress import serve

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

base_dir = os.path.dirname(os.path.abspath(__file__))
cnn_model_path = os.path.join(base_dir, 'Model', 'cnn_model.h5')
resnet_model_path = os.path.join(base_dir, 'Model', 'resnet_model.h5')


# Load your CNN and ResNet50 models
#cnn_model = tf.keras.models.load_model(r'Model\\cnn_model.h5')
cnn_model = tf.keras.models.load_model(cnn_model_path)
resnet_model = tf.keras.models.load_model(resnet_model_path)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def preprocess_image(image):
    # Resize and preprocess the image for the model
    image = image.resize((128, 128))  # Adjust size as needed
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        logging.error("No file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Get the selected model
    model_type = request.form.get("model")
    
    print("Received Form Data:", request.form)
    print("Model Selected:", request.form.get("model"))
    print(f"Model received: '{model_type}' (Type: {type(model_type)})")
     
     
    logging.debug(f"Received request with model: {model_type}")
    
    # Check if the file is an image
    if file.filename == '':
        logging.error("No file selected")
        return jsonify({'error': 'No file selected or invalid type'}), 400
    
    if model_type not in ["resnet", "cnn"]:
        return jsonify({"error": f"Invalid model choice: {model_type}. Choose 'resnet' or 'cnn'"}), 400

    try:
        # Open and preprocess the image
        image = Image.open(file.stream)
        image.verify()  # Verify that the file is a valid image
        image = Image.open(file.stream)  # Reopen the file for processing
        image = preprocess_image(image)
       
        ''' filename = secure_filename(file.filename)
        filepath = os.path.join("uploads", filename)
        file.save(filepath)
    
        image = preprocess_image(image)'''
        
        # Make prediction based on the selected model
        
        '''if model_type == 'cnn':
            prediction = cnn_model.predict(image)
        elif model_type == 'resnet':
            prediction = resnet_model.predict(image)
        else:
            logging.error("Invalid model selected")
            return jsonify({'error': 'Invalid model selected'}), 400
        '''
        
        model = resnet_model if model_type == "resnet" else cnn_model
        prediction = model.predict(image)
        
        # Process prediction result (example logic)
        result = 'Bleached' if prediction[0][0] > 0.6 else 'Healthy'
        
        logging.debug(f"Prediction Result: {result}")
        return jsonify({'result': result})
    
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    serve(app, host='0.0.0.0', port=port)
    #from waitress import serve  # Use production server
    #serve(app, host="0.0.0.0", port=8000)
    #app.run(debug=True)
    #app.run(debug=True, host ="0.0.0.0", port=5000)
    

    
    

'''from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import logging
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load updated models (update paths based on train_model.py)
cnn_model = tf.keras.models.load_model(r'C:\\newprograms\\All_Projects\\Corals_new\\cnn_model.h5')
resnet_model = tf.keras.models.load_model(r'C:\\newprograms\\All_Projects\\Corals_new\\resnet_model.h5')

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def preprocess_image(image):
    """
    Preprocess the image based on updated logic from train_model.py.
    """
    # Resize and preprocess the image for the model
    image = image.resize((224, 224))  # Update size if train_model.py uses a different input size
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests.
    """
    if 'file' not in request.files:
        logging.error("No file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    model_type = request.form.get("model")

    logging.debug(f"Received request with model: {model_type}")

    if file.filename == '':
        logging.error("No file selected")
        return jsonify({'error': 'No file selected or invalid type'}), 400

    if model_type not in ["resnet", "cnn"]:
        return jsonify({"error": f"Invalid model choice: {model_type}. Choose 'resnet' or 'cnn'"}), 400

    try:
        # Open and preprocess the image
        image = Image.open(file.stream)
        image = preprocess_image(image)

        # Select the model based on the request
        model = resnet_model if model_type == "resnet" else cnn_model
        prediction = model.predict(image)

        # Process prediction result (update logic if train_model.py changes output format)
        result = 'Healthy' if prediction[0][0] > 0.7 else 'Bleached'

        logging.debug(f"Prediction Result: {result}")
        return jsonify({'result': result})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(debug=True, host="0.0.0.0", port=5000)
    '''