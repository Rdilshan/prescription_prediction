from flask import Flask, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras.preprocessing import image
import logging
from PIL import Image

# List of class names
class_names = [
    'Amoxicillin', 'Ampicillin', 'Azithromycin', 'Azlocillin', 'Aztreonam',
    'Benzylpenicillin', 'Carbenicillin', 'Cefixime', 'Ceftaroline', 'Ceftazidime',
    'Ceftobiprole', 'Ceftolozane', 'Ceftriaxone', 'Cefuroxime', 'Cinoxacin',
    'Ciprofloxacin', 'Clavulanic acid', 'Clindamycin', 'Cloxacillin', 'Co-amoxiclav',
    'Colistin', 'Dalbavancin', 'Daptomycin', 'Delafloxacin', 'Doripenem',
    'Doxycycline', 'Enoxacin', 'Ertapenem', 'Erythromycin', 'Faropenem',
    'Fleroxacin', 'Fosfomycin', 'Gatifloxacin', 'Gentamicin', 'Hetacillin',
    'Imipenem', 'Levofloxacin', 'Linezolid', 'Meropenem', 'Meticillin',
    'Metronidazole', 'Mezlocillin', 'Moxifloxacin', 'Nafcillin', 'Netilmicin',
    'Nitrofurantoin', 'Norfloxacin', 'Ofloxacin', 'Pazufloxacin', 'Pefloxacin',
    'Penicillin V', 'Piperacillin', 'Plazomicin', 'Ribostamycin', 'Rifampicin',
    'Rufloxacin', 'Sparfloxacin', 'Spiramycin', 'Sulbactam', 'Sulfadoxine',
    'Sulfamethoxazole', 'Tazobactam', 'Tedizolid', 'Teicoplanin', 'Tetracycline',
    'Ticarcillin', 'Tigecycline', 'Trimethoprim', 'Trovafloxacin', 'Vancomycin'
]

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route("/")
def greet_json():
    return jsonify({"message": "Hello, World!"})

@app.route("/predict")
def predict():
    try:
        # Load the model
        loaded_model = tf.keras.models.load_model('./model4.h5', custom_objects={'KerasLayer': hub.KerasLayer})

        # Path to the image (make sure the path is correct)
        img_path = './9xk5eki7jc.jpg'

        # Load and preprocess the image
        img = Image.open(img_path)
        img = img.resize((600, 120))  # Correctly resize to (600, 120)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        # Make a prediction
        prediction = loaded_model.predict(img)

        # Get the predicted class
        predicted_class = class_names[np.argmax(prediction)]

        return jsonify({"predicted_class": predicted_class})
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860)
