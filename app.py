from flask import Flask, jsonify, request
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from PIL import Image
import io

# Define the class names
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

# Load the model
model_path = './model4.h5'
model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

def extract_and_predict(image, model, class_names):
    # Convert the image to RGB format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    h, w, c = img.shape
    if w > 1000:
        new_w = 1000
        ar = w / h
        new_h = int(new_w / ar)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Apply thresholding
    thresh_img = threasholding(img)

    # Apply dilation to merge text lines
    kernel = np.ones((8, 110), np.uint16)
    dilated = cv2.dilate(thresh_img, kernel, iterations=1)

    # Find contours
    (contours, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    predictions = []
    for ctr in sorted_contours_lines:
        x, y, w, h = cv2.boundingRect(ctr)
        if w > 10 and h > 10:
            # Crop the region from the image
            cropped_img = img[y:y+h, x:x+w]
            
            # Convert to PIL Image and resize if needed
            pil_img = Image.fromarray(cropped_img)
            pil_img = pil_img.resize((600, 120))  # Resize to model's input size
            img_array = np.array(pil_img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]
            predictions.append((predicted_class))

            # Optionally draw rectangle and label on the image
            cv2.rectangle(img, (x, y), (x+w, y+h), (40, 100, 250), 2)
            cv2.putText(img, predicted_class, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return predictions

@app.route("/", methods=["GET"])
def greet_json():
    return jsonify({"message": "Image prediction service is running."})

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        predictions = extract_and_predict(image, model, class_names)
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def threasholding(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)
    return thresh


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860)
