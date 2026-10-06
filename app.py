from flask import Flask, request, jsonify, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Load the pre-trained model
model = load_model('component.h5')

# Define the class labels and descriptions
class_labels = [
    'Bevel-gear', 'bearing', 'clutch', 'filter', 'fuel-tank',
    'helical_gear', 'piston', 'rack-pinion', 'shocker', 'spark-plug',
    'spur-gear', 'valve', 'wheel','cylinder'
]

descriptions = {
    'Bevel-gear': 'A bevel gear is a gear where the axes of the shafts are at an angle to each other, usually 90 degrees. They are used to transmit power between shafts that are not parallel.',
    'bearing': 'A bearing is a machine element that allows for rotational movement and reduces friction between moving parts. It supports and guides the rotating parts.',
    'clutch': 'A clutch is a mechanical device that engages and disengages power transmission between shafts, allowing for smooth operation of vehicles or machinery.',
    'filter': 'A filter is a device used to remove impurities or particles from fluids or gases. It ensures clean operation and protects components from damage.',
    'cylinder': 'A cylinder is an engine component in which the piston moves up and down. It forms part of the combustion chamber and helps convert combustion energy into mechanical power.',
    'fuel-tank': 'A fuel tank stores fuel for engines or machinery. It provides a reservoir for fuel to be drawn into the engine as needed.',
    'helical_gear': 'A helical gear has teeth cut at an angle to the axis of rotation. This design provides smoother and quieter operation compared to straight gears.',
    'piston': 'A piston is a component that moves up and down inside a cylinder to convert pressure into mechanical motion in an engine.',
    'rack-pinion': 'A rack and pinion is a type of gear mechanism used to convert rotational motion into linear motion. The rack is a flat, toothed component that interacts with a rotating pinion gear.',
    'shocker': 'A shock absorber, or shocker, is a component used to absorb and dampen vibrations and impacts, providing a smoother ride in vehicles or machinery.',
    'spark-plug': 'A spark plug ignites the air-fuel mixture in an engine’s cylinder by creating a spark. It is essential for the engine’s combustion process.',
    'spur-gear': 'A spur gear is a gear with straight teeth that are parallel to the axis of rotation. It is used for transmitting motion between parallel shafts.',
    'valve': 'A valve controls the flow of fluids or gases in a system. It can open, close, or regulate flow to ensure proper operation.',
    'wheel': 'A wheel is a circular component that rotates around an axle to facilitate movement. It is a fundamental part of many machines and vehicles.'
}

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def preprocess_image(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((300, 300))  # Adjust to your model's input size
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    img = Image.open(file_path)
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]

    # Set the threshold for the minimum confidence to consider the prediction valid
    threshold = 0.35

    # If the confidence is below the threshold, classify as "Not an automobile part"
    if confidence < threshold:
        class_label = "Not an automobile part"
        description = "The uploaded image does not appear to be an automobile part."
    else:
        class_label = class_labels[predicted_class_idx]
        description = descriptions[class_label]

    return render_template('result.html',
                           image_file=file.filename,
                           class_label=class_label,
                           confidence=round(float(confidence), 2),
                           description=description)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)