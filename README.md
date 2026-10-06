# Automobile Parts Classification

An AI-powered web application that identifies automobile parts from uploaded images using a pre-trained TensorFlow/Keras deep learning model.

The application is built using Python, Flask, TensorFlow, NumPy, Pillow, HTML, and CSS. Users can upload an automobile-part image, and the system predicts the corresponding part along with its confidence score and description.

## Project Overview

Identifying automobile components manually can be difficult, especially for beginners and people who are not familiar with vehicle parts.

This project provides an image classification system that automatically recognizes automobile parts from images.

The application uses a pre-trained deep learning model (`component.h5`) to classify images into 14 automobile-part categories.

## Features

- Upload an automobile-part image
- AI-based image classification
- Displays prediction confidence
- Provides a description of the predicted part
- Flask-based web application
- Real-time prediction
- Simple and user-friendly interface
- Uses a pre-trained TensorFlow/Keras model
- Model managed using Git LFS

## Automobile Parts Supported

The model supports the following categories:

1. Bevel Gear
2. Bearing
3. Clutch
4. Filter
5. Fuel Tank
6. Helical Gear
7. Piston
8. Rack and Pinion
9. Shocker
10. Spark Plug
11. Spur Gear
12. Valve
13. Wheel
14. Cylinder

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Flask | Web application framework |
| TensorFlow | Deep learning model |
| Keras | Model loading and inference |
| NumPy | Numerical operations |
| Pillow | Image processing |
| HTML | Web page structure |
| CSS | Web page styling |
| Git | Version control |
| Git LFS | Large model file management |

## Project Structure

```text
AutomobilePartsClassification/
│
├── app.py
├── component.h5
├── requirements.txt
├── .gitignore
├── .gitattributes
│
├── dataset/
│   └── Automobile-parts/
│       ├── bearing/
│       ├── Bevel-gear/
│       ├── clutch/
│       ├── filter/
│       ├── fuel-tank/
│       ├── helical_gear/
│       ├── piston/
│       ├── rack-pinion/
│       ├── shocker/
│       ├── spark-plug/
│       ├── spur-gear/
│       ├── valve/
│       ├── wheel/
│       └── cylinder/
│
└── templates/
    ├── index.html
    └── result.html
How the System Works
User uploads image
        |
        v
Flask receives image
        |
        v
Image preprocessing
        |
        v
Resize image to 300 x 300
        |
        v
Normalize pixel values
        |
        v
TensorFlow/Keras model
        |
        v
Prediction
        |
        v
Calculate confidence
        |
        v
Display predicted automobile part
        |
        v
Display part description
Model Information

The project uses a pre-trained TensorFlow/Keras model:

component.h5
Model Input
300 x 300 x 3

The uploaded image is resized to 300 x 300 pixels and converted to RGB format before prediction.

Pixel values are normalized between 0 and 1.

Model Output

The model produces probabilities for 14 classes using a softmax output layer.

The class with the highest probability is selected as the predicted automobile part.

Installation
1. Clone the Repository
git clone https://github.com/meghanavaddi2402/AutomobilePartsClassification.git

Move into the project directory:

cd AutomobilePartsClassification
2. Create a Virtual Environment

For Windows:

python -m venv .venv

Activate the virtual environment:

.venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
4. Run the Application
python app.py

The Flask server will start at:

http://127.0.0.1:5000

Open the URL in your web browser.

Using the Application
Open the Flask application in your browser.
Select an automobile-part image.
Upload the image.
Click the prediction button.
The application displays:
Predicted automobile part
Confidence score
Description of the part
Example

For an uploaded bearing image, the application may produce:

Predicted Part: bearing

Confidence: 0.95

Description:
A bearing is a machine element that allows for rotational
movement and reduces friction between moving parts.

The actual confidence depends on the uploaded image and model prediction.

Dataset

The project contains images belonging to different automobile-part categories.

The dataset is organized into separate folders according to the automobile-part class.

Example:

dataset/
└── Automobile-parts/
    ├── bearing/
    ├── clutch/
    ├── piston/
    ├── spark-plug/
    └── ...
Large Model File

The trained model:

component.h5

is a large file and is managed using Git Large File Storage (Git LFS).

To check whether Git LFS is installed:

git lfs --version

To download the LFS files after cloning the repository:

git lfs pull
Future Enhancements
Improve model classification accuracy
Add more automobile-part categories
Add camera-based image detection
Improve image preprocessing
Add prediction history
Add model performance metrics
Deploy the application to a cloud platform
Improve the user interface
Add user authentication
Learning Outcomes

This project provided practical experience with:

Python programming
Flask web development
Image preprocessing
TensorFlow and Keras
Deep learning image classification
File upload handling
Git and GitHub
Git Large File Storage
HTML and CSS integration with Flask
