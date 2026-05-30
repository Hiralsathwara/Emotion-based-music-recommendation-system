
import pytest
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, recall_score
import cv2
import os

# Load the trained model
model = load_model('model.h5')

# Helper function to preprocess input data for the model
def preprocess_image(image_path):
    # Read image using OpenCV
    img = cv2.imread(image_path)
    # Convert to grayscale if necessary
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize to the expected input size of the model
    img = cv2.resize(img, (48, 48))
    # Normalize pixel values
    img = img / 255.0
    # Expand dimensions to match model input
    img = np.expand_dims(img, axis=[0, -1])
    return img

# Test dataset path
test_data_dir = 'data/test_images'

def test_model_accuracy():
    # Load test data
    image_files = [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir)]
    true_labels = [...]  # Fill in with correct labels for the test dataset
    
    predictions = []
    for image_path in image_files:
        # Preprocess each image
        img = preprocess_image(image_path)
        # Predict emotion using the model
        pred = model.predict(img)
        # Append predicted class
        predictions.append(np.argmax(pred))

    # Calculate accuracy and recall
    accuracy = accuracy_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions, average='macro')

    # Assert model meets expected performance thresholds
    assert accuracy >= 0.7, f"Model accuracy is below expected threshold: {accuracy}"
    assert recall >= 0.7, f"Model recall is below expected threshold: {recall}"

if __name__ == "__main__":
    pytest.main()
