import cv2
import numpy as np
import pytesseract
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Path to the test image
test_image_path = "/home/k/GITHUB/MAZICM/WhisperWheel-AI/car.jpg"

# Load the image using OpenCV
img = cv2.imread(test_image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load a pre-trained model for license plate detection
# Assuming you have a function `detect_license_plate` that returns bounding box coordinates
# For demonstration, we use Haar Cascade Classifier for license plate detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
plates = plate_cascade.detectMultiScale(gray, 1.1, 10)

# Assume the first detected plate is the target license plate
for (x, y, w, h) in plates:
    license_plate = gray[y:y+h, x:x+w]
    break

# Preprocess the license plate for better OCR results
license_plate = cv2.resize(license_plate, (0, 0), fx=2, fy=2)
_, license_plate = cv2.threshold(license_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Recognize characters using Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"  # Update the path to tesseract executable if necessary
custom_config = r'--oem 3 --psm 8'  # Set OCR options
text = pytesseract.image_to_string(license_plate, config=custom_config)

# Print the recognized text
print(f"Detected license plate text: {text}")

# Display the detected license plate and the full image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(license_plate, cmap='gray')
plt.title("Detected License Plate")
plt.axis('off')

plt.show()
