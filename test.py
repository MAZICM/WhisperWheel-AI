import cv2
import numpy as np
import pytesseract
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Path to the test image
test_image_path = "/home/k/GITHUB/MAZICM/WhisperWheel-AI/car.jpg"

# Load the image using OpenCV
img = cv2.imread(test_image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load a pre-trained model for license plate detection
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

# Set the paths for the dataset
train_dir = "/home/k/Downloads/data/train"  # Update with the actual path
val_dir = "/home/k/Downloads/data/val"      # Update with the actual path

# Load the dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(128, 128),
    batch_size=32,
    label_mode='categorical'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=(128, 128),
    batch_size=32,
    label_mode='categorical'
)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_ds.class_names), activation='softmax')  # Adjust the number of classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_ds, validation_data=val_ds, epochs=2000)

# Evaluate the model
loss, accuracy = model.evaluate(val_ds)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Function to preprocess individual character images
def preprocess_char_img(img, img_size=(128, 128)):
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert to 3 channels
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    return img

# Split the license plate into individual characters
char_boxes = pytesseract.image_to_boxes(license_plate, config=custom_config)
characters = []

for box in char_boxes.splitlines():
    b = box.split(' ')
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    char_img = license_plate[license_plate.shape[0] - h:license_plate.shape[0] - y, x:w]
    characters.append(char_img)

# Predict each character using the CNN model
predicted_text = ""
class_names = train_ds.class_names  # List of class names from the training dataset

for char_img in characters:
    char_img = preprocess_char_img(char_img)
    predictions = model.predict(char_img)
    predicted_class = np.argmax(predictions[0])
    predicted_text += class_names[predicted_class]

print(f"Predicted license plate text: {predicted_text}")
