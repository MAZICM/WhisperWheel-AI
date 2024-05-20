
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Check if the data directories exist
path = 'C:\\Users\\mouhi\\GITHUB\\MazicM\\WhisperWheel-AI\\data\\data\\'
print("Training data directory exists:", os.path.exists(os.path.join(path, 'train')))
print("Validation data directory exists:", os.path.exists(os.path.join(path, 'val')))

# Load and preprocess a sample image from the training set
sample_image_path = os.path.join(path, 'car.jpg')  # Replace 'class_name' and 'sample_image.jpg' with actual values
sample_image = cv2.imread(sample_image_path)
sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)  # Convert image to RGB format
sample_image = cv2.resize(sample_image, (28, 28)) / 255.0  # Resize and rescale pixel values
print("Sample image shape:", sample_image.shape)

# Display the sample image
cv2.imshow('Sample Image', sample_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Verify the data generators
train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1)
train_generator = train_datagen.flow_from_directory(
    os.path.join(path, 'train'),
    target_size=(28, 28),
    batch_size=1,
    class_mode='sparse'
)

validation_generator = train_datagen.flow_from_directory(
    os.path.join(path, 'val'),
    target_size=(28, 28),
    batch_size=1,
    class_mode='sparse'
)

# Define model architecture
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(36, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)