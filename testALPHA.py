import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

# Set the paths
train_dir = "/home/k/Downloads/data/train"  # Update with the actual path
val_dir = "/home/k/Downloads/data/val"      # Update with the actual path
test_image_path = "//home/k/GITHUB/MAZICM/WhisperWheel-AI/car.jpg"  # Update with the actual path to the car image

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
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

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

# Load and preprocess the test image
def load_and_preprocess_image(img_path, img_size=(128, 128)):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Load the test image
test_img = load_and_preprocess_image(test_image_path)

# Make a prediction
predictions = model.predict(test_img)
predicted_class = np.argmax(predictions[0])

# Print the predicted class
class_names = train_ds.class_names  # List of class names from the training dataset
print(f"Predicted class: {class_names[predicted_class]}")

# Display the image with the prediction
plt.imshow(image.load_img(test_image_path))
plt.title(f"Predicted class: {class_names[predicted_class]}")
plt.axis('off')
plt.show()
