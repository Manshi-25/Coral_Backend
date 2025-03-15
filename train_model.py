'''import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50

# Define dataset path
DATASET_PATH = "C:\\newprograms\\CORALS_MODEL\\Model\\DATASET"
CATEGORIES = ["healthy_corals", "bleached_corals"]
LABELS = {"healthy_corals": 0, "bleached_corals": 1}

# Paths for train and test
train_path = os.path.join(DATASET_PATH, "train")

test_path = os.path.join(DATASET_PATH, "test")

# Ensure directories exist
if not os.path.exists(train_path):
    raise ValueError(f"❌ Train directory not found: {train_path}")

if not os.path.exists(test_path):
    raise ValueError(f"❌ Test directory not found: {test_path}")


# Get all images in a directory
def get_all_images(main_path):
    all_images = []
    class_folders = os.listdir(main_path)  # Get class folder names
    
    for class_folder in class_folders:
        class_folder_path = os.path.join(main_path, class_folder)  # Path to class folder
        
        if os.path.isdir(class_folder_path):  # Ensure it's a folder
            images = [os.path.join(class_folder_path, img)  # Get full image path
                      for img in os.listdir(class_folder_path) 
                      if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.extend(images)  # Add images to the list

    return all_images

# Load image paths
train_images = get_all_images(train_path)
test_images = get_all_images(test_path)

if not train_images:
    raise ValueError("❌ No images found in the train directory! Check dataset structure.")

print(f"✅ Found {len(train_images)} training images.")

#print(f"Total Testing Images: {len(test_images)}")
#print("First 5 training images:", train_images[:5]) 


# Optimized Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load images
def load_images(image_paths, directory, size=(224, 224)):
    images, labels = [], []
    for img_name in image_paths:
        img_path = os.path.join(directory, img_name)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, size)
            img = img / 255.0
            images.append(img)
            label = LABELS["healthy_corals"] if "healthy" in img_name.lower() else LABELS["bleached_corals"]
            labels.append(label)
    return np.array(images), np.array(labels)

X, y = load_images(train_images, train_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

print(f"✅ Dataset loaded: {len(X_train)} training samples, {len(X_test)} test samples.")

# Load ResNet50 Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Phase 1: Freeze all base model layers
base_model.trainable = False

x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(2, activation='softmax')(x)
resnet_model = Model(inputs=base_model.input, outputs=x)

# Compile model (initial higher learning rate)
resnet_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

#prit model summary 
resnet_model.summary()

epochs=30

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)

# Phase 1: Train top layers
history1 = resnet_model.fit(datagen.flow(X_train, y_train, batch_size=16),
                            validation_data=(X_test, y_test),
                            epochs=epochs,
                            callbacks=[reduce_lr, early_stopping])

# Phase 2: Fine-tune last 15 layers
for layer in base_model.layers[-15:]:
    layer.trainable = True

resnet_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history2 = resnet_model.fit(datagen.flow(X_train, y_train, batch_size=16),
                            validation_data=(X_test, y_test),
                            epochs=epochs,
                            callbacks=[reduce_lr, early_stopping])

# Define CNN Model
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(2, activation='softmax')
])

cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
cnn_history = cnn_model.fit(datagen.flow(X_train, y_train, batch_size=16),
                            validation_data=(X_test, y_test),
                            epochs=epochs,
                            callbacks=[reduce_lr, early_stopping])

# Plot Accuracy and Loss
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history1.history['accuracy'] + history2.history['accuracy'], label='ResNet50 Train')
plt.plot(history1.history['val_accuracy'] + history2.history['val_accuracy'], label='ResNet50 Val')
plt.plot(cnn_history.history['accuracy'], label='CNN Train')
plt.plot(cnn_history.history['val_accuracy'], label='CNN Val')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1,2,2)
plt.plot(history1.history['loss'] + history2.history['loss'], label='ResNet50 Train')
plt.plot(history1.history['val_loss'] + history2.history['val_loss'], label='ResNet50 Val')
plt.plot(cnn_history.history['loss'], label='CNN Train')
plt.plot(cnn_history.history['val_loss'], label='CNN Val')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.show()

# Save Models
resnet_model.save(os.path.join(DATASET_PATH, "RESNET50.h5"))
cnn_model.save(os.path.join(DATASET_PATH, "CNN.h5"))
print("✅ Models saved successfully!")
'''











import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# Define constants
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

# Dataset paths
train_dir = "C:\\newprograms\\All_Projects\\Corals_new\\Model\\Dataset\\Training"
val_dir = "C:\\newprograms\\All_Projects\\Corals_new\\Model\\Dataset\\Validation"
test_dir = "C:\\newprograms\\All_Projects\\Corals_new\\Model\\Dataset\\Testing"

# Load dataset
train_dataset = keras.preprocessing.image_dataset_from_directory(
    train_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
val_dataset = keras.preprocessing.image_dataset_from_directory(
    val_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
test_dataset = keras.preprocessing.image_dataset_from_directory(
    test_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)

# Get number of classes
num_classes = len(train_dataset.class_names)

# Normalize images
def process_images(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_dataset = train_dataset.map(process_images)
val_dataset = val_dataset.map(process_images)
test_dataset = test_dataset.map(process_images)

# Define CNN Model
def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train CNN Model
cnn_model = create_cnn_model()
history_cnn = cnn_model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)
cnn_model.save("cnn_model.h5")

# Define ResNet50 Model
def create_resnet_model():
    base_model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False  # Freeze base model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train ResNet Model
resnet_model = create_resnet_model()
history_resnet = resnet_model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)
resnet_model.save("resnet_model.h5")

# Evaluate Models
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(test_dataset)
print(f"CNN Test Accuracy: {cnn_test_acc:.4f}")

resnet_test_loss, resnet_test_acc = resnet_model.evaluate(test_dataset)
print(f"ResNet Test Accuracy: {resnet_test_acc:.4f}")

# Plot Training Results
def plot_results(history, model_name):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], 'r', label='Train Loss')
    plt.plot(history.history['val_loss'], 'b', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], 'r', label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], 'b', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.legend()
    
    plt.show()

plot_results(history_cnn, "CNN")
plot_results(history_resnet, "ResNet")
