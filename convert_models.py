'''import tensorflow as tf

def convert_model(model_path, output_path, model_identifier='TFL3'):
    """Converts a Keras model (.h5) to TensorFlow Lite (.tflite) format."""
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Enable optimization to reduce model size
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Reduce size
    tflite_model = converter.convert()

    # Save the converted model
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Converted {model_path} to {output_path} with identifier '{model_identifier}'")

# Convert CNN and ResNet models with the correct identifier
convert_model("C:\\newprograms\\All_Projects\\Corals_new\\Model\\cnn_model.h5", "cnn_model.tflite", model_identifier='TFL3')
convert_model("C:\\newprograms\\All_Projects\\Corals_new\\Model\\resnet_model.h5", "resnet_model.tflite", model_identifier='TFL3')

'''


import tensorflow as tf

# Load your trained models
cnn_model = tf.keras.models.load_model("cnn_model.h5")
resnet_model = tf.keras.models.load_model("resnet_model.h5")

# Convert CNN Model
cnn_converter = tf.lite.TFLiteConverter.from_keras_model(cnn_model)
cnn_converter.experimental_new_converter = True  # Ensure TFL3 compatibility
cnn_tflite_model = cnn_converter.convert()
with open("cnn_model.tflite", "wb") as f:
    f.write(cnn_tflite_model)

# Convert ResNet Model
resnet_converter = tf.lite.TFLiteConverter.from_keras_model(resnet_model)
resnet_converter.experimental_new_converter = True
resnet_tflite_model = resnet_converter.convert()
with open("resnet_model.tflite", "wb") as f:
    f.write(resnet_tflite_model)

print("Models successfully converted to TFLite format.")
