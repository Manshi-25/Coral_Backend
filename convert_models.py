import tensorflow as tf

def convert_model(model_path, output_path):
    """Converts a Keras model (.h5) to TensorFlow Lite (.tflite) format."""
    model = tf.keras.models.load_model(model_path)
    
    # Enable optimization to reduce model size
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Reduce size
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Converted {model_path} to {output_path}")

# Convert CNN and ResNet models
convert_model("C:\\newprograms\\All_Projects\\Corals_new\\Model\\cnn_model.h5", "cnn_model.tflite")
convert_model("C:\\newprograms\\All_Projects\\Corals_new\\Model\\resnet_model.h5", "resnet_model.tflite")
