import tensorflow as tf
from src.model_loader import load_model_tf 
from src.preprocess import preprocess_track 

def run_inference_tf(track_df, model_path, scaler_path):
    # preprocess_track should output a NumPy array or similar
    processed_data = preprocess_track(track_df, scaler_path)

    # Convert processed data to a TensorFlow tensor
    input_tensor = tf.constant(processed_data, dtype=tf.float32)

    # Load the TensorFlow model
    # Assuming load_model_tf is a function that loads your TF model
    model = load_model_tf(model_path)

    # Run inference
    output = model(input_tensor, training=False) # Set training=False for inference

    # Apply sigmoid activation
    prediction = tf.sigmoid(output).numpy() # Convert to NumPy if needed

    return prediction
