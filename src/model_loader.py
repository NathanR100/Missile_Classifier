
import os
import tensorflow as tf

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# # To delete the saved model file
# model_path = 'my_model.keras'
# if os.path.exists(model_path):
#     os.remove(model_path)
#     print(f"Deleted saved model: {model_path}")
# else:
#     print(f"Model file not found: {model_path}")
