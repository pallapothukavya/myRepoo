import os
import tensorflow as tf
try:
    print("TensorFlow version:", tf.__version__)
    from tensorflow.keras.models import load_model
    print("Successfully imported load_model from tensorflow.keras.models")
    
    model_path = 'multiclass.h5'
    if os.path.exists(model_path):
        print(f"Loading model from {model_path} with tensorflow.keras...")
        model = load_model(model_path)
        print("Successfully loaded model with tensorflow.keras.")
    else:
        print(f"Model file {model_path} not found.")
except Exception as e:
    print("Error with tensorflow.keras:", e)
