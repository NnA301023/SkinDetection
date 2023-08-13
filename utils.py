import streamlit as st 
from tensorflow.keras.models import load_model


def set_img_as_background(img_path):
    """
    Display an image as the background for the Streamlit app.

    Parameters:
        img_path (str): Path to the image file to be used as the background.

    Returns:
        None
    """
    st.image(img_path)

@st.cache(allow_output_mutation = True)
def load_trained_model():
    """
    Load a trained Keras model from the specified path.

    Parameters:
        model_path (str): Path to the trained model file (.h5).

    Returns:
        tensorflow.keras.models.Model: The loaded Keras model.
    """
    model = load_model("model/KlasifikasiWajah-pest-65.23.h5")
    model.summary()
    return model 

def predict_model(image, model):
    """
    Make predictions using a trained model.

    Parameters:
        image (numpy.ndarray): Input image as a NumPy array.
        model (tensorflow.keras.models.Model): Trained Keras model.

    Returns:
        numpy.ndarray: Predicted probabilities for each class.
    """
    y_probs = model.predict(image, verbose = 0)
    return y_probs