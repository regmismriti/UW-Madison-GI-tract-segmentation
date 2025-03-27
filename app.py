import tensorflow as tf
import gradio as gr
import numpy as np
import cv2
from PIL import Image
import os

# Load the trained model
# Load the trained segmentation model
MODEL_PATH = "resunet_model.h5"

model = tf.keras.models.load_model(MODEL_PATH,compile = False)

# Constants
THRESHOLD = 0.1
IMAGE_SIZE = (128, 128)
LABELS = ["Large Bowel", "Small Bowel", "Stomach"]
class2hexcolor = {"Large Bowel": "#FFFF00", "Small Bowel": "#800080", "Stomach": "#FF0000"}

# Function to preprocess images
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict segmentation masks
def predict_mask(image):
    original_size = image.size
    input_image = preprocess_image(image)
    prediction = model.predict(input_image)[0]

    masks = []
    for i in range(3):
        mask = (prediction[:, :, i] > THRESHOLD).astype(np.uint8)
        mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
        masks.append((mask, LABELS[i]))
    
    return image, masks

# Define Gradio interface
def gradio_interface(image):
    image, masks = predict_mask(image)
    return image, [(mask, label) for mask, label in masks]

with gr.Blocks(title="OncoSegAi") as gradio_app:
    gr.Markdown("<h1><center>Medical Image Segmentation</center></h1>")

    with gr.Row():
        img_input = gr.Image(type="pil", label="Input Image")
        img_output = gr.AnnotatedImage(label="Predictions", color_map=class2hexcolor)

    predict_btn = gr.Button("Generate Predictions")
    predict_btn.click(gradio_interface, inputs=img_input, outputs=img_output)

# Start Gradio server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # Use Render's port dynamically
    gradio_app.launch(server_name="0.0.0.0", server_port=port,share=True, show_api=False)
 

