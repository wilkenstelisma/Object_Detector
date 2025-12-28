import streamlit as st
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import requests

# Load the pre-trained object detection pipeline (cached to avoid reloading)
@st.cache_resource
def load_object_detector():
    return pipeline('object-detection')

object_detector = load_object_detector()

st.title('Object Detection App')
image_url = st.text_input('Enter Image URL', 'https://tse1.mm.bing.net/th/id/OIP.zxKIH61D8swSwuelyI6oNQHaD4?rs=1&pid=ImgDetMain&o=7&rm=3')

if image_url:
    try:
        # Load the image from the URL
        img = Image.open(requests.get(image_url, stream=True).raw)

        # Perform object detection
        results = object_detector(img)

        # Draw the bounding boxes on a copy of the image
        img_with_boxes = img.copy()
        draw = ImageDraw.Draw(img_with_boxes)

        # Define a font if available, otherwise use default
        try:
            font = ImageFont.truetype("arial.ttf", 15) # Example font, might not be available in Colab
        except IOError:
            font = ImageFont.load_default()

        for result in results:
            box = result['box']
            label = result['label']
            score = result['score']

            # Draw the box
            draw.rectangle([(box['xmin'], box['ymin']), (box['xmax'], box['ymax'])], outline='red', width=3)
            # Draw the label
            draw.text((box['xmin'], box['ymin'] - 18), f"{label} ({score:.2f})", fill='red', font=font)

        st.image(img_with_boxes, caption='Detected Objects', use_container_width=True)

        st.subheader('Detected objects:')
        for result in results:
            st.write(f"- {result['label']} with confidence {result['score']:.2f}")

    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.write("Please enter an image URL to perform object detection.")
