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
if "image_url" not in st.session_state:
    st.session_state.image_url = "https://tse1.mm.bing.net/th/id/OIP.zxKIH61D8swSwuelyI6oNQHaD4?rs=1&pid=ImgDetMain&o=7&rm=3"
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_image" not in st.session_state:
    st.session_state.last_image = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None
if "clear_trigger" not in st.session_state:
    st.session_state.clear_trigger = False
if "run_trigger" not in st.session_state:
    st.session_state.run_trigger = False

def _trigger_run():
    # When the user presses Enter in the URL box, request a run on the next render.
    st.session_state.run_trigger = True
    st.session_state.last_error = None
    st.rerun()

# Apply pending clear before rendering widgets to avoid state mutation errors
if st.session_state.clear_trigger:
    st.session_state.image_url = ""
    st.session_state.last_results = None
    st.session_state.last_image = None
    st.session_state.last_error = None
    st.session_state.run_trigger = False
    st.session_state.clear_trigger = False

st.text_input("Enter Image URL", key="image_url", on_change=_trigger_run)
col_run, col_clear = st.columns(2)
run_clicked = col_run.button("Run", type="primary")
clear_clicked = col_clear.button("Clear")

image_url = st.session_state.image_url.strip()

if clear_clicked:
    st.session_state.clear_trigger = True
    st.session_state.run_trigger = False
    st.rerun()

should_run = run_clicked or st.session_state.run_trigger

if should_run:
    # Reset run trigger so it only fires once per Enter press
    st.session_state.run_trigger = False
    if not image_url:
        st.session_state.last_results = None
        st.session_state.last_image = None
        st.session_state.last_error = "Please enter an image URL before running."
    else:
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

            st.session_state.last_results = results
            st.session_state.last_image = img_with_boxes
            st.session_state.last_error = None

        except Exception as e:
            st.session_state.last_results = None
            st.session_state.last_image = None
            st.session_state.last_error = f"Error processing image: {e}"

if st.session_state.last_error:
    st.error(st.session_state.last_error)
elif st.session_state.last_image is not None and st.session_state.last_results is not None:
    st.image(st.session_state.last_image, caption='Detected Objects', use_container_width=True)
    st.subheader('Detected objects:')
    for result in st.session_state.last_results:
        st.write(f"- {result['label']} with confidence {result['score']:.2f}")
else:
    st.info("Enter an image URL, then click Run. Use Clear to reset the URL.")
