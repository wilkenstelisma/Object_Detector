# Object Detection App

Streamlit app that performs object detection on an image fetched from a URL using Hugging Face’s `pipeline('object-detection')` (DETR + timm). It annotates detected objects with red bounding boxes and confidence labels and lists the detections below the image.

## Features
- Enter any image URL and run detection (press Enter in the URL box or click **Run**).
- **Clear** button resets the URL and previous results.
- Pretrained DETR model loaded once via `st.cache_resource`.
- Bounding boxes and labels drawn with PIL; results listed with confidence scores.

## Requirements
- Python 3.9+ (venv recommended).
- See `requirements.txt` (CPU-only torch wheels via `--extra-index-url https://download.pytorch.org/whl/cpu`):
  - streamlit, transformers, torch, torchvision, timm, pillow, requests

## Setup (Windows PowerShell)
```powershell
cd "c:\Users\wilke\Coding Projects\Image Detector"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
> Note: Using the venv’s python avoids PowerShell execution-policy issues with `Activate.ps1`.

## Run
```powershell
cd "c:\Users\wilke\Coding Projects\Image Detector"
.\.venv\Scripts\python.exe -m streamlit run App.py
```
Open the provided local URL (typically http://localhost:8501).

## Usage
1) Enter an image URL (a default sample is prefilled).  
2) Press Enter or click **Run** to trigger detection.  
3) View the annotated image and the detections list.  
4) Click **Clear** to reset the URL and results.

## Notes and troubleshooting
- First run downloads the model from Hugging Face; requires network and may take time.
- If fonts are missing, the app falls back to PIL’s default font.
- “No module named streamlit”: ensure you’re using the venv Python (`.\.venv\Scripts\python.exe -m streamlit run App.py`).
- If detection errors occur, they will appear in the UI; Clear resets the state so you can retry.***
