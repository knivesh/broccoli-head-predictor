import torch
import functools
import time
import io
import os
from PIL import Image
import numpy as np

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from utils import get_model_instance_segmentation, get_transform, run_prediction, draw_predictions_on_image

NUM_CLASSES = 2
CHECKPOINT_PATH=r"models\experiment1_epoch_4.pth"
TEMPLATES = Jinja2Templates(directory="templates")
LATEST_IMAGE_DATA = None

def get_device():
    """Using CPU only for inference at the moment"""
    device = torch.device('cpu')
    print("Using CPU for all tensor operations.")
    return device

@functools.lru_cache(maxsize=1)
def get_model():
    """Loads the model once and caches it for all subsequent requests."""
    # Edge Case: Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        raise RuntimeError(f"Model checkpoint not found at: {CHECKPOINT_PATH}")

    try:
        device = get_device()
        model = get_model_instance_segmentation(NUM_CLASSES)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"Broccoli Model successfully loaded on {device}.")
        return model
    except Exception as e:
        print(f"FATAL: Error loading PyTorch model: {e}")
        raise RuntimeError("Failed to initialize model. Check checkpoint and utils.py architecture.") from e

# Instantiate the callable transform object ONCE at startup
EVAL_TRANSFORM = get_transform(train=False)

def eval_transform(image_pil: Image.Image) -> torch.Tensor:
    """
    Converts PIL Image to the raw torch.uint8 tensor (C, H, W) format 
    that the T.Compose pipeline expects after a read_image equivalent.
    """
    # 1. Convert PIL Image to NumPy array (H, W, C)
    image_np = np.array(image_pil)

    # 2. Convert NumPy array to PyTorch Tensor (H, W, C)
    image_tensor_uint8 = torch.from_numpy(image_np)

    # 3. Permute to (C, H, W) and ensure type is uint8 (like read_image)
    if image_tensor_uint8.ndim == 3:
        image_tensor_uint8 = image_tensor_uint8.permute(2, 0, 1).contiguous()

    # 4. Apply your exact T.Compose logic (EVAL_TRANSFORM)
    return EVAL_TRANSFORM(image_tensor_uint8)

def predict_boxes_and_keypoints(image_pil: Image.Image):
    """
    Main function to orchestrate pre-processing, prediction, and drawing.
    """
    device = get_device()
    model = get_model() # Get the cached model

    # 1. Pre-processing
    input_tensor = eval_transform(image_pil)
    input_tensor = input_tensor[:3, ...] # Ensure RGB

    # 2. Pure Prediction
    pred = run_prediction(input_tensor, model, device)

    # 3. Drawing and Visualization
    output_image_pil = draw_predictions_on_image(input_tensor, pred)

    return output_image_pil

# FASTAPI APP SETUP
app = FastAPI(title="Broccoli Inference API")

# Attempt to load model at startup for quick failure detection
try:
    get_model()
except RuntimeError:
    pass

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """Renders the simple HTML frontend from the template."""
    current_time = int(time.time())

    return TEMPLATES.TemplateResponse(
            "index.html", 
            {
                "request": request, 
                "image_available": LATEST_IMAGE_DATA is not None,
                "cache_buster": current_time
            }
        )

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    """Handles image upload, converts to JPEG, and stores its bytes."""
    global LATEST_IMAGE_DATA

    # Edge Case: Check for file type/MIME type (Restricted to PNG)
    if file.content_type != "image/png":
        raise HTTPException(status_code=400, detail=f"Only PNG image files are allowed. Received: {file.content_type}")

    try:
        image_bytes = await file.read()

        # Edge Case: Check for empty file
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        output_buffer = io.BytesIO()
        img.save(output_buffer, format="JPEG")

        LATEST_IMAGE_DATA = output_buffer.getvalue()

    except Exception as e:
        print(f"Error processing image upload: {e}")
        raise HTTPException(status_code=500, detail="Could not process the uploaded PNG image. Check file integrity.")

    return HTMLResponse(content="""<script>window.location.href = '/';</script>""")

@app.get("/image/")
async def serve_image():
    """Serves the latest image data (uploaded or predicted) as a JPEG stream."""
    global LATEST_IMAGE_DATA

    # Edge Case: No image data uploaded yet
    if LATEST_IMAGE_DATA is None:
        raise HTTPException(status_code=404, detail="No image available to display. Please upload an image first.")

    return StreamingResponse(io.BytesIO(LATEST_IMAGE_DATA), media_type="image/jpeg")

@app.post("/predict/")
async def predict_image():
    """Runs inference on the latest uploaded image."""
    global LATEST_IMAGE_DATA

    # Edge Case: No image data to predict on
    if LATEST_IMAGE_DATA is None:
        raise HTTPException(status_code=404, detail="Cannot run prediction: No image has been uploaded.")

    try:
        input_image = Image.open(io.BytesIO(LATEST_IMAGE_DATA)).convert("RGB")
        output_image = predict_boxes_and_keypoints(input_image)

        output_byte_array = io.BytesIO()
        output_image.save(output_byte_array, format="JPEG")

        LATEST_IMAGE_DATA = output_byte_array.getvalue()

    except Exception as e:
        print(f"Prediction execution failed: {e}")
        # Re-raise as 500 internal server error
        raise HTTPException(status_code=500, detail=f"Prediction failed due to an internal error: {e}")

    return HTMLResponse(content="""<script>window.location.href = '/';</script>""")

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
