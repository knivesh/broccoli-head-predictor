import torch
import functools
import time
import io
from PIL import Image
import numpy as np

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from utils import get_model_instance_segmentation, get_transform, run_prediction, draw_predictions_on_image

NUM_CLASSES = 2
CHECKPOINT_PATH=r"models\experiment1_epoch_4.pth"
TEMPLATES = Jinja2Templates(directory="templates")
LATEST_IMAGE_DATA = None

def get_device():
    """Dynamically determines the best device (CUDA/CPU) for the model."""
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator() 
    else:
        device = torch.device('cpu')
    device = torch.device('cpu')
    return device

@functools.lru_cache(maxsize=1)
def get_model():
    """Loads the model once and caches it for all subsequent requests."""
    device = get_device()

    model = get_model_instance_segmentation(NUM_CLASSES)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    model.eval()
    return model

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

app = FastAPI(title="Broccoli Inference API")

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

    image_bytes = await file.read()

    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')

    output_buffer = io.BytesIO()
    img.save(output_buffer, format="JPEG")

    LATEST_IMAGE_DATA = output_buffer.getvalue()

    return HTMLResponse(content="""<script>window.location.href = '/';</script>""")

@app.get("/image/")
async def serve_image():
    """Serves the latest image data (uploaded or predicted) as a JPEG stream."""
    global LATEST_IMAGE_DATA

    return StreamingResponse(io.BytesIO(LATEST_IMAGE_DATA), media_type="image/jpeg")

@app.post("/predict/")
async def predict_image():
    """Runs inference on the latest uploaded image."""
    global LATEST_IMAGE_DATA

    input_image = Image.open(io.BytesIO(LATEST_IMAGE_DATA)).convert("RGB")
    output_image = predict_boxes_and_keypoints(input_image)

    output_byte_array = io.BytesIO()
    output_image.save(output_byte_array, format="JPEG")

    LATEST_IMAGE_DATA = output_byte_array.getvalue()

    return HTMLResponse(content="""<script>window.location.href = '/';</script>""")

if __name__=="__main__":
    import uvicorn
    get_model()
    uvicorn.run(app, host="127.0.0.1", port=8000)
