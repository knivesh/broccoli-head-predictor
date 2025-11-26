import torch
import functools
import time
import io
from PIL import Image

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from utils import get_model_instance_segmentation

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

if __name__=="__main__":
    import uvicorn
    get_model()
    uvicorn.run(app, host="127.0.0.1", port=8000)
