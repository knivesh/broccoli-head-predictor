import torch
import functools
import time

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
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
    current_time = int(time.time())

    return TEMPLATES.TemplateResponse(
            "index.html", 
            {
                "request": request, 
                "image_available": LATEST_IMAGE_DATA is not None,
                "cache_buster": current_time
            }
        )

if __name__=="__main__":
    import uvicorn
    get_model()
    uvicorn.run(app, host="127.0.0.1", port=8000)
