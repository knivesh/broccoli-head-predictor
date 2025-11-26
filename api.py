import torch

from utils import get_model_instance_segmentation

NUM_CLASSES = 2
CHECKPOINT_PATH=r"models\experiment1_epoch_4.pth"

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

if __name__=="__main__":
    model = get_model()
    print(f"Model state: {model.training=}")
    print(f"Model device: {next(model.parameters()).device}")