import torch

def get_device():
    """Dynamically determines the best device (CUDA/CPU) for the model."""
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator() 
    else:
        device = torch.device('cpu')
    return device

if __name__=="__main__":
    print(get_device())
