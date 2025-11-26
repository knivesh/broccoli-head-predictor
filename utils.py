import torch

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

from torchvision.ops import nms
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import functional as T_F
from torchvision.transforms import v2 as T

from PIL import Image, ImageDraw

SCORE_THRESHOLD = 0.8
NMS_THRESHOLD = 0.3

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the keypoint classifier
    in_features_keypoint = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    num_keypoints = 1

    # Keypoint predictor
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(
        in_features_keypoint,
        num_keypoints
    )

    return model

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def run_prediction(input_tensor: torch.Tensor, model: torch.nn.Module, device: torch.device):
    """
    Runs model inference, filters scores, and applies NMS.
    Returns the cleaned prediction dictionary.
    """
    with torch.no_grad():
        # 1. Add batch dimension and get predictions
        # Input tensor is C, H, W, so we add a batch dimension: 1, C, H, W
        predictions = model([input_tensor.to(device)])
        pred = predictions[0]

        # 2. Score Filtering
        keep_score = pred["scores"] >= SCORE_THRESHOLD
        for k in pred.keys():
            pred[k] = pred[k][keep_score]

        # 3. Apply NMS
        if len(pred["boxes"]) > 0:
            keep = nms(pred["boxes"], pred["scores"], NMS_THRESHOLD)
            for k in pred.keys():
                pred[k] = pred[k][keep]

        return pred

def draw_predictions_on_image(image_tensor: torch.Tensor, pred: dict) -> Image.Image:
    """
    Draws bounding boxes and keypoints on the image tensor.
    Returns the final PIL Image.
    """
    # 1. Prepare image tensor for drawing (0-255 uint8)
    image_to_draw = (255.0 * image_tensor).to(torch.uint8).cpu()

    final_scores = pred["scores"].cpu().numpy()
    pred_labels = [f"Broccoli: {score:.3f}" for score in final_scores]
    pred_boxes = pred["boxes"].long().cpu()

    # 2. Draw Bounding Boxes using torchvision
    output_tensor = draw_bounding_boxes(
        image_to_draw,
        pred_boxes,
        pred_labels,
        colors="yellow",
        width=2
    )

    # 3. Convert to PIL Image for Keypoint drawing
    output_image_pil = T_F.to_pil_image(output_tensor)

    # 4. Draw Keypoints using Pillow
    if "keypoints" in pred and len(pred["keypoints"]) > 0:
        draw = ImageDraw.Draw(output_image_pil)
        pred_keypoints = pred["keypoints"].cpu()
        radius = 3

        for kp in pred_keypoints:
            # Keypoints are typically N x 3 (x, y, visibility). We use x, y.
            x_coord, y_coord = kp[0, 0].item(), kp[0, 1].item()

            draw.ellipse(
                (x_coord - radius, y_coord - radius, 
                 x_coord + radius, y_coord + radius),
                fill=(255, 0, 0) # Red dot
            )

    return output_image_pil
