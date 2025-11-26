import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

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