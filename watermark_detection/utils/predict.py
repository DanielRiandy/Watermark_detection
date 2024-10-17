import torch
import numpy as np
from PIL import Image


def predict_unseen_data(classification_model, detection_model, image_np, classification_transform, detection_transform):
    if not isinstance(image_np, np.ndarray):
        raise ValueError("Input must be a numpy array")

    # Apply transforms
    classification_input = classification_transform(image=image_np)['image'].unsqueeze(0)
    detection_input = detection_transform(image=image_np, bboxes=[], category_ids=[])['image'].unsqueeze(0)

    # Move inputs to the same device as the models
    device = next(classification_model.parameters()).device
    classification_input = classification_input.to(device)
    detection_input = detection_input.to(device)

    # Make predictions
    with torch.no_grad():
        classification_output = classification_model(classification_input)
        detection_output = detection_model(detection_input)[0]

    # Process outputs
    watermark_prob = torch.softmax(classification_output, dim=1)[0][1].item()

    return watermark_prob, detection_output