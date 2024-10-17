import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.model import WatermarkClassifier, get_object_detection_model
from utils.transforms import get_classification_transform, get_detection_transform
from utils.utils import load_model, visualize_results
from utils.predict import predict_unseen_data

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    classification_model_path = '/Users/driandy/Desktop/Works/Personal/99 group/watermark_detection/models/classification_models/best_classification_model.pth'
    detection_model_path = '/Users/driandy/Desktop/Works/Personal/99 group/watermark_detection/models/object_detection_models/final_model.pth'

    classification_model = load_model(WatermarkClassifier(), classification_model_path, device)
    detection_model = load_model(get_object_detection_model(num_classes=2), detection_model_path, device)

    # Get transforms
    classification_transform = get_classification_transform()
    detection_transform = get_detection_transform()

    # Predict on unseen data
    unseen_image_path = '/Users/driandy/Desktop/Works/Personal/99 group/watermark_detection/data/all_data/IqyfMrqcScPxb9bS.jpg'
    original_image, watermark_prob, detection_prediction = predict_unseen_data(
        classification_model, detection_model, unseen_image_path,
        classification_transform, detection_transform
    )

    # Print results
    print(f"Image Classification Result:")
    print(f"  Watermark Probability: {watermark_prob:.4f}")
    print(f"  Prediction: {'R123 Watermark' if watermark_prob > 0.5 else 'No R123 Watermark'}")

    print("\nObject Detection Results:")
    r123_detections = [(score.item(), label.item()) for score, label in zip(detection_prediction['scores'], detection_prediction['labels']) if score > 0.5]
    print(f"  Detected Objects: {len(r123_detections)} (scores > 0.5)")
    print("  Top 5 detections:")
    for i, (score, label) in enumerate(sorted(r123_detections, reverse=True)[:5], 1):
        label_name = "r123" if label == 1 else "non_r123"
        print(f"    {i}. {label_name}: {score:.4f}")

    # Visualize results
    visualize_results(original_image, watermark_prob, detection_prediction)

if __name__ == "__main__":
    main()