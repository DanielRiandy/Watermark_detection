import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_model(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def visualize_results(original_image, watermark_prob, detection_output):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    ax1.imshow(original_image)
    ax1.set_title(f'Classification: {"Watermark" if watermark_prob > 0.5 else "No Watermark"}\nProbability: {watermark_prob:.4f}')
    ax1.axis('off')

    ax2.imshow(original_image)
    height, width = original_image.shape[:2]
    for box, score, label in zip(detection_output['boxes'], detection_output['scores'], detection_output['labels']):
        if score > 0.3:
            box = box.cpu().numpy()
            x1, y1, x2, y2 = [int(coord * dim / 224) for coord, dim in zip(box, [width, height, width, height])]
            label_name = "r123" if label.item() == 1 else "non_r123"
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
            ax2.add_patch(rect)
            ax2.text(x1, y1, f'{label_name}: {score.item():.2f}', color='red', fontsize=8,
                     bbox=dict(facecolor='white', alpha=0.5))
    ax2.set_title('Object Detection')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()