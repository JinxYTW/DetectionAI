# fasterrcnn_inference.py
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# CONFIGURATION
MODEL_PATH = "./fasterrcnn_coco128.pth"  # chemin vers le fichier .pth
IMAGE_DIR = "./img"   # dossier contenant les images
NUM_CLASSES = 81                       # pour coco128 (80 + 1 background)
CONFIDENCE_THRESHOLD = 0.5
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Transformation unique
transform = T.Compose([T.ToTensor()])

# Charger le modèle et son architecture
model = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Lister les images dans le dossier
def get_image_paths(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith((".jpg", ".png"))]

def predict_and_plot(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = model(img_tensor)[0]

    boxes = prediction['boxes'].cpu()
    labels = prediction['labels'].cpu()
    scores = prediction['scores'].cpu()

    # Affichage avec matplotlib
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    for box, label, score in zip(boxes, labels, scores):
        if score > CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f'{label.item()} ({score:.2f})',
                    color='black', bbox=dict(facecolor='lime', alpha=0.5))

    ax.axis('off')
    plt.show()

if __name__ == "__main__":
    image_paths = get_image_paths(IMAGE_DIR)
    for img_path in image_paths:
        print(f"Traitement de {img_path}...")
        predict_and_plot(img_path)
