import torch
import cv2
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
#from torchvision.models.detection import FastRCNNPredictor
from PIL import Image

# Replace this line:
# from torchvision.models.detection import FastRCNNPredictor

# Define a custom predictor
import torch.nn as nn
class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas

# CONFIGURATION
MODEL_PATH = "./fasterrcnn_coco128.pth"  # chemin vers ton modèle sauvegardé
NUM_CLASSES = 81  # Ex : 2 classes + 1 pour le fond
CONFIDENCE_THRESHOLD = 0.5
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Charger le modèle Faster R-CNN
model = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()  # Passer en mode évaluation

# Préparer les transformations
transform = T.Compose([T.ToTensor()])

# Ouvrir la webcam
cap = cv2.VideoCapture(0)  # 0 pour la webcam par défaut

while True:
    # Lire une image depuis la webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Appliquer les transformations
    image = transform(frame).unsqueeze(0).to(DEVICE)

    # Appliquer le modèle (passage en mode sans gradients)
    with torch.no_grad():
        prediction = model(image)

    # Extraire les résultats de détection
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    # Filtrer les prédictions par score (par exemple, garder celles avec un score supérieur à 0.5)
    threshold = CONFIDENCE_THRESHOLD
    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            # Dessiner la boîte de détection
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'Class {label}: {score:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Afficher l'image avec les détections
    cv2.imshow("Webcam - Faster R-CNN Detection", frame)

    # Sortir de la boucle si l'utilisateur appuie sur 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la webcam et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
