import torch
import cv2
from transformers import DeformableDetrImageProcessor
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
from transformers import AutoImageProcessor, AutoModelForObjectDetection

import numpy as np
from PIL import Image


# Fonction utilitaire pour convertir les boxes du format cxcywh à xyxy
def box_cxcywh_to_xyxy(boxes):
    x_c, y_c, w, h = boxes.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

# CONFIGURATION
MODEL_PATH = "deformable_detr.pth"  # fichier contenant le state_dict
NUM_CLASSES = 81  # Nombre de classes de ton dataset (80 COCO + 1 background si entraîné comme tel)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.5

# Recréer le modèle et le processor

model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")


# Adapter le nombre de classes
in_features_class = model.detr.class_embed.in_features
in_features_bbox = model.detr.bbox_embed.in_features

model.detr.class_embed = torch.nn.Linear(in_features_class, NUM_CLASSES)
model.detr.bbox_embed = torch.nn.Linear(in_features_bbox, 4)  # 4 coords par box


model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Charger le processor (même que celui utilisé à l'entraînement)
processor = DeformableDetrImageProcessor.from_pretrained("SenseTime/deformable-detr")

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image BGR (OpenCV) en RGB (PIL)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Prétraitement de l’image avec le processor
    encoding = processor(images=image, return_tensors="pt").to(DEVICE)

    # Inférence
    with torch.no_grad():
        outputs = model(**encoding)

    # Traitement des résultats
    logits = outputs.logits.softmax(-1)[0, :, :-1]  # Retirer la classe "no-object"
    boxes = outputs.pred_boxes[0]

    scores, labels = logits.max(-1)
    keep = scores > CONFIDENCE_THRESHOLD
    scores = scores[keep]
    labels = labels[keep]
    boxes = boxes[keep]

    # Convertir les boxes (format normalisé xywh) en coordonnées absolues xyxy
    h, w = image.size[1], image.size[0]
    boxes = boxes.cpu() * torch.tensor([w, h, w, h])
    boxes = box_cxcywh_to_xyxy(boxes)

    # Dessiner les boîtes sur l’image d’origine (OpenCV BGR)
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.int().tolist()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Class {label.item()} ({score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Afficher l'image
    cv2.imshow("Webcam - Deformable DETR", frame)

    # Sortir avec 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libérer la webcam
cap.release()
cv2.destroyAllWindows()


