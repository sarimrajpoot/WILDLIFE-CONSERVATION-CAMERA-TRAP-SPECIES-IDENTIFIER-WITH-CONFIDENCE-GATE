import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
import numpy as np
import torch.nn.functional as F

# -----------------------
# Device
# -----------------------
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# -----------------------
# Image Transform
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------
# Class Mapping
# -----------------------
idx_to_class = {
    0: 'cane',
    1: 'cavallo',
    2: 'elefante',
    3: 'farfalla',
    4: 'gallina',
    5: 'gatto',
    6: 'mucca',
    7: 'pecora',
    8: 'ragno',
    9: 'scoiattolo'
}

# -----------------------
# Rare species
# -----------------------
RARE_SPECIES = [
    "elefante",
    "farfalla",
    "scoiattolo"
]

CONFIDENCE_THRESHOLD = 0.85
UNCERTAINTY_THRESHOLD = 0.08

# -----------------------
# Model
# -----------------------
model = models.resnet18(weights=None)

num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 10)
)

model.load_state_dict(
    torch.load(
        "models/best_phase2_finetuned.pth",
        map_location=device
    )
)

model = model.to(device)

# -----------------------
# MC Dropout
# -----------------------
def enable_mc_dropout(model):

    model.eval()

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(model, image_tensor, n_passes=15):

    enable_mc_dropout(model)

    image_tensor = image_tensor.to(device)

    predictions = []

    with torch.no_grad():

        for _ in range(n_passes):

            outputs = model(image_tensor)

            probs = F.softmax(outputs, dim=1)

            predictions.append(probs.cpu().numpy())

    predictions = np.array(predictions)

    mean_probs = predictions.mean(axis=0)
    std_probs = predictions.std(axis=0)

    return mean_probs, std_probs

# -----------------------
# Confidence Gate
# -----------------------
def confidence_gate(mean_probs, std_probs):

    confidence = np.max(mean_probs)
    pred_idx = np.argmax(mean_probs)
    predicted_class = idx_to_class[pred_idx]
    uncertainty = std_probs[0][pred_idx]

    top2 = np.argsort(mean_probs[0])[-2:]

    top1_conf = mean_probs[0][top2[-1]]
    top2_conf = mean_probs[0][top2[-2]]

    top2_classes = [
        idx_to_class[top2[-1]],
        idx_to_class[top2[-2]]
    ]

    # Rare species priority
    if predicted_class in RARE_SPECIES and confidence > CONFIDENCE_THRESHOLD:

        category = "REVIEW: RARE SPECIES MATCH"
        message = "Possible rare species sighting — flagged for priority review."

    # Ambiguous
    elif abs(top1_conf - top2_conf) <= 0.10:

        category = "REVIEW: AMBIGUOUS"
        message = f"Model uncertain between {top2_classes[0]} and {top2_classes[1]}."

    # Low confidence
    elif confidence <= CONFIDENCE_THRESHOLD or uncertainty >= UNCERTAINTY_THRESHOLD:

        category = "REVIEW: LOW CONFIDENCE"
        message = "Prediction confidence is low — send for human review."

    # High confidence
    else:

        category = "HIGH CONFIDENCE"
        message = "Prediction accepted automatically."

    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "uncertainty_score": float(uncertainty),
        "review_category": category,
        "message": message
    }

# -----------------------
# Folder Processing
# -----------------------
def process_folder(folder_path):

    results = []

    for filename in os.listdir(folder_path):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(folder_path, filename)

        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)

        mean_probs, std_probs = mc_dropout_predict(model, image)

        result = confidence_gate(mean_probs, std_probs)

        results.append({
            "filename": filename,
            "predicted_class": result["predicted_class"],
            "confidence": round(result["confidence"], 4),
            "uncertainty_score": round(result["uncertainty_score"], 4),
            "review_category": result["review_category"]
        })

    return pd.DataFrame(results)

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":

    folder_path = "sample_images"

    df = process_folder(folder_path)

    df.to_csv("camera_trap_predictions.csv", index=False)

    print(df)

    print("\nReview Distribution (%)")
    print(
        df["review_category"].value_counts(normalize=True) * 100
    )