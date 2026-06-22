import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import torch.nn.functional as F

# -----------------------
# Page config
# -----------------------
st.set_page_config(
    page_title="Wildlife Camera Trap Classifier",
    layout="centered"
)

st.title("🐾 Wildlife Camera Trap Species Identifier")
st.write("Upload an image to classify species + get confidence gate analysis")

# -----------------------
# Device
# -----------------------
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# -----------------------
# Transform
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
# Class mapping
# -----------------------
idx_to_class = {
    0: "dog",
    1: "horse",
    2: "elephant",
    3: "butterfly",
    4: "chicken",
    5: "cat",
    6: "cow",
    7: "sheep",
    8: "spider",
    9: "squirrel"
}

RARE_SPECIES = ["elephant", "butterfly", "squirrel"]

# -----------------------
# Load model
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

model.to(device)
model.eval()

# -----------------------
# MC Dropout
# -----------------------
def mc_dropout_predict(model, image_tensor, n_passes=15):

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

    image_tensor = image_tensor.to(device)

    preds = []

    with torch.no_grad():
        for _ in range(n_passes):
            out = model(image_tensor)
            probs = F.softmax(out, dim=1)
            preds.append(probs.cpu().numpy())

    preds = np.array(preds)

    return preds.mean(axis=0), preds.std(axis=0)

# -----------------------
# Confidence Gate
# -----------------------
def confidence_gate(mean_probs, std_probs):

    confidence = np.max(mean_probs)
    pred_idx = np.argmax(mean_probs)
    predicted_class = idx_to_class[pred_idx]
    uncertainty = std_probs[0][pred_idx]

    if predicted_class in RARE_SPECIES and confidence > 0.85:
        category = "REVIEW: RARE SPECIES MATCH"
        message = "🚨 Rare species detected! Send for priority review."

    elif confidence < 0.85 or uncertainty > 0.08:
        category = "REVIEW: LOW CONFIDENCE"
        message = "⚠️ Low confidence prediction. Needs human review."

    else:
        category = "HIGH CONFIDENCE"
        message = "✅ Prediction is reliable."

    return predicted_class, confidence, uncertainty, category, message

# -----------------------
# Upload image
# -----------------------
uploaded_file = st.file_uploader(
    "Upload a camera-trap image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)

    mean_probs, std_probs = mc_dropout_predict(model, img_tensor)

    pred_class, conf, uncert, category, message = confidence_gate(
        mean_probs, std_probs
    )

    st.subheader("Prediction Result")

    st.write(f"**Species:** {pred_class}")
    st.write(f"**Confidence:** {conf:.3f}")
    st.write(f"**Uncertainty:** {uncert:.3f}")

    st.write(f"**Category:** {category}")
    st.info(message)