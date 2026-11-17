from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms

CLASS_NAMES = ["acne", "blackheads", "dark spots", "pores", "wrinkles"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, len(CLASS_NAMES))
    return model

def load_model():
    path = Path("models/efficientnet_b0_phase4_best.pth")
    if not path.exists():
        return None

    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    state = {k.removeprefix("module."): v for k, v in state.items()}

    model = build_model()
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

def predict_image(model, image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    index = probabilities.argmax().item()
    scores = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
    return CLASS_NAMES[index], float(probabilities[index]), scores
