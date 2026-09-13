
import os
import sys

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"D:\osteoarthrits\SIH_OA_XGBoost\models\xray\best_densenet121.pth"

NUM_CLASSES = 5
IMAGE_SIZE = 224

# Dataset folder mapping
# train/
#   0 -> KL Grade 0
#   1 -> KL Grade 1
#   2 -> KL Grade 2
#   3 -> KL Grade 3
#   4 -> KL Grade 4

CLASS_NAMES = {
    0: "KL Grade 0",
    1: "KL Grade 1",
    2: "KL Grade 2",
    3: "KL Grade 3",
    4: "KL Grade 4",
}


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading DenseNet121 X-ray model...")

model = models.densenet121(weights=None)

model.classifier = nn.Linear(
    model.classifier.in_features,
    NUM_CLASSES
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)

model.to(device)
model.eval()

print("Model loaded successfully.")
print(f"Model classes: {NUM_CLASSES}")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_xray(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = (
            probabilities[0, predicted_class].item()
            * 100
        )

    predicted_grade = CLASS_NAMES[predicted_class]

    class_probabilities = {}

    for class_index in range(NUM_CLASSES):

        class_probabilities[CLASS_NAMES[class_index]] = (
            probabilities[0, class_index].item() * 100
        )

    return {
        "class_index": predicted_class,
        "grade": predicted_grade,
        "confidence": confidence,
        "probabilities": class_probabilities,
    }


# ============================================================
# COMMAND-LINE MODE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print()
        print("Usage:")
        print(
            r'python xray_prediction.py "path\to\xray.jpg"'
        )
        print()

        sys.exit(1)

    image_path = sys.argv[1]

    try:

        result = predict_xray(image_path)

        print()
        print("=" * 60)
        print("        OSTEOCARE X-RAY SCREENING")
        print("=" * 60)

        print()
        print(
            f"Predicted class : {result['class_index']}"
        )

        print(
            f"Predicted grade : {result['grade']}"
        )

        print(
            f"Confidence      : {result['confidence']:.2f}%"
        )

        print()
        print("Class probabilities:")
        print("-" * 40)

        for grade, probability in result[
            "probabilities"
        ].items():

            print(
                f"{grade:<15} : {probability:.2f}%"
            )

        print()
        print("=" * 60)
        print(
            "NOTE: This is an AI-based screening result,"
        )
        print(
            "not a medical diagnosis."
        )
        print("=" * 60)
        print()

    except Exception as e:

        print()
        print("X-ray prediction failed.")
        print(f"Error: {e}")
        print()

        sys.exit(1)