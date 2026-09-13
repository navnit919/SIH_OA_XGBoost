import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATASET_ROOT = r"D:\sih2026\KneeXrayData\KneeXrayData\ClsKLData\kneeKL224"
MODEL_PATH = r"D:\osteoarthrits\SIH_OA_XGBoost\models\xray\best_densenet121.pth"

SPLIT = "val"
DEVICE = torch.device("cpu")
NUM_CLASSES = 5

CLASS_NAMES = [
    "KL Grade 0",
    "KL Grade 1",
    "KL Grade 2",
    "KL Grade 3",
    "KL Grade 4",
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset_path = os.path.join(DATASET_ROOT, SPLIT)

dataset = ImageFolder(
    dataset_path,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

print()
print("=" * 60)
print("X-RAY MODEL EVALUATION")
print("=" * 60)
print(f"Dataset : {dataset_path}")
print(f"Images  : {len(dataset)}")
print(f"Classes : {dataset.class_to_idx}")
print()

model = models.densenet121(weights=None)

model.classifier = nn.Linear(
    model.classifier.in_features,
    NUM_CLASSES
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

clean_state_dict = {}

for key, value in state_dict.items():
    if key.startswith("module."):
        key = key[7:]
    clean_state_dict[key] = value

model.load_state_dict(clean_state_dict)
model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print("Starting evaluation...")
print()

all_labels = []
all_predictions = []

with torch.no_grad():

    total_batches = len(loader)

    for batch_number, (images, labels) in enumerate(loader, start=1):

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.tolist())

        if batch_number % 10 == 0 or batch_number == total_batches:
            print(
                f"Processed {batch_number}/{total_batches} batches"
            )

print()
print("=" * 60)
print("EVALUATION FINISHED")
print("=" * 60)

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print()
print(f"Overall Accuracy: {accuracy * 100:.2f}%")

print()
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1, 2, 3, 4],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1, 2, 3, 4]
)

print()
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print()
print("Rows    = Actual")
print("Columns = Predicted")
print()

print("          0     1     2     3     4")

for i, row in enumerate(cm):
    print(
        f"{i}       "
        + " ".join(f"{value:5d}" for value in row)
    )

print()
print("=" * 60)
print("PER-CLASS ACCURACY")
print("=" * 60)

for i in range(NUM_CLASSES):

    total = cm[i].sum()
    correct = cm[i, i]

    if total > 0:
        class_accuracy = correct / total * 100
    else:
        class_accuracy = 0

    print(
        f"{CLASS_NAMES[i]}: "
        f"{class_accuracy:.2f}% "
        f"({correct}/{total})"
    )

print()
print("=" * 60)
print("DONE")
print("=" * 60)