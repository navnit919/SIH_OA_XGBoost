
import json
from pathlib import Path

BASE_DIR = Path(r"D:\osteoarthrits\SIH_OA_XGBoost")
MODEL_DIR = BASE_DIR / "models" / "gait"

V4_PATH = MODEL_DIR / "oa_xgboost_gait_metrics_v4.json"
V5_PATH = MODEL_DIR / "oa_xgboost_gait_metrics_v5.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_value(data, possible_names):
    """
    Searches recursively for a metric key.
    Handles differences between V4 and V5 JSON structures.
    """
    if isinstance(data, dict):
        for key, value in data.items():

            key_clean = str(key).lower().replace("-", "_").replace(" ", "_")

            if key_clean in possible_names:
                return value

            result = find_value(value, possible_names)

            if result is not None:
                return result

    elif isinstance(data, list):
        for item in data:
            result = find_value(item, possible_names)

            if result is not None:
                return result

    return None


def get_metric(data, metric):
    aliases = {
        "accuracy": {
            "accuracy",
            "acc",
        },
        "precision": {
            "precision",
            "positive_predictive_value",
            "ppv",
        },
        "sensitivity": {
            "sensitivity",
            "recall",
            "true_positive_rate",
            "tpr",
        },
        "specificity": {
            "specificity",
            "true_negative_rate",
            "tnr",
        },
        "f1": {
            "f1",
            "f1_score",
            "f1score",
        },
        "roc_auc": {
            "roc_auc",
            "roc_auc_score",
            "auc",
        },
    }

    return find_value(data, aliases[metric])


def get_features(data):
    features = find_value(data, {"features", "feature_names"})

    if isinstance(features, list):
        return features

    return []


def get_confusion_matrix(data):
    cm = find_value(
        data,
        {
            "confusion_matrix",
            "confusionmatrix",
            "cm",
        }
    )

    return cm


v4 = load_json(V4_PATH)
v5 = load_json(V5_PATH)


metrics = [
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "roc_auc",
]


print("=" * 80)
print("OA GAIT MODEL COMPARISON — V4 vs V5")
print("=" * 80)

print()
print(
    f"{'Metric':<20}"
    f"{'V4':>12}"
    f"{'V5':>12}"
    f"{'Change':>12}"
    f"{'Better':>12}"
)

print("-" * 80)

results = {}

for metric in metrics:

    v4_value = get_metric(v4, metric)
    v5_value = get_metric(v5, metric)

    results[metric] = {
        "v4": v4_value,
        "v5": v5_value,
    }

    if v4_value is None or v5_value is None:

        print(
            f"{metric:<20}"
            f"{'N/A':>12}"
            f"{'N/A':>12}"
            f"{'N/A':>12}"
            f"{'N/A':>12}"
        )

        continue

    v4_value = float(v4_value)
    v5_value = float(v5_value)

    change = v5_value - v4_value

    if change > 0.000001:
        better = "V5"
    elif change < -0.000001:
        better = "V4"
    else:
        better = "Same"

    print(
        f"{metric:<20}"
        f"{v4_value * 100:>11.2f}%"
        f"{v5_value * 100:>11.2f}%"
        f"{change * 100:>+11.2f}%"
        f"{better:>12}"
    )


print()
print("=" * 80)
print("FEATURE COUNT")
print("=" * 80)

v4_features = get_features(v4)
v5_features = get_features(v5)

print()
print(f"V4 features: {len(v4_features)}")

for feature in v4_features:
    print(f"  - {feature}")

print()
print(f"V5 features: {len(v5_features)}")

for feature in v5_features:
    print(f"  - {feature}")


print()
print("=" * 80)
print("CONFUSION MATRICES")
print("=" * 80)

print()
print("V4:")
print(get_confusion_matrix(v4))

print()
print("V5:")
print(get_confusion_matrix(v5))


print()
print("=" * 80)
print("FINAL RECOMMENDATION")
print("=" * 80)

v4_sens = get_metric(v4, "sensitivity")
v5_sens = get_metric(v5, "sensitivity")

v4_auc = get_metric(v4, "roc_auc")
v5_auc = get_metric(v5, "roc_auc")

v4_acc = get_metric(v4, "accuracy")
v5_acc = get_metric(v5, "accuracy")


if all(x is not None for x in [v4_sens, v5_sens, v4_auc, v5_auc]):

    print()

    if v5_sens > v4_sens and v5_auc >= v4_auc:

        print("V5 is the stronger candidate for OA screening.")

    elif v4_sens > v5_sens and v4_auc >= v5_auc:

        print("V4 is the stronger candidate for OA screening.")

    else:

        print("Neither model clearly dominates.")
        print("The final choice should consider the screening objective.")

else:

    print()
    print("Unable to automatically determine the final model.")
    print("One or more required metrics are missing from the JSON files.")


print()
print("=" * 80)
print("KEY RESULTS")
print("=" * 80)

if v4_acc is not None and v5_acc is not None:

    print()
    print("Accuracy:")
    print(f"  V4 = {float(v4_acc) * 100:.2f}%")
    print(f"  V5 = {float(v5_acc) * 100:.2f}%")

if v4_sens is not None and v5_sens is not None:

    print()
    print("Sensitivity:")
    print(f"  V4 = {float(v4_sens) * 100:.2f}%")
    print(f"  V5 = {float(v5_sens) * 100:.2f}%")

if v4_auc is not None and v5_auc is not None:

    print()
    print("ROC-AUC:")
    print(f"  V4 = {float(v4_auc) * 100:.2f}%")
    print(f"  V5 = {float(v5_auc) * 100:.2f}%")


print()
print("=" * 80)
print("DONE")
print("=" * 80)