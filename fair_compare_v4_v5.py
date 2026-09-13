
import json
from pathlib import Path

BASE_DIR = Path(r"D:\osteoarthrits\SIH_OA_XGBoost")
MODEL_DIR = BASE_DIR / "models" / "gait"

V4_PATH = MODEL_DIR / "oa_xgboost_gait_metrics_v4.json"
V5_PATH = MODEL_DIR / "oa_xgboost_gait_metrics_v5.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_overall_metrics(data):
    """
    Find the overall OOF metrics dictionary.
    Avoid accidentally reading metrics from Fold 1.
    """

    # Most likely locations
    candidates = [
        data.get("overall_oof"),
        data.get("overall_metrics"),
        data.get("metrics"),
    ]

    for candidate in candidates:
        if isinstance(candidate, dict):
            keys = {str(k).lower() for k in candidate.keys()}

            if (
                "accuracy" in keys
                or "sensitivity" in keys
                or "recall" in keys
            ):
                return candidate

    # Recursive search for dictionaries containing multiple
    # standard metrics.
    def recursive_search(obj):

        if isinstance(obj, dict):

            keys = {
                str(k).lower().replace("-", "_").replace(" ", "_")
                for k in obj.keys()
            }

            metric_keys = {
                "accuracy",
                "precision",
                "sensitivity",
                "specificity",
                "recall",
                "f1",
                "f1_score",
                "roc_auc",
            }

            matches = len(keys.intersection(metric_keys))

            if matches >= 3:
                return obj

            for value in obj.values():
                result = recursive_search(value)

                if result is not None:
                    return result

        elif isinstance(obj, list):

            for item in obj:
                result = recursive_search(item)

                if result is not None:
                    return result

        return None

    return recursive_search(data)


def get_metric(metrics, names):

    if metrics is None:
        return None

    normalized = {
        str(k).lower().replace("-", "_").replace(" ", "_"): v
        for k, v in metrics.items()
    }

    for name in names:
        if name in normalized:
            return normalized[name]

    return None


def print_model_metrics(name, data):

    metrics = find_overall_metrics(data)

    print()
    print("=" * 80)
    print(f"{name} — OVERALL METRICS FOUND IN JSON")
    print("=" * 80)

    if metrics is None:
        print("Could not locate overall metrics.")
        return None

    aliases = {
        "accuracy": ["accuracy", "acc"],
        "precision": ["precision", "ppv"],
        "sensitivity": [
            "sensitivity",
            "recall",
            "true_positive_rate",
            "tpr",
        ],
        "specificity": [
            "specificity",
            "true_negative_rate",
            "tnr",
        ],
        "f1": ["f1", "f1_score", "f1score"],
        "roc_auc": ["roc_auc", "roc_auc_score", "auc"],
    }

    result = {}

    for metric, names in aliases.items():

        value = get_metric(metrics, names)

        if value is None:
            result[metric] = None
            print(f"{metric:<15}: N/A")
        else:
            value = float(value)
            result[metric] = value
            print(f"{metric:<15}: {value * 100:.2f}%")

    return result


v4 = load_json(V4_PATH)
v5 = load_json(V5_PATH)

v4_metrics = print_model_metrics("V4", v4)
v5_metrics = print_model_metrics("V5", v5)


print()
print("=" * 80)
print("FAIR V4 vs V5 COMPARISON")
print("Same 0.50 decision threshold")
print("=" * 80)

if v4_metrics is None or v5_metrics is None:

    print()
    print("ERROR: Could not find both overall metric dictionaries.")
    print("The JSON structures need to be inspected.")

else:

    metrics = [
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "roc_auc",
    ]

    print()
    print(
        f"{'Metric':<20}"
        f"{'V4':>12}"
        f"{'V5':>12}"
        f"{'Change':>12}"
        f"{'Better':>12}"
    )

    print("-" * 80)

    for metric in metrics:

        v4_value = v4_metrics.get(metric)
        v5_value = v5_metrics.get(metric)

        if v4_value is None or v5_value is None:

            print(
                f"{metric:<20}"
                f"{'N/A':>12}"
                f"{'N/A':>12}"
                f"{'N/A':>12}"
                f"{'N/A':>12}"
            )

            continue

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
print("SCREENING-FOCUSED INTERPRETATION")
print("=" * 80)

if v4_metrics and v5_metrics:

    v4_sens = v4_metrics.get("sensitivity")
    v5_sens = v5_metrics.get("sensitivity")

    v4_spec = v4_metrics.get("specificity")
    v5_spec = v5_metrics.get("specificity")

    v4_f1 = v4_metrics.get("f1")
    v5_f1 = v5_metrics.get("f1")

    v4_auc = v4_metrics.get("roc_auc")
    v5_auc = v5_metrics.get("roc_auc")

    print()

    if v4_sens is not None and v5_sens is not None:
        print(
            f"Sensitivity: V4={v4_sens * 100:.2f}% | "
            f"V5={v5_sens * 100:.2f}%"
        )

    if v4_spec is not None and v5_spec is not None:
        print(
            f"Specificity: V4={v4_spec * 100:.2f}% | "
            f"V5={v5_spec * 100:.2f}%"
        )

    if v4_f1 is not None and v5_f1 is not None:
        print(
            f"F1 Score:    V4={v4_f1 * 100:.2f}% | "
            f"V5={v5_f1 * 100:.2f}%"
        )

    if v4_auc is not None and v5_auc is not None:
        print(
            f"ROC-AUC:     V4={v4_auc * 100:.2f}% | "
            f"V5={v5_auc * 100:.2f}%"
        )

    print()
    print("Important:")
    print(
        "For OA screening, sensitivity is particularly important "
        "because missing an OA-positive case is undesirable."
    )

    print()
    print(
        "Do NOT use accuracy alone to select the final model."
    )


print()
print("=" * 80)
print("DONE")
print("=" * 80)