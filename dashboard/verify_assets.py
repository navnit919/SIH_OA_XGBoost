# verify_assets.py
import os
import torch
import joblib
import json

ROOT = r"D:\osteoarthrits\SIH_OA_XGBoost"

print("="*60)
print("1. VERIFYING X-RAY DENSENET-121 CHECKPOINT")
print("="*60)
xray_model_path = os.path.join(ROOT, "models", "xray", "best_densenet121.pth")
if os.path.exists(xray_model_path):
    checkpoint = torch.load(xray_model_path, map_location="cpu")
    # Handle state_dict vs full model
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    
    # Inspect final classification layer
    fc_keys = [k for k in state_dict.keys() if "classifier" in k or "fc" in k]
    print(f"[+] Found classifier keys: {fc_keys}")
    for k in fc_keys:
        if "weight" in k:
            out_features, in_features = state_dict[k].shape
            print(f"[+] Output shape for '{k}': {out_features} classes (KL 0-4 expected = 5)")
else:
    print(f"[-] File not found: {xray_model_path}")

print("\n" + "="*60)
print("2. VERIFYING GAIT XGBOOST V5 MODEL")
print("="*60)
gait_model_path = os.path.join(ROOT, "models", "gait", "oa_xgboost_gait_model_v5.pkl")
if os.path.exists(gait_model_path):
    gait_model = joblib.load(gait_model_path)
    print(f"[+] Model Type: {type(gait_model)}")
    if hasattr(gait_model, "feature_names_in_"):
        print(f"[+] Number of features: {len(gait_model.feature_names_in_)}")
        print(f"[+] Features: {list(gait_model.feature_names_in_)[:6]}...")
    if hasattr(gait_model, "classes_"):
        print(f"[+] Target classes: {gait_model.classes_}")
else:
    print(f"[-] File not found: {gait_model_path}")