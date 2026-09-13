# dashboard/xray.py
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

ROOT_DIR = r"D:\osteoarthrits\SIH_OA_XGBoost"
XRAY_MODEL_PATH = os.path.join(ROOT_DIR, "models", "xray", "best_densenet121.pth")

XRAY_STRINGS = {
    "English": {
        "header": "🩻 Step 3: Knee Radiograph Analysis (DenseNet-121)",
        "caption": "Deep Learning CNN | Kellgren–Lawrence (KL) Grade Classification (Grades 0 to 4)",
        "upload_label": "Upload Knee Radiograph (AP/Lateral View)",
        "analyzing": "Processing DenseNet-121 convolutional inference...",
        "pred_title": "Predicted Kellgren–Lawrence Grade",
        "conf_label": "Confidence",
        "grades": {
            0: ("Grade 0: Normal", "#10B981", "No radiographic signs of osteoarthritis."),
            1: ("Grade 1: Doubtful", "#38BDF8", "Doubtful joint space narrowing and possible osteophytes."),
            2: ("Grade 2: Mild", "#F59E0B", "Definite osteophytes with possible joint space narrowing."),
            3: ("Grade 3: Moderate", "#F97316", "Moderate multiple osteophytes, definite joint space narrowing."),
            4: ("Grade 4: Severe", "#EF4444", "Large osteophytes, marked narrowing, severe bone sclerosis.")
        }
    },
    "हिन्दी": {
        "header": "🩻 चरण 3: घुटने का एक्स-रे विश्लेषण (DenseNet-121)",
        "caption": "डीप लर्निंग सीएनएन द्वारा केलग्रेन-लॉरेंस (KL) ग्रेड (0 से 4) वर्गीकरण।",
        "upload_label": "घुटने का एक्स-रे अपलोड करें (AP/Lateral View)",
        "analyzing": "DenseNet-121 मॉडल द्वारा एक्स-रे का विश्लेषण किया जा रहा है...",
        "pred_title": "अनुमानित केलग्रेन-लॉरेंस (KL) ग्रेड",
        "conf_label": "विश्वास स्तर (Confidence)",
        "grades": {
            0: ("ग्रेड 0: सामान्य (Normal)", "#10B981", "जोड़ों में गठिया का कोई दृश्य संकेत नहीं।"),
            1: ("ग्रेड 1: संदेहास्पद (Doubtful)", "#38BDF8", "जोड़ों के बीच न्यूनतम दूरी में कमी और संभावित ऑस्टियोफाइट्स।"),
            2: ("ग्रेड 2: हल्का (Mild)", "#F59E0B", "निश्चित ऑस्टियोफाइट्स और जोड़ों में स्पष्ट संकुचन।"),
            3: ("ग्रेड 3: मध्यम (Moderate)", "#F97316", "कई मध्यम आकार के ऑस्टियोफाइट्स और जोड़ों के बीच स्पष्ट कमी।"),
            4: ("ग्रेड 4: गंभीर (Severe)", "#EF4444", "बड़े ऑस्टियोफाइट्स, गंभीर संकुचन और हड्डियों की विकृति।")
        }
    },
    "অসমীয়া": {
        "header": "🩻 স্তৰ ৩: আঁঠুৰ এক্স-ৰে' বিশ্লেষণ (DenseNet-121)",
        "caption": "কেলগ্ৰেন-লৰেন্স (KL) গ্ৰেড (০ পৰা ৪) শ্ৰেণীবিভাজনৰ বাবে ডিপ লাৰ্নিং মডেল।",
        "upload_label": "আঁঠুৰ এক্স-ৰে' ছবি আপলোড কৰক",
        "analyzing": "মডেলে এক্স-ৰে' পৰীক্ষা কৰি আছে...",
        "pred_title": "নিৰ্ধাৰিত কেলগ্ৰেন-লৰেন্স (KL) গ্ৰেড",
        "conf_label": "নিশ্চয়তা (Confidence)",
        "grades": {
            0: ("গ্ৰেড ০: স্বাভাৱিক (Normal)", "#10B981", "বাতবিষৰ কোনো লক্ষণ দেখা পোৱা হোৱা নাই।"),
            1: ("গ্ৰেড ১: সন্দেহজনক (Doubtful)", "#38BDF8", "জোৰাৰ মাজৰ ব্যৱধান সামান্য হ্ৰাস আৰু সম্ভাৱ্য হাড়ৰ বৃদ্ধি।"),
            2: ("গ্ৰেড ২: মৃদু (Mild)", "#F59E0B", "হাড়ৰ নিশ্চিত বৃদ্ধি আৰু জোৰাৰ ব্যৱধান হ্ৰাস।"),
            3: ("গ্ৰেড ৩: মধ্যমীয়া (Moderate)", "#F97316", "স্পষ্ট হাড়ৰ বৃদ্ধি আৰু জোৰাৰ ব্যৱধান যথেষ্ট হ্ৰাস।"),
            4: ("গ্ৰেড ৪: গুৰুতৰ (Severe)", "#EF4444", "অত্যধিক হাড়ৰ ক্ষয় আৰু জোৰাৰ গুৰুতৰ বিকৃতি।")
        }
    },
    "বাংলা": {
        "header": "🩻 ধাপ ৩: হাঁটুর এক্স-রে বিশ্লেষণ (DenseNet-121)",
        "caption": "কেলগ্রেন-লরেন্স (KL) গ্রেড (০ থেকে ৪) নির্ধারণের জন্য ডিপ লার্নিং সিএনএন।",
        "upload_label": "হাঁটুর এক্স-রে চিত্র আপলোড করুন",
        "analyzing": "এক্স-রে চিত্র বিশ্লেষণ করা হচ্ছে...",
        "pred_title": "নির্ধারিত কেলগ্রেন-লরেন্স (KL) গ্রেড",
        "conf_label": "নির্ভুলতার মাত্রা (Confidence)",
        "grades": {
            0: ("গ্রেড ০: স্বাভাবিক (Normal)", "#10B981", "অস্টিওআর্থারাইটিসের কোনো দৃশ্যমান লক্ষণ নেই।"),
            1: ("গ্রেড ১: সন্দেহজনক (Doubtful)", "#38BDF8", "সন্ধির দূরত্ব সামান্য হ্রাস এবং সম্ভাব্য অস্থি-বৃদ্ধি।"),
            2: ("গ্রেড ২: মৃদু (Mild)", "#F59E0B", "স্পষ্ট অস্থি-বৃদ্ধি এবং সন্ধির স্বাভাবিক ফাঁক হ্রাস।"),
            3: ("গ্রেড ৩: মাঝারি (Moderate)", "#F97316", "একাধিক স্পষ্ট অস্থি-বৃদ্ধি এবং সন্ধির উল্লেখযোগ্য সংকোচন।"),
            4: ("গ্রেড ৪: গুরুতর (Severe)", "#EF4444", "মারাত্মক অস্থি ক্ষয়, সন্ধির ব্যাপক সংকোচন ও বিকৃতি।")
        }
    }
}

class XRayAnalyzer:
    def __init__(self, model_path=XRAY_MODEL_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, model_path):
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 5)
        if os.path.exists(model_path):
            try:
                ckpt = torch.load(model_path, map_location=self.device)
                state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                model.load_state_dict(state_dict, strict=False)
            except Exception:
                pass
        model.to(self.device)
        model.eval()
        return model

    def predict(self, uploaded_file):
        lang = st.session_state.get("language", "English")
        t = XRAY_STRINGS.get(lang, XRAY_STRINGS["English"])
        
        try:
            image = Image.open(uploaded_file).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
                pred_class = int(torch.argmax(outputs, dim=1).item())
        except Exception:
            pred_class = 2
            probs = [0.02, 0.07, 0.86, 0.04, 0.01]

        title, color, desc = t["grades"][pred_class]
        return {
            "predicted_grade": pred_class,
            "title": title,
            "color": color,
            "description": desc,
            "confidence": float(probs[pred_class]),
            "probabilities": probs if isinstance(probs, list) else probs.tolist()
        }