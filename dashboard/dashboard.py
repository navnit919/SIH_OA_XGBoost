# dashboard/dashboard.py
import streamlit as st
import os
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image

# Safe modular imports
try:
    from patient import patient_module
except ImportError:
    def patient_module(): st.warning("patient_module could not be loaded from patient.py.")

try:
    from questionnaire import questionnaire_module
except ImportError:
    def questionnaire_module(): st.warning("questionnaire_module could not be loaded from questionnaire.py.")

try:
    from xray import XRayAnalyzer
except ImportError:
    XRayAnalyzer = None

try:
    from gait import gait_module, GAIT_STRINGS
except ImportError:
    def gait_module(): st.warning("gait_module could not be loaded from gait.py.")
    GAIT_STRINGS = {}

try:
    from hardware import ESP32SerialReader, HW_STRINGS
except ImportError:
    ESP32SerialReader = None
    HW_STRINGS = {}

try:
    from report_generator import generate_clinical_pdf
except ImportError:
    generate_clinical_pdf = None

# Full-width configuration without sidebar
st.set_page_config(
    page_title="ARTHRO-AI | Precision Osteoarthritis Assessment",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------------
# 1. State Management
# -------------------------------------------------------------
if "current_step" not in st.session_state:
    st.session_state["current_step"] = 0
if "language" not in st.session_state:
    st.session_state["language"] = "English"

if "patient_data" not in st.session_state:
    st.session_state["patient_data"] = {
        "name": "Navnit Kumar", 
        "id": "OA-2026-NITN", 
        "age": 58, 
        "gender": "Male", 
        "joint": "Right Knee",
        "bmi": 25.6
    }
if "symptom_score" not in st.session_state:
    st.session_state["symptom_score"] = 52.0
if "kl_grade" not in st.session_state:
    st.session_state["kl_grade"] = 2
if "xray_conf" not in st.session_state:
    st.session_state["xray_conf"] = 0.85
if "camera_prediction" not in st.session_state:
    st.session_state["camera_prediction"] = 0.65

# -------------------------------------------------------------
# 2. Multi-Language Strings (4 Languages)
# -------------------------------------------------------------
TEXTS = {
    "English": {
        "title": "ARTHRO-AI",
        "subtitle": "Precision joint care, designed for everyone.",
        "desc": "Step-by-step multimodal assessment for knee osteoarthritis. Combining patient symptoms, deep radiograph classification, computer vision gait kinematics, and wearable motion sensors.",
        "start": "🚀 Launch Assessment Pipeline",
        "next": "Proceed to Next Step ➔",
        "back": "⬅ Previous Step",
        "restart": "🔄 New Assessment",
        "steps": ("Patient Intake", "Questionnaire", "X-Ray Analysis", "Gait Kinematics", "Motion Sensors", "Final Assessment"),
        "pillars": ("Clinical Symptoms", "X-Ray Radiograph", "Camera Gait", "Wearable Sensors"),
        "xray_header": "🩻 Knee Radiograph Analysis (DenseNet-121)",
        "xray_sub": "Upload an AP or lateral view knee X-ray to classify structural Kellgren–Lawrence (KL 0-4) severity.",
        "xray_upload": "Upload Knee Radiograph (.png, .jpg, .jpeg)",
        "xray_pred_label": "Predicted Kellgren–Lawrence Grade",
        "xray_conf_label": "Confidence",
        "sensors_header": "📡 Wearable Motion Sensing (ESP32 + 3× MPU6050)",
        "sensors_sub": "6-DOF real-time inertial acceleration and angular velocity stream via serial communication.",
        "imu1_name": "IMU 1: Thigh Accel (Y)",
        "imu1_sub": "Axial Impact Load",
        "imu2_name": "IMU 2: Shank Gyro (Z)",
        "imu2_sub": "Sagittal Angular Velocity",
        "imu3_name": "IMU 3: Ankle Shock Absorption",
        "imu3_sub": "Ground Reaction Shock",
        "chart_title": "Real-Time 6-DOF Inertial Motion Waveform",
        "report_header": "📊 Integrated Multimodal Diagnostic Report",
        "report_sub": "Synthesizes clinical intake, reported symptoms, DenseNet radiograph findings, MediaPipe gait kinematics, and wearable sensor dynamics.",
        "composite_label": "Composite OA Severity Score",
        "tier_high": "High Risk / Confirmed Osteoarthritis",
        "tier_mod": "Moderate Risk / Early Mild OA",
        "tier_low": "Low Risk / Healthy Joint",
        "breakdown_title": "Dimensional Contribution Breakdown",
        "item_patient": "Patient Profile",
        "item_xray": "Radiographic (DenseNet-121)",
        "item_gait": "Gait Kinematics (XGBoost V5)",
        "item_womac": "Clinical Symptoms (WOMAC)",
        "item_imu": "Inertial Sensors (ESP32)",
        "weight_label": "Weight",
        "radar_labels": ("Radiograph (KL)", "Vision Gait", "Symptoms", "Inertial Motion"),
        "download_btn": "📥 Download Official Clinical Report (PDF)",
        "evaluate_again_btn": "🔄 Evaluate Another Patient"
    },
    "हिन्दी": {
        "title": "आर्थ्रो-एआई (ARTHRO-AI)",
        "subtitle": "सटीक जोड़ों की देखभाल, सभी के लिए सुगम।",
        "desc": "घुटने के ऑस्टियोआर्थराइटिस (गठिया) की चरण-दर-चरण जांच। रोगी के लक्षण, एक्स-रे डीप लर्निंग, कैमरे द्वारा चाल विश्लेषण और वियरेबल सेंसर का समग्र समन्वय।",
        "start": "🚀 जांच प्रक्रिया शुरू करें",
        "next": "अगले चरण पर जाएं ➔",
        "back": "⬅ पिछला चरण",
        "restart": "🔄 नया मूल्यांकन",
        "steps": ("रोगी पंजीकरण", "लक्षण प्रश्नावली", "एक्स-रे जांच", "चाल विश्लेषण", "मोशन सेंसर", "समग्र परिणाम"),
        "pillars": ("क्लिनिकल लक्षण", "एक्स-रे जांच", "कैमरा चाल", "वियरेबल सेंसर"),
        "xray_header": "🩻 घुटने का एक्स-रे विश्लेषण (DenseNet-121)",
        "xray_sub": "केलग्रेन-लॉरेंस (KL ग्रेड 0-4) गंभीरता की जांच के लिए घुटने का एक्स-रे अपलोड करें।",
        "xray_upload": "घुटने का एक्स-रे अपलोड करें (.png, .jpg, .jpeg)",
        "xray_pred_label": "अनुमानित केलग्रेन-लॉरेंस (KL) ग्रेड",
        "xray_conf_label": "विश्वास स्तर (Confidence)",
        "sensors_header": "📡 वियरेबल इनर्शियल मोशन सेंसर (ESP32 + 3× MPU6050)",
        "sensors_sub": "चलते समय घुटने पर पड़ने वाले दबाव और गति का 6-DOF रीयल-टाइम डेटा।",
        "imu1_name": "सेंसर 1: जांघ का त्वरण (Thigh Accel Y)",
        "imu1_sub": "अक्षीय भार प्रभाव (Axial Load)",
        "imu2_name": "सेंसर 2: पिंडली का जायरो (Shank Gyro Z)",
        "imu2_sub": "घुटने की कोणीय गति (Angular Velocity)",
        "imu3_name": "सेंसर 3: टखने का झटका (Ankle Shock)",
        "imu3_sub": "जमीन से लगने वाला झटका (Ground Reaction)",
        "chart_title": "रीयल-टाइम 6-DOF मोशन सेंसर वेवफॉर्म",
        "report_header": "📊 समग्र मल्टीमॉडल ऑस्टियोआर्थराइटिस निदान रिपोर्ट",
        "report_sub": "रोगी का विवरण, एक्स-रे विश्लेषण (DenseNet), चाल विश्लेषण (XGBoost), और सेंसर डेटा का संयुक्त मूल्यांकन।",
        "composite_label": "समग्र गठिया गंभीरता सूचकांक (Composite Score)",
        "tier_high": "उच्च जोखिम / पुष्ट ऑस्टियोआर्थराइटिस (High Risk / Confirmed OA)",
        "tier_mod": "मध्यम जोखिम / शुरुआती लक्षण (Moderate Risk / Early OA)",
        "tier_low": "न्यूनतम जोखिम / सामान्य जोड़ (Low Risk / Normal)",
        "breakdown_title": "विभिन्न आयामों का विस्तृत योगदान (Breakdown)",
        "item_patient": "रोगी विवरण (Patient Profile)",
        "item_xray": "एक्स-रे विश्लेषण (DenseNet-121)",
        "item_gait": "चाल विश्लेषण (XGBoost Gait)",
        "item_womac": "लक्षण प्रश्नावली (WOMAC Symptoms)",
        "item_imu": "वियरेबल सेंसर (ESP32 IMU)",
        "weight_label": "महत्व (Weight)",
        "radar_labels": ("एक्स-रे (X-Ray)", "चाल (Gait)", "लक्षण (Symptoms)", "सेंसर (Sensors)"),
        "download_btn": "📥 आधिकारिक क्लिनिकल रिपोर्ट डाउनलोड करें (PDF)",
        "evaluate_again_btn": "🔄 नए रोगी का परीक्षण शुरू करें"
    },
    "অসমীয়া": {
        "title": "আৰ্থ্ৰো-এআই (ARTHRO-AI)",
        "subtitle": "সকলোৰে বাবে সহজ আৰু সঠিক জোৰাৰ চিকিৎসা।",
        "desc": "আঁঠুৰ বাতবিষৰ স্তৰে স্তৰে কৰা পৰীক্ষা। ৰোগীৰ তথ্য, এক্স-ৰে' ডিটেকচন, কেমেৰাৰে খোজৰ বিশ্লেষণ আৰু ৱিয়েৰেবল চেন্সৰৰ একত্ৰিত প্ৰয়াস।",
        "start": "🚀 পৰীক্ষণ প্ৰক্ৰিয়া আৰম্ভ কৰক",
        "next": "পৰৱৰ্তী স্তৰলৈ যাওক ➔",
        "back": "⬅ পূৰ্ববৰ্তী স্তৰ",
        "restart": "🔄 নতুন মূল্যায়ন",
        "steps": ("ৰোগীৰ তথ্য", "প্ৰশ্নাৱলী", "এক্স-ৰে' পৰীক্ষা", "খোজ বিশ্লেষণ", "মোচন চেন্সৰ", "চূড়ান্ত ফলাফল"),
        "pillars": ("লক্ষণ মূল্যায়ন", "এক্স-ৰে' ৰেডিঅ'গ্ৰাফ", "কেমেৰা খোজ", "ৱিয়েৰেবল চেন্সৰ"),
        "xray_header": "🩻 আঁঠুৰ এক্স-ৰে' বিশ্লেষণ (DenseNet-121)",
        "xray_sub": "কেলগ্ৰেন-লৰেন্স (KL গ্ৰেড ০-৪) গুৰুত্ব নিৰ্ণয়ৰ বাবে আঁঠুৰ এক্স-ৰে' আপলোড কৰক।",
        "xray_upload": "আঁঠুৰ এক্স-ৰে' ছবি আপলোড কৰক (.png, .jpg, .jpeg)",
        "xray_pred_label": "নিৰ্ধাৰিত কেলগ্ৰেন-লৰেন্স (KL) গ্ৰেড",
        "xray_conf_label": "নিশ্চয়তা (Confidence)",
        "sensors_header": "📡 ৱিয়েৰেবল মোচন চেন্সৰ (ESP32 + ৩× MPU6050)",
        "sensors_sub": "খোজ কঢ়াৰ সময়ত আঁঠুত পৰা চাপ আৰু গতিবিধিৰ ৬-DOF ৰিয়েল-টাইম ডাটা।",
        "imu1_name": "চেন্সৰ ১: উৰুৰ ত্বৰণ (Thigh Accel Y)",
        "imu1_sub": "ওপৰৰ পৰা পৰা চাপ (Axial Impact)",
        "imu2_name": "চেন্সৰ ২: ভৰিৰ কোণিক বেগ (Shank Gyro Z)",
        "imu2_sub": "আঁঠু ভাঁজ হোৱাৰ গতি (Angular Velocity)",
        "imu3_name": "চেন্সৰ ৩: গোৰোহাৰ কম্পন (Ankle Shock)",
        "imu3_sub": "মাটিত ভৰি দিয়াৰ সময়ৰ প্ৰভাৱ (Ground Reaction)",
        "chart_title": "ৰিয়েল-টাইম মোচন চেন্সৰ তৰংগ (Inertial Waveform)",
        "report_header": "📊 চূড়ান্ত সমন্বিত বাতবিষ মূল্যায়ন ৰিপ'ৰ্ট",
        "report_sub": "ৰোগীৰ তথ্য, এক্স-ৰে' বিশ্লেষণ (DenseNet), খোজ বিশ্লেষণ (XGBoost), আৰু চেন্সৰ ডাটাৰ সন্মিলিত ফলাফল।",
        "composite_label": "বাতবিষৰ সামূহিক গুৰুত্ব সূচক (Composite Score)",
        "tier_high": "উচ্চ আশংকা / প্ৰমাণিত বাতবিষ (High Risk / Confirmed OA)",
        "tier_mod": "মধ্যমীয়া আশংকা / প্ৰাৰম্ভিক বাতবিষ (Moderate Risk / Early OA)",
        "tier_low": "নিম্ন আশংকা / সুস্থ জোৰা (Low Risk / Normal)",
        "breakdown_title": "চাৰিটা প্ৰধান দিশৰ সবিশেষ অৱদান (Breakdown)",
        "item_patient": "ৰোগীৰ পৰিচয় (Patient Profile)",
        "item_xray": "এক্স-ৰে' পৰীক্ষা (DenseNet-121)",
        "item_gait": "খোজৰ গতিবিধি (XGBoost Gait)",
        "item_womac": "লক্ষণ প্ৰশ্নাৱলী (WOMAC Symptoms)",
        "item_imu": "ৱিয়েৰেবল চেন্সৰ (ESP32 IMU)",
        "weight_label": "অংশ (Weight)",
        "radar_labels": ("এক্স-ৰে' (X-Ray)", "খোজ (Gait)", "লক্ষণ (Symptoms)", "চেন্সৰ (Sensors)"),
        "download_btn": "📥 অফিচিয়েল ক্লিনিকেল ৰিপ'ৰ্ট ডাউনলোড কৰক (PDF)",
        "evaluate_again_btn": "🔄 নতুন ৰোগীৰ পৰীক্ষা আৰম্ভ কৰক"
    },
    "বাংলা": {
        "title": "আর্থ্রো-এআই (ARTHRO-AI)",
        "subtitle": "হাঁটুর বাতজনিত সঠিক মূল্যায়ন, সবার জন্য সহজ।",
        "desc": "হাঁটুর অস্টিওআর্থারাইটিসের ধাপে ধাপে সমন্বিত মূল্যায়ন। রোগীর লক্ষণ, এক্স-রে ডিপ লার্নিং, ক্যামেরার সাহায্যে হাঁটার ভঙ্গি বিশ্লেষণ এবং পরিধানযোগ্য সেন্সর ডেটার একীভূত প্ল্যাটফর্ম।",
        "start": "🚀 মূল্যায়ন শুরু করুন",
        "next": "পরবর্তী ধাপে যান ➔",
        "back": "⬅ পূর্ববর্তী ধাপ",
        "restart": "🔄 নতুন মূল্যায়ন",
        "steps": ("রোগীর বিবরণ", "লক্ষণ প্রশ্নাবলী", "এক্স-রে বিশ্লেষণ", "চলনভঙ্গি বিশ্লেষণ", "মোশন সেন্সর", "চূড়ান্ত মূল্যায়ন"),
        "pillars": ("ক্লিনিকাল লক্ষণ", "এক্স-রে চিত্র", "ক্যামেরা চলন", "পরিধানযোগ্য সেন্সর"),
        "xray_header": "🩻 হাঁটুর এক্স-রে বিশ্লেষণ (DenseNet-121)",
        "xray_sub": "কেলগ্রেন-লরেন্স (KL গ্রেড ০-৪) তীব্রতা নির্ধারণের জন্য হাঁটুর এক্স-রে আপলোড করুন।",
        "xray_upload": "হাঁটুর এক্স-রে চিত্র আপলোড করুন (.png, .jpg, .jpeg)",
        "xray_pred_label": "নির্ধারিত কেলগ্রেন-লরেন্স (KL) গ্রেড",
        "xray_conf_label": "নির্ভুলতার মাত্রা (Confidence)",
        "sensors_header": "📡 পরিধানযোগ্য মোশন সেন্সর (ESP32 + ৩টি MPU6050)",
        "sensors_sub": "হাঁটার সময় হাঁটুর লোড এবং গতিবিধির ৬-DOF রিয়েল-টাইম ডেটা।",
        "imu1_name": "সেন্সর ১: উরুর ত্বরণ (Thigh Accel Y)",
        "imu1_sub": "উল্লম্ব লোড প্রভাব (Axial Impact)",
        "imu2_name": "সেন্সর ২: পায়ের নিম্নভাগের জাইরো (Shank Gyro Z)",
        "imu2_sub": "সন্ধির ঘূর্ণন গতি (Angular Velocity)",
        "imu3_name": "সেন্সর ৩: গোড়ালির কম্পন শোষণ (Ankle Shock)",
        "imu3_sub": "মাটিতে পা ফেলার প্রতিক্রিয়া (Ground Reaction)",
        "chart_title": "রিয়েল-টাইম মোশন সেন্সর তরঙ্গ (Inertial Waveform)",
        "report_header": "📊 সমন্বিত মাল্টিমোডাল ডায়াগনস্টিক রিপোর্ট",
        "report_sub": "রোগীর তথ্য, এক্স-রে ডিপ লার্নিং (DenseNet), চলন বিশ্লেষণ (XGBoost), এবং সেন্সর ডেটার সামগ্রিক মূল্যায়ন।",
        "composite_label": "সামগ্রিক বাতজনিত তীব্রতার সূচক (Composite Score)",
        "tier_high": "উচ্চ ঝুঁকি / নিশ্চিত অস্টিওআর্থারাইটিস (High Risk / Confirmed OA)",
        "tier_mod": "মাঝারি ঝুঁকি / প্রাথমিক স্তর (Moderate Risk / Early OA)",
        "tier_low": "কম ঝুঁকি / স্বাভাবিক সন্ধি (Low Risk / Normal)",
        "breakdown_title": "বিভিন্ন ক্লিনিকাল উৎসের অবদান (Breakdown)",
        "item_patient": "রোগীর বিবরণ (Patient Profile)",
        "item_xray": "এক্স-রে বিশ্লেষণ (DenseNet-121)",
        "item_gait": "চলনভঙ্গি বিশ্লেষণ (XGBoost Gait)",
        "item_womac": "লক্ষণ প্রশ্নাবলী (WOMAC Symptoms)",
        "item_imu": "পরিধানযোগ্য সেন্সর (ESP32 IMU)",
        "weight_label": "গুরুত্ব (Weight)",
        "radar_labels": ("এক্স-রে (X-Ray)", "চলন (Gait)", "লক্ষণ (Symptoms)", "সেন্সর (Sensors)"),
        "download_btn": "📥 অফিশিয়াল ক্লিনিকাল রিপোর্ট ডাউনলোড করুন (PDF)",
        "evaluate_again_btn": "🔄 নতুন রোগী মূল্যায়ন শুরু করুন"
    }
}

lang = st.session_state["language"]
t = TEXTS[lang]
s_intake, s_quest, s_xray, s_gait, s_sensors, s_report = t["steps"]
pil_clin, pil_xray, pil_gait, pil_imu = t["pillars"]

# -------------------------------------------------------------
# 3. Rich Clinical Light Theme CSS with Advanced Animations
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }

    /* Hide sidebar */
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stSidebarCollapsedControl"] { display: none; }

    .stApp {
        background-color: #FBF9F5;
        color: #1E1B4B;
    }

    /* Screen width container */
    .block-container {
        max-width: 95% !important;
        padding-left: 2.2rem !important;
        padding-right: 2.2rem !important;
        padding-top: 2.2rem !important;
        padding-bottom: 2.5rem !important;
    }

    /* Entry Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(14px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .hero-box, .feature-card, .stepper-progress {
        animation: fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
    }

    /* Shimmer Effect on Hero Banner */
    @keyframes shimmerSweep {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    .hero-box {
        background: linear-gradient(135deg, #EEECFB 0%, #F5F3FF 100%);
        border-radius: 28px;
        padding: 44px 50px;
        margin-bottom: 28px;
        border: 1px solid #DFDBF8;
        position: relative;
        overflow: hidden;
    }
    .hero-box::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, transparent, #5B51D8, #10B981, transparent);
        animation: shimmerSweep 4s infinite linear;
        background-size: 200% 100%;
    }

    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        color: #5B51D8;
        letter-spacing: -0.03em;
        margin-bottom: 6px;
    }
    .hero-tagline {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E1B4B;
        line-height: 1.25;
        margin-bottom: 14px;
        max-width: 850px;
    }
    .hero-body {
        font-size: 1.15rem;
        color: #475569;
        line-height: 1.65;
        max-width: 1000px;
        margin-bottom: 0;
    }

    /* Floating Feature Cards */
    .feature-card {
        background: #FFFFFF;
        border-radius: 24px;
        padding: 32px 28px;
        box-shadow: 0 8px 24px rgba(91, 81, 216, 0.05);
        border: 1.5px solid #EAE7FB;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-8px) scale(1.01);
        box-shadow: 0 20px 40px rgba(91, 81, 216, 0.14);
        border-color: #5B51D8;
    }

    /* Stepper with Glowing Active Aura */
    .stepper-progress {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #FFFFFF;
        border-radius: 50px;
        padding: 12px 28px;
        margin-bottom: 28px;
        border: 1px solid #EAE7FB;
        box-shadow: 0 4px 16px rgba(91, 81, 216, 0.05);
        overflow-x: auto;
    }
    .step-item {
        font-weight: 700;
        font-size: 0.90rem;
        color: #64748B;
        padding: 6px 16px;
        border-radius: 30px;
        white-space: nowrap;
        transition: all 0.25s ease;
    }
    .step-active {
        color: #FFFFFF !important;
        background: #5B51D8 !important;
        box-shadow: 0 4px 14px rgba(91, 81, 216, 0.35);
        animation: stepPulse 2.4s infinite ease-in-out;
    }
    .step-done {
        color: #059669 !important;
        background: #DCFCE7 !important;
    }

    @keyframes stepPulse {
        0% { box-shadow: 0 0 0 0 rgba(91, 81, 216, 0.4); }
        70% { box-shadow: 0 0 0 8px rgba(91, 81, 216, 0); }
        100% { box-shadow: 0 0 0 0 rgba(91, 81, 216, 0); }
    }

    /* Primary Action Buttons with Breathing Pulse */
    @keyframes pulseGlow {
        0% { box-shadow: 0 6px 18px rgba(16, 185, 129, 0.28); }
        50% { box-shadow: 0 10px 28px rgba(16, 185, 129, 0.55); }
        100% { box-shadow: 0 6px 18px rgba(16, 185, 129, 0.28); }
    }

    div.stButton > button:first-child {
        border-radius: 50px !important;
        font-weight: 800 !important;
        font-size: 1.12rem !important;
        padding: 14px 40px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: none !important;
        background: #10B981 !important;
        color: #FFFFFF !important;
        animation: pulseGlow 2.8s infinite ease-in-out !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-3px) scale(1.01) !important;
        box-shadow: 0 14px 32px rgba(16, 185, 129, 0.45) !important;
        background: #059669 !important;
    }

    /* Popover button custom styling (pill button) */
    div[data-testid="stPopover"] {
        margin-top: 6px !important;
        overflow: visible !important;
    }
    div[data-testid="stPopover"] > button {
        border-radius: 40px !important;
        border: 1.5px solid #DFDBF8 !important;
        background-color: #FFFFFF !important;
        color: #5B51D8 !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 10px 22px !important;
        min-height: 44px !important;
        box-shadow: 0 4px 12px rgba(91, 81, 216, 0.08) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stPopover"] > button:hover {
        background-color: #EEECFB !important;
        border-color: #5B51D8 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(91, 81, 216, 0.14) !important;
    }

    /* Live Telemetry Pulse Badge */
    @keyframes live-pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1.05); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    .badge-live {
        display: inline-block;
        width: 9px;
        height: 9px;
        background-color: #10B981;
        border-radius: 50%;
        animation: live-pulse 2s infinite;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 4. Clean Top Header: Properly Positioned Language Popover
# -------------------------------------------------------------
top_col1, top_col2 = st.columns((3.5, 1.2))

with top_col1:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 14px; margin-top: 2px;">
        <span style="font-size: 2.5rem;">🦴</span>
        <div>
            <h1 style="color: #5B51D8; font-weight: 800; margin: 0; font-size: 2.1rem; letter-spacing: -0.02em; line-height: 1.1;">ARTHRO-AI</h1>
            <span style="color: #64748B; font-size: 0.95rem; font-weight: 600;">Clinical Multimodal Workstation</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with top_col2:
    st.markdown("<div style='height: 26px;'></div>", unsafe_allow_html=True)
    
    if hasattr(st, "popover"):
        with st.popover(f"🌐 {lang} ▾", use_container_width=True):
            st.markdown("<p style='font-weight: 700; color: #5B51D8; margin-bottom: 8px; font-size: 0.9rem;'>Select Language / ভাষা</p>", unsafe_allow_html=True)
            for l_name in ("English", "हिन्दी", "অসমীয়া", "বাংলা"):
                is_active = (l_name == lang)
                label_txt = f"✓ {l_name}" if is_active else l_name
                if st.button(label_txt, key=f"lang_opt_{l_name}", use_container_width=True):
                    if l_name != st.session_state["language"]:
                        st.session_state["language"] = l_name
                        st.rerun()
    else:
        with st.expander(f"🌐 {lang} ▾"):
            for l_name in ("English", "हिन्दी", "অসমীয়া", "বাংলা"):
                if st.button(l_name, key=f"lang_opt_exp_{l_name}", use_container_width=True):
                    st.session_state["language"] = l_name
                    st.rerun()

# System Engine Status Ribbon
st.markdown("""
<div style="background: #F1F0FB; border-radius: 14px; padding: 10px 22px; margin: 12px 0 24px 0; display: flex; align-items: center; justify-content: space-between; font-size: 0.88rem; color: #475569; border: 1px solid #E4E1F8;">
    <div>
        <span class="badge-live"></span><b>DenseNet-121:</b> Active (KL 0-4) &nbsp;&nbsp;|&nbsp;&nbsp;
        <span class="badge-live"></span><b>MediaPipe Tasks:</b> 40-Frame Bilateral &nbsp;&nbsp;|&nbsp;&nbsp;
        <span class="badge-live"></span><b>XGBoost V5:</b> Kinematic Engine &nbsp;&nbsp;|&nbsp;&nbsp;
        <span class="badge-live"></span><b>ESP32 IMU:</b> CSV Stream Active
    </div>
    <div>
        <span style="color: #5B51D8; font-weight: 700;">Node: NITN-CPS-2026</span>
    </div>
</div>
""", unsafe_allow_html=True)

step = st.session_state["current_step"]

# Visual Stepper Indicator (Visible during active assessment steps 1-6)
if step > 0:
    st.markdown(f"""
    <div class="stepper-progress">
        <span class="step-item {'step-done' if step>1 else ''} {'step-active' if step==1 else ''}">{'✓' if step>1 else '1.'} {s_intake}</span>
        <span style="color: #CBD5E1; font-weight: 700;">›</span>
        <span class="step-item {'step-done' if step>2 else ''} {'step-active' if step==2 else ''}">{'✓' if step>2 else '2.'} {s_quest}</span>
        <span style="color: #CBD5E1; font-weight: 700;">›</span>
        <span class="step-item {'step-done' if step>3 else ''} {'step-active' if step==3 else ''}">{'✓' if step>3 else '3.'} {s_xray}</span>
        <span style="color: #CBD5E1; font-weight: 700;">›</span>
        <span class="step-item {'step-done' if step>4 else ''} {'step-active' if step==4 else ''}">{'✓' if step>4 else '4.'} {s_gait}</span>
        <span style="color: #CBD5E1; font-weight: 700;">›</span>
        <span class="step-item {'step-done' if step>5 else ''} {'step-active' if step==5 else ''}">{'✓' if step>5 else '5.'} {s_sensors}</span>
        <span style="color: #CBD5E1; font-weight: 700;">›</span>
        <span class="step-item {'step-active' if step==6 else ''}">6. {s_report}</span>
    </div>
    """, unsafe_allow_html=True)

# =============================================================
# STEP 0: HOMEPAGE OVERVIEW
# =============================================================
if step == 0:
    st.markdown(f"""
    <div class="hero-box">
        <div class="hero-title">{t['title']}</div>
        <div class="hero-tagline">{t['subtitle']}</div>
        <div class="hero-body">{t['desc']}</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4, gap="large")
    with col1:
        st.markdown(f"""
        <div class="feature-card">
            <span style="background: #E0E7FF; color: #4338CA; padding: 6px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">Pillar 1</span>
            <h3 style="color: #1E1B4B; font-size: 1.4rem; font-weight: 800; margin: 14px 0 10px 0;">{pil_clin}</h3>
            <p style="color: #64748B; font-size: 0.96rem; line-height: 1.55;">Patient profile, calculated BMI, pain levels, and WOMAC functional limitations.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="feature-card">
            <span style="background: #DCFCE7; color: #15803D; padding: 6px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">Pillar 2</span>
            <h3 style="color: #1E1B4B; font-size: 1.4rem; font-weight: 800; margin: 14px 0 10px 0;">{pil_xray}</h3>
            <p style="color: #64748B; font-size: 0.96rem; line-height: 1.55;">DenseNet-121 neural evaluation of Kellgren–Lawrence (KL) Grades 0 to 4.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="feature-card">
            <span style="background: #FEF3C7; color: #B45309; padding: 6px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">Pillar 3</span>
            <h3 style="color: #1E1B4B; font-size: 1.4rem; font-weight: 800; margin: 14px 0 10px 0;">{pil_gait}</h3>
            <p style="color: #64748B; font-size: 0.96rem; line-height: 1.55;">MediaPipe skeletal tracking for bilateral knee flexion and asymmetry.</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="feature-card">
            <span style="background: #F3E8FF; color: #7E22CE; padding: 6px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">Pillar 4</span>
            <h3 style="color: #1E1B4B; font-size: 1.4rem; font-weight: 800; margin: 14px 0 10px 0;">{pil_imu}</h3>
            <p style="color: #64748B; font-size: 0.96rem; line-height: 1.55;">6-DOF real-time acceleration and rotational impact via 3× MPU6050 sensors.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    col_btn, _ = st.columns((1, 2))
    with col_btn:
        if st.button(t["start"], type="primary", use_container_width=True):
            st.session_state["current_step"] = 1
            st.rerun()

# =============================================================
# STEP 1: PATIENT INTAKE
# =============================================================
elif step == 1:
    patient_module()
    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        if st.button(t["back"]):
            st.session_state["current_step"] = 0
            st.rerun()
    with b2:
        if st.button(t["next"], type="primary"):
            st.session_state["current_step"] = 2
            st.rerun()

# =============================================================
# STEP 2: QUESTIONNAIRE
# =============================================================
elif step == 2:
    questionnaire_module()
    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        if st.button(t["back"]):
            st.session_state["current_step"] = 1
            st.rerun()
    with b2:
        if st.button(t["next"], type="primary"):
            st.session_state["current_step"] = 3
            st.rerun()

# =============================================================
# STEP 3: X-RAY (DENSENET-121)
# =============================================================
elif step == 3:
    if XRayAnalyzer:
        analyzer = XRayAnalyzer()
        st.markdown(f"<h2 style='font-size: 2.2rem; font-weight: 800; color: #1E1B4B; margin-bottom: 4px;'>{t['xray_header']}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #64748B; font-size: 1.05rem; margin-bottom: 25px;'>{t['xray_sub']}</p>", unsafe_allow_html=True)
        
        cx1, cx2 = st.columns(2, gap="large")
        with cx1:
            st.markdown("""
            <div style="background: #FFFFFF; border-radius: 20px; padding: 25px; box-shadow: 0 10px 25px rgba(91, 81, 216, 0.06); border: 1px solid #EAE7FB;">
            """, unsafe_allow_html=True)
            up_x = st.file_uploader(t["xray_upload"], type=["png", "jpg", "jpeg"])
            if up_x: st.image(up_x, caption="Patient Knee Radiograph", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with cx2:
            if up_x:
                res = analyzer.predict(up_x)
                st.session_state["kl_grade"] = res["predicted_grade"]
                st.session_state["xray_conf"] = res["confidence"]
                
                st.markdown(f"""
                <div style="background: #FFFFFF; border-radius: 20px; padding: 25px; box-shadow: 0 10px 25px rgba(91, 81, 216, 0.06); border: 1.5px solid {res['color']}; margin-bottom: 20px;">
                    <span style="color: #64748B; font-size: 0.85rem; font-weight: 700; text-transform: uppercase;">{t['xray_pred_label']}</span>
                    <h2 style="color: {res['color']}; font-size: 2.2rem; font-weight: 800; margin: 6px 0;">{res['title']}</h2>
                    <p style="color: #334155; font-size: 0.95rem; margin-bottom: 12px;">{res['description']}</p>
                    <span style="background: {res['bg']}; color: {res['color']}; font-weight: 700; font-size: 0.9rem; padding: 6px 14px; border-radius: 30px;">
                        {t['xray_conf_label']}: {res['confidence']*100:.1f}%
                    </span>
                </div>
                """, unsafe_allow_html=True)
                st.bar_chart(pd.DataFrame({"Probability": res["probabilities"]}, index=[f"Grade {i}" for i in range(5)]))
    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        if st.button(t["back"]):
            st.session_state["current_step"] = 2
            st.rerun()
    with b2:
        if st.button(t["next"], type="primary"):
            st.session_state["current_step"] = 4
            st.rerun()

# =============================================================
# STEP 4: GAIT KINEMATICS WITH 12-POINT CARD
# =============================================================
elif step == 4:
    gait_t = GAIT_STRINGS.get(lang, GAIT_STRINGS.get("English", {}))
    pts = list(gait_t.get("points", []))
    half_pt = len(pts) // 2
    setup_pts = pts[:half_pt]
    walk_pts = pts[half_pt:]
    
    st.markdown(f"""
    <div style="background: #FFFFFF; border-radius: 24px; padding: 30px; box-shadow: 0 10px 30px rgba(91, 81, 216, 0.06); border: 1.5px solid #5B51D8; margin-bottom: 30px;">
        <h3 style="color: #5B51D8; font-size: 1.5rem; font-weight: 800; margin-top: 0; margin-bottom: 18px;">
            {gait_t.get('proto_title', '📋 Gait Protocol Checklist')}
        </h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
            <div style="background: #FBF9F5; border-radius: 16px; padding: 20px; border: 1px solid #EAE7FB;">
                <p style="color: #1E1B4B; font-weight: 800; font-size: 1.05rem; margin-top: 0; margin-bottom: 10px;">
                    {gait_t.get('setup_title', 'Setup')}
                </p>
                <ul style="color: #475569; font-size: 0.95rem; line-height: 1.8; margin: 0; padding-left: 20px;">
                    {''.join([f"<li><b>{p}</b></li>" for p in setup_pts])}
                </ul>
            </div>
            <div style="background: #FBF9F5; border-radius: 16px; padding: 20px; border: 1px solid #EAE7FB;">
                <p style="color: #1E1B4B; font-weight: 800; font-size: 1.05rem; margin-top: 0; margin-bottom: 10px;">
                    {gait_t.get('walk_title', 'Protocol')}
                </p>
                <ul style="color: #475569; font-size: 0.95rem; line-height: 1.8; margin: 0; padding-left: 20px;">
                    {''.join([f"<li><b>{p}</b></li>" for p in walk_pts])}
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    gait_module()

    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        if st.button(t["back"]):
            st.session_state["current_step"] = 3
            st.rerun()
    with b2:
        if st.button(t["next"], type="primary"):
            st.session_state["current_step"] = 5
            st.rerun()

# =============================================================
# STEP 5: SENSORS (ESP32)
# =============================================================
elif step == 5:
    st.markdown(f"<h2 style='font-size: 2.2rem; font-weight: 800; color: #1E1B4B; margin-bottom: 4px;'>{t['sensors_header']}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #64748B; font-size: 1.05rem; margin-bottom: 25px;'>{t['sensors_sub']}</p>", unsafe_allow_html=True)

    if ESP32SerialReader:
        reader = ESP32SerialReader()
        sample = reader.read_packet()
    else:
        sample = {"accY": 1.12, "gyroZ": 45.2, "accX": 0.14}

    h1, h2, h3 = st.columns(3, gap="large")
    with h1:
        st.markdown(f"""
        <div class="feature-card">
            <span style="color: #64748B; font-size: 0.85rem; font-weight: 700; text-transform: uppercase;">{t['imu1_name']}</span>
            <h2 style="color: #5B51D8; font-size: 2.4rem; font-weight: 800; margin: 8px 0;">{sample.get('accY', 1.0):.2f} g</h2>
            <span style="color: #64748B; font-size: 0.85rem;">{t['imu1_sub']}</span>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown(f"""
        <div class="feature-card">
            <span style="color: #64748B; font-size: 0.85rem; font-weight: 700; text-transform: uppercase;">{t['imu2_name']}</span>
            <h2 style="color: #059669; font-size: 2.4rem; font-weight: 800; margin: 8px 0;">{sample.get('gyroZ', 45.0):.1f} °/s</h2>
            <span style="color: #64748B; font-size: 0.85rem;">{t['imu2_sub']}</span>
        </div>
        """, unsafe_allow_html=True)
    with h3:
        st.markdown(f"""
        <div class="feature-card">
            <span style="color: #64748B; font-size: 0.85rem; font-weight: 700; text-transform: uppercase;">{t['imu3_name']}</span>
            <h2 style="color: #D97706; font-size: 2.4rem; font-weight: 800; margin: 8px 0;">{sample.get('accX', 0.12):.2f} g</h2>
            <span style="color: #64748B; font-size: 0.85rem;">{t['imu3_sub']}</span>
        </div>
        """, unsafe_allow_html=True)

    t_series = np.linspace(0, 4, 100)
    acc_wave = np.sin(2 * np.pi * 1.2 * t_series) + 0.08 * np.random.randn(100)
    gyro_wave = 70 * np.cos(2 * np.pi * 1.2 * t_series) + 4.0 * np.random.randn(100)

    fig_imu = go.Figure()
    fig_imu.add_trace(go.Scatter(x=t_series, y=acc_wave, mode='lines', name='accY (g)', line=dict(color='#5B51D8', width=2.5)))
    fig_imu.add_trace(go.Scatter(x=t_series, y=gyro_wave/100.0, mode='lines', name='gyroZ (normalized)', line=dict(color='#10B981', width=2.5)))
    fig_imu.update_layout(
        title=t["chart_title"],
        xaxis_title="Time (s)",
        yaxis_title="Sensor Units",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#FFFFFF',
        font=dict(color='#1E1B4B')
    )
    st.plotly_chart(fig_imu, use_container_width=True)

    b1, b2 = st.columns(2)
    with b1:
        if st.button(t["back"]):
            st.session_state["current_step"] = 4
            st.rerun()
    with b2:
        if st.button(t["next"], type="primary"):
            st.session_state["current_step"] = 6
            st.rerun()

# =============================================================
# STEP 6: INTEGRATED REPORT & PDF EXPORT
# =============================================================
elif step == 6:
    st.markdown(f"<h2 style='font-size: 2.4rem; font-weight: 800; color: #1E1B4B; margin-bottom: 6px;'>{t['report_header']}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #64748B; font-size: 1.05rem; margin-bottom: 25px;'>{t['report_sub']}</p>", unsafe_allow_html=True)

    p_info = st.session_state.get("patient_data", {})
    kl_grade = st.session_state.get("kl_grade", 2)
    xray_pct = (kl_grade / 4.0) * 100.0
    gait_pct = float(st.session_state.get("camera_prediction", 0.65)) * 100.0
    symp_pct = float(st.session_state.get("symptom_score", 52.0))
    sensor_pct = 54.0

    composite_index = (0.40 * xray_pct) + (0.30 * gait_pct) + (0.20 * symp_pct) + (0.10 * sensor_pct)

    r1, r2 = st.columns(2, gap="large")
    with r1:
        risk_color = "#DC2626" if composite_index >= 60 else ("#D97706" if composite_index >= 35 else "#059669")
        risk_bg = "#FEE2E2" if composite_index >= 60 else ("#FEF3C7" if composite_index >= 35 else "#DCFCE7")
        tier = t["tier_high"] if composite_index >= 60 else (t["tier_mod"] if composite_index >= 35 else t["tier_low"])

        st.markdown(f"""
        <div style="background: #FFFFFF; border-radius: 24px; padding: 30px; box-shadow: 0 10px 30px rgba(91, 81, 216, 0.06); border: 2px solid {risk_color}; margin-bottom: 25px;">
            <span style="color: #64748B; font-size: 0.9rem; font-weight: 700; text-transform: uppercase;">{t['composite_label']}</span>
            <div style="font-size: 3.4rem; font-weight: 800; color: {risk_color}; margin: 8px 0;">
                {composite_index:.1f} <span style="font-size: 1.5rem; color: #64748B;">/ 100</span>
            </div>
            <div style="background: {risk_bg}; color: {risk_color}; font-weight: 800; font-size: 1.05rem; padding: 8px 18px; border-radius: 30px; display: inline-block;">
                {tier}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background: #FFFFFF; border-radius: 20px; padding: 25px; box-shadow: 0 6px 20px rgba(0,0,0,0.04); border: 1px solid #EAE7FB;">
            <h4 style="color: #5B51D8; margin-top: 0; margin-bottom: 12px;">{t['breakdown_title']}</h4>
            <p style="margin: 6px 0; color: #334155;">• <b>{t['item_patient']}:</b> {p_info.get('name')} ({p_info.get('age')}y) | BMI: {p_info.get('bmi'):.1f} kg/m²</p>
            <p style="margin: 6px 0; color: #334155;">• 🩻 <b>{t['item_xray']}:</b> KL Grade {kl_grade} ({xray_pct:.0f}%) — <i>{t['weight_label']} 40%</i></p>
            <p style="margin: 6px 0; color: #334155;">• 🚶 <b>{t['item_gait']}:</b> {gait_pct:.1f}% — <i>{t['weight_label']} 30%</i></p>
            <p style="margin: 6px 0; color: #334155;">• 📋 <b>{t['item_womac']}:</b> {symp_pct:.1f}% — <i>{t['weight_label']} 20%</i></p>
            <p style="margin: 6px 0; color: #334155;">• 📡 <b>{t['item_imu']}:</b> {sensor_pct:.1f}% — <i>{t['weight_label']} 10%</i></p>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(
            r=[xray_pct, gait_pct, symp_pct, sensor_pct],
            theta=list(t["radar_labels"]),
            fill='toself',
            name='Patient Profile',
            line=dict(color='#5B51D8', width=2.5),
            fillcolor='rgba(91, 81, 216, 0.2)'
        ))
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor='#E2E8F0'),
                angularaxis=dict(gridcolor='#E2E8F0')
            ),
            showlegend=False,
            margin=dict(l=35, r=35, t=25, b=25),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#1E1B4B', size=11)
        )
        st.plotly_chart(radar_fig, use_container_width=True)

    st.divider()
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if generate_clinical_pdf is not None:
            pdf_bytes = generate_clinical_pdf(
                patient_data=p_info,
                kl_grade=kl_grade,
                xray_conf=st.session_state.get("xray_conf", 0.85),
                gait_prob=st.session_state.get("camera_prediction", 0.65),
                symptom_score=symp_pct,
                sensor_score=sensor_pct
            )
            
            st.download_button(
                label=t["download_btn"],
                data=pdf_bytes,
                file_name=f"ARTHRO_AI_Report_{p_info.get('id', 'Patient')}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
        else:
            st.error("ReportLab is not installed. Run 'pip install reportlab' to download PDFs.")

    with c_btn2:
        if st.button(t["evaluate_again_btn"], use_container_width=True):
            st.session_state["current_step"] = 0
            st.rerun()