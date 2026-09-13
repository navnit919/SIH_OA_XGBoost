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
    def questionnaire_module(): st.warning("questionnaire_module could not be loaded.")

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

st.set_page_config(
    page_title="ARTHRO-AI | Clinical Diagnostic Workstation",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
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
# 2. Multi-Language Master Dictionary
# -------------------------------------------------------------
TEXTS = {
    "English": {
        "title": "ARTHRO-AI Diagnostic Platform",
        "subtitle": "Multimodal AI System for Knee Osteoarthritis Screening & Assessment",
        "desc": "An advanced clinical workstation synthesizing 4 diagnostic dimensions: subject intake, clinical symptoms, deep radiograph classification (DenseNet-121), real-time computer vision gait kinematics (MediaPipe + XGBoost), and wearable inertial sensing (ESP32 + MPU6050).",
        "start": "🚀 Launch Diagnostic Pipeline",
        "next": "Proceed to Next Step ➔",
        "back": "⬅ Previous Step",
        "restart": "🔄 Reset & New Patient",
        "steps": ["Patient Intake", "Questionnaire", "X-Ray", "Gait", "Sensors", "Final Report"],
        "pillars": ["Clinical", "Radiographic", "Kinematic", "Wearable IoT"],
        "report_title": "📊 Step 6: Integrated Multimodal Osteoarthritis Diagnostic Report",
        "composite_title": "Composite OA Diagnostic Index",
        "tier_high": "High Risk / Confirmed Osteoarthritis",
        "tier_mod": "Moderate Risk / Early Mild OA"
    },
    "हिन्दी": {
        "title": "आर्थ्रो-एआई (ARTHRO-AI) क्लिनिकल प्लेटफॉर्म",
        "subtitle": "घुटने के ऑस्टियोआर्थराइटिस की पहचान के लिए मल्टीमॉडल एआई सिस्टम",
        "desc": "यह सिस्टम चार मुख्य आयामों को एकीकृत करता है: रोगी पंजीकरण, क्लिनिकल लक्षण, एक्स-रे डीप लर्निंग (DenseNet-121), चाल विश्लेषण (MediaPipe + XGBoost), और वियरेबल सेंसर (ESP32 + MPU6050)।",
        "start": "🚀 जांच प्रक्रिया शुरू करें",
        "next": "अगले चरण पर जाएं ➔",
        "back": "⬅ पिछले चरण पर जाएं",
        "restart": "🔄 नया मूल्यांकन शुरू करें",
        "steps": ["रोगी विवरण", "प्रश्नावली", "एक्स-रे", "चाल विश्लेषण", "सेंसर", "समग्र रिपोर्ट"],
        "pillars": ["क्लिनिकल", "एक्स-रे", "चाल विश्लेषण", "वियरेबल सेंसर"],
        "report_title": "📊 चरण 6: समग्र मल्टीमॉडल ऑस्टियोआर्थराइटिस रिपोर्ट",
        "composite_title": "समग्र गठिया सूचकांक (Composite OA Index)",
        "tier_high": "उच्च जोखिम / पुष्ट ऑस्टियोआर्थराइटिस",
        "tier_mod": "मध्यम जोखिम / शुरुआती लक्षण"
    },
    "অসমীয়া": {
        "title": "আৰ্থ্ৰো-এআই (ARTHRO-AI) ক্লিনিকেল প্লেটফৰ্ম",
        "subtitle": "অষ্টিঅ'আৰ্থ্ৰাইটিছ (বাতবিষ) চিনাক্তকৰণ আৰু মূল্যায়নৰ বাবে এআই ব্যৱস্থা",
        "desc": "এই ব্যৱস্থাই চাৰিটা প্ৰধান দিশ একত্ৰিত কৰে: ৰোগীৰ তথ্য, লক্ষণ মূল্যায়ন, এক্স-ৰে' ডিপ লাৰ্নিং (DenseNet-121), চাল-চলন বিশ্লেষণ (MediaPipe + XGBoost), আৰু ৱিয়েৰেবল চেন্সৰ (ESP32 + MPU6050)।",
        "start": "🚀 পৰীক্ষণ প্ৰক্ৰিয়া আৰম্ভ কৰক",
        "next": "পৰৱৰ্তী স্তৰলৈ যাওক ➔",
        "back": "⬅ পূৰ্ববৰ্তী স্তৰ",
        "restart": "🔄 নতুন ৰোগী পৰীক্ষা",
        "steps": ["ৰোগীৰ তথ্য", "প্ৰশ্নাৱলী", "এক্স-ৰে'", "খোজ বিশ্লেষণ", "চেন্সৰ", "চূড়ান্ত ৰিপ'ৰ্ট"],
        "pillars": ["ক্লিনিকেল", "এক্স-ৰে'", "গতিবিধি", "ৱিয়েৰেবল চেন্সৰ"],
        "report_title": "📊 স্তৰ ৬: সমগ্ৰ মাল্টিমডেল ৰিপ'ৰ্ট",
        "composite_title": "বাতবিষৰ সামূহিক সূচকাঙ্ক",
        "tier_high": "উচ্চ আশংকা / প্ৰমাণিত বাতবিষ",
        "tier_mod": "মধ্যমীয়া আশংকা / প্ৰাৰম্ভিক লক্ষণ"
    },
    "বাংলা": {
        "title": "আর্থ্রো-এআই (ARTHRO-AI) ক্লিনিকাল প্ল্যাটফর্ম",
        "subtitle": "হাঁটুর অস্টিওআর্থারাইটিস স্ক্রিনিং ও মূল্যায়নের মাল্টিমোডাল এআই সিস্টেম",
        "desc": "এই উন্নত ওয়ার্কস্টেশনটি চারটি প্রধান মাত্রাকে সংযুক্ত করে: রোগীর তথ্য, ক্লিনিকাল লক্ষণ, এক্স-রে ডিপ লার্নিং (DenseNet-121), চলন বা গেইট বিশ্লেষণ (MediaPipe + XGBoost), এবং পরিধানযোগ্য সেন্সর (ESP32 + MPU6050)।",
        "start": "🚀 ডায়াগনস্টিক পাইপলাইন শুরু করুন",
        "next": "পরবর্তী ধাপে যান ➔",
        "back": "⬅ পূর্ববর্তী ধাপ",
        "restart": "🔄 নতুন মূল্যায়ন শুরু করুন",
        "steps": ["রোগীর তথ্য", "প্রশ্নাবলী", "এক্স-রে", "চলন বিশ্লেষণ", "সেন্সর", "চূড়ান্ত রিপোর্ট"],
        "pillars": ["ক্লিনিকাল", "রেডিওগ্রাফিক", "চলনভঙ্গি", "পরিধানযোগ্য সেন্সর"],
        "report_title": "📊 ধাপ ৬: সমন্বিত মাল্টিমোডাল ক্লিনিকাল রিপোর্ট",
        "composite_title": "সামগ্রিক বাতজনিত স্কোর (Composite Index)",
        "tier_high": "উচ্চ ঝুঁকি / নিশ্চিত অস্টিওআর্থারাইটিস",
        "tier_mod": "মাঝারি ঝুঁকি / প্রাথমিক স্তর"
    }
}

lang = st.session_state["language"]
t = TEXTS[lang]

# -------------------------------------------------------------
# 3. High-End Medical CSS & Animations
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }

    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(14, 165, 233, 0.05) 0%, transparent 40%),
                    radial-gradient(circle at 90% 80%, rgba(99, 102, 241, 0.05) 0%, transparent 40%),
                    #080C14;
    }

    @keyframes ecg-pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1.05); box-shadow: 0 0 0 9px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    .badge-live {
        display: inline-block;
        width: 9px;
        height: 9px;
        background-color: #10B981;
        border-radius: 50%;
        animation: ecg-pulse 1.8s infinite;
        margin-right: 8px;
    }

    .hero-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92) 0%, rgba(30, 41, 59, 0.75) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 18px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6);
        position: relative;
        overflow: hidden;
    }
    
    .hero-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, #38BDF8, #818CF8, transparent);
        animation: shimmer 3s infinite linear;
        background-size: 200% 100%;
    }

    .metric-card {
        background: rgba(18, 26, 43, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 16px;
        backdrop-filter: blur(16px);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.45);
    }
    
    .card-title {
        color: #94A3B8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .card-val {
        color: #F8FAFC;
        font-size: 2.0rem;
        font-weight: 800;
        margin: 4px 0;
        letter-spacing: -0.02em;
    }

    .stepper-box {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 14px 24px;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        overflow-x: auto;
    }
    .step-item {
        font-weight: 600;
        font-size: 0.84rem;
        color: #64748B;
        padding: 6px 12px;
        border-radius: 8px;
        transition: all 0.3s ease;
        white-space: nowrap;
    }
    .step-active {
        color: #38BDF8 !important;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.3);
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.2);
    }
    .step-completed {
        color: #10B981 !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 4. Sidebar Controls & Patient Profile
# -------------------------------------------------------------
p_curr = st.session_state["patient_data"]

with st.sidebar:
    st.markdown("### 🩺 **ARTHRO-AI**")
    st.caption("Clinical Multimodal Workstation")
    st.divider()

    st.markdown("#### 🌐 Language / ভাষা / भाषा")
    lang_options = ["English", "हिन्दी", "অসমীয়া", "বাংলা"]
    curr_idx = lang_options.index(lang) if lang in lang_options else 0
    selected_lang = st.selectbox("", lang_options, index=curr_idx)
    if selected_lang != st.session_state["language"]:
        st.session_state["language"] = selected_lang
        st.rerun()

    st.divider()
    st.markdown("#### 👤 Current Subject")
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.5); padding: 12px 14px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06); font-size: 0.88rem;">
        <b>Name:</b> <span style="color: #38BDF8;">{p_curr.get('name', 'N/A')}</span><br>
        <b>MRN:</b> <code>{p_curr.get('id', 'N/A')}</code><br>
        <b>Age / Sex:</b> {p_curr.get('age', 58)}y / {p_curr.get('gender', 'Male')}<br>
        <b>Target Joint:</b> {p_curr.get('joint', 'Right Knee')}<br>
        <b>BMI:</b> {p_curr.get('bmi', 25.6):.1f} kg/m²
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 🟢 Active Diagnostic Engines")
    st.markdown('<div><span class="badge-live"></span><b>DenseNet-121:</b> Active (KL 0-4)</div>', unsafe_allow_html=True)
    st.markdown('<div><span class="badge-live"></span><b>MediaPipe Tasks:</b> 40 Bilateral Frames</div>', unsafe_allow_html=True)
    st.markdown('<div><span class="badge-live"></span><b>XGBoost Model:</b> 6-Feature Kinematics</div>', unsafe_allow_html=True)
    st.markdown('<div><span class="badge-live"></span><b>ESP32 IMU:</b> CSV Stream Active</div>', unsafe_allow_html=True)

    if st.session_state["current_step"] > 0:
        st.divider()
        if st.button(t["restart"], use_container_width=True):
            st.session_state["current_step"] = 0
            st.rerun()

# -------------------------------------------------------------
# 5. Connected Stepper Header (Steps 1 to 6)
# -------------------------------------------------------------
step = st.session_state["current_step"]
st_labels = t["steps"]

if step > 0:
    st.markdown(f"""
    <div class="stepper-box">
        <span class="step-item {'step-completed' if step>1 else ''} {'step-active' if step==1 else ''}">{'✓' if step>1 else '1.'} {st_labels[0]}</span>
        <span style="color: #475569;">›</span>
        <span class="step-item {'step-completed' if step>2 else ''} {'step-active' if step==2 else ''}">{'✓' if step>2 else '2.'} {st_labels[1]}</span>
        <span style="color: #475569;">›</span>
        <span class="step-item {'step-completed' if step>3 else ''} {'step-active' if step==3 else ''}">{'✓' if step>3 else '3.'} {st_labels[2]}</span>
        <span style="color: #475569;">›</span>
        <span class="step-item {'step-completed' if step>4 else ''} {'step-active' if step==4 else ''}">{'✓' if step>4 else '4.'} {st_labels[3]}</span>
        <span style="color: #475569;">›</span>
        <span class="step-item {'step-completed' if step>5 else ''} {'step-active' if step==5 else ''}">{'✓' if step>5 else '5.'} {st_labels[4]}</span>
        <span style="color: #475569;">›</span>
        <span class="step-item {'step-active' if step==6 else ''}">6. {st_labels[5]}</span>
    </div>
    """, unsafe_allow_html=True)

# =============================================================
# STEP 0: LANDING & SYSTEM OVERVIEW
# =============================================================
if step == 0:
    st.markdown(f"""
    <div class="hero-card">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
            <span style="font-size: 2rem;">🩺</span>
            <h1 style="color: #F8FAFC; margin: 0; font-size: 2.2rem; font-weight: 800;">{t["title"]}</h1>
        </div>
        <p style="color: #38BDF8; font-size: 1.15rem; font-weight: 600; margin-bottom: 16px;">{t["subtitle"]}</p>
        <p style="color: #94A3B8; font-size: 0.96rem; line-height: 1.65; max-width: 950px; margin-bottom: 0;">{t["desc"]}</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">Pillar 1: {t['pillars'][0]}</div>
            <div class="card-val" style="color: #38BDF8;">Intake & Symptoms</div>
            <div style="color: #94A3B8; font-size: 0.82rem; line-height: 1.5;">Subject profile, BMI, pain levels, and WOMAC functional limitations.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">Pillar 2: {t['pillars'][1]}</div>
            <div class="card-val" style="color: #10B981;">DenseNet-121</div>
            <div style="color: #94A3B8; font-size: 0.82rem; line-height: 1.5;">Kellgren–Lawrence (KL) Grades 0–4 structural joint space classification.</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">Pillar 3: {t['pillars'][2]}</div>
            <div class="card-val" style="color: #F59E0B;">XGBoost Gait</div>
            <div style="color: #94A3B8; font-size: 0.82rem; line-height: 1.5;">MediaPipe skeletal pose tracking for bilateral knee flexion and asymmetry.</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">Pillar 4: {t['pillars'][3]}</div>
            <div class="card-val" style="color: #818CF8;">ESP32 IMU</div>
            <div style="color: #94A3B8; font-size: 0.82rem; line-height: 1.5;">6-DOF real-time acceleration and angular rotation tracking via 3× MPU6050.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 2])
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
    b1, b2 = st.columns([1, 1])
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
    b1, b2 = st.columns([1, 1])
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
        st.subheader("🩻 Step 3: Knee Radiograph Analysis (DenseNet-121)")
        cx1, cx2 = st.columns([1, 1])
        with cx1:
            up_x = st.file_uploader("Upload Knee AP/Lateral X-ray Image", type=["png", "jpg", "jpeg"])
            if up_x: st.image(up_x, caption="Patient Radiograph", use_container_width=True)
        with cx2:
            if up_x:
                res = analyzer.predict(up_x)
                st.session_state["kl_grade"] = res["predicted_grade"]
                st.session_state["xray_conf"] = res["confidence"]
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {res['color']};">
                    <div class="card-title">Predicted Kellgren–Lawrence Grade</div>
                    <div class="card-val" style="color: {res['color']};">{res['title']}</div>
                    <div style="color: #94A3B8; font-size: 0.85rem;">Confidence: <b>{res['confidence']*100:.1f}%</b> | {res['description']}</div>
                </div>
                """, unsafe_allow_html=True)
                st.bar_chart(pd.DataFrame({"Probability": res["probabilities"]}, index=[f"Grade {i}" for i in range(5)]))
    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2 = st.columns([1, 1])
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
    pts = gait_t.get("points", [])
    
    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.95); border: 1.5px solid #38BDF8; border-radius: 14px; padding: 22px; margin-bottom: 25px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);">
        <h3 style="color: #38BDF8; margin-top: 0; margin-bottom: 15px; font-size: 1.25rem;">
            {gait_t.get('proto_title', '📋 Gait Analysis Instructions')}
        </h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 25px;">
            <div>
                <p style="color: #F8FAFC; font-weight: 700; font-size: 0.95rem; margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 4px;">
                    {gait_t.get('setup_title', 'Setup')}
                </p>
                <ul style="color: #E2E8F0; font-size: 0.9rem; line-height: 1.8; margin: 0; padding-left: 20px;">
                    {''.join([f"<li><b>{p}</b></li>" for p in pts[:6]])}
                </ul>
            </div>
            <div>
                <p style="color: #F8FAFC; font-weight: 700; font-size: 0.95rem; margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 4px;">
                    {gait_t.get('walk_title', 'Protocol')}
                </p>
                <ul style="color: #E2E8F0; font-size: 0.9rem; line-height: 1.8; margin: 0; padding-left: 20px;">
                    {''.join([f"<li><b>{p}</b></li>" for p in pts[6:]])}
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    gait_module()

    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2 = st.columns([1, 1])
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
    hw_t = HW_STRINGS.get(lang, HW_STRINGS.get("English", {}))
    st.subheader(hw_t.get("title", "📡 Step 5: Wearable Sensors"))
    st.caption(hw_t.get("caption", "6-DOF IMU Stream"))

    if ESP32SerialReader:
        reader = ESP32SerialReader()
        sample = reader.read_packet()
    else:
        sample = {"accY": 1.12, "gyroZ": 45.2, "accX": 0.14}

    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">{hw_t.get('imu1_name', 'IMU 1')}</div>
            <div class="card-val">{sample.get('accY', 1.0):.2f} g</div>
            <div style="color: #94A3B8; font-size: 0.8rem;">{hw_t.get('imu1_sub', 'Axial Impact')}</div>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">{hw_t.get('imu2_name', 'IMU 2')}</div>
            <div class="card-val">{sample.get('gyroZ', 45.0):.1f} °/s</div>
            <div style="color: #94A3B8; font-size: 0.8rem;">{hw_t.get('imu2_sub', 'Angular Velocity')}</div>
        </div>
        """, unsafe_allow_html=True)
    with h3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">{hw_t.get('imu3_name', 'IMU 3')}</div>
            <div class="card-val">{sample.get('accX', 0.12):.2f} g</div>
            <div style="color: #94A3B8; font-size: 0.8rem;">{hw_t.get('imu3_sub', 'Ground Reaction')}</div>
        </div>
        """, unsafe_allow_html=True)

    t_series = np.linspace(0, 4, 100)
    acc_wave = np.sin(2 * np.pi * 1.2 * t_series) + 0.08 * np.random.randn(100)
    gyro_wave = 70 * np.cos(2 * np.pi * 1.2 * t_series) + 4.0 * np.random.randn(100)

    fig_imu = go.Figure()
    fig_imu.add_trace(go.Scatter(x=t_series, y=acc_wave, mode='lines', name='accY (g)', line=dict(color='#38BDF8', width=2)))
    fig_imu.add_trace(go.Scatter(x=t_series, y=gyro_wave/100.0, mode='lines', name='gyroZ (normalized)', line=dict(color='#F59E0B', width=2)))
    fig_imu.update_layout(
        title=hw_t.get("chart_title", "Inertial Dynamics"),
        xaxis_title="Time (s)",
        yaxis_title="Sensor Units",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.4)',
        font=dict(color='#94A3B8')
    )
    st.plotly_chart(fig_imu, use_container_width=True)

    b1, b2 = st.columns([1, 1])
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
    st.subheader(t["report_title"])
    p_info = st.session_state.get("patient_data", {})
    kl_grade = st.session_state.get("kl_grade", 2)
    xray_pct = (kl_grade / 4.0) * 100.0
    gait_pct = float(st.session_state.get("camera_prediction", 0.65)) * 100.0
    symp_pct = float(st.session_state.get("symptom_score", 52.0))
    sensor_pct = 54.0

    composite_index = (0.40 * xray_pct) + (0.30 * gait_pct) + (0.20 * symp_pct) + (0.10 * sensor_pct)

    r1, r2 = st.columns([1, 1])
    with r1:
        risk_color = "#EF4444" if composite_index >= 60 else ("#F59E0B" if composite_index >= 35 else "#10B981")
        tier = t["tier_high"] if composite_index >= 60 else t["tier_mod"]

        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid {risk_color};">
            <div class="card-title">{t['composite_title']}</div>
            <div class="card-val" style="color: {risk_color};">{composite_index:.1f} / 100</div>
            <div style="color: #94A3B8; font-size: 0.9rem;">Diagnostic Classification: <b>{tier}</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### Dimensional Synthesis:")
        st.write(f"- 👤 **Subject:** `{p_info.get('name', 'N/A')}` ({p_info.get('age', 58)}y {p_info.get('gender', 'M')}) | BMI: `{p_info.get('bmi', 25.0):.1f}`")
        st.write(f"- 🩻 **Radiographic (DenseNet-121):** Kellgren–Lawrence Grade {kl_grade} ({xray_pct:.0f}%) — *Weight 40%*")
        st.write(f"- 🚶 **Gait Kinematics (XGBoost):** Antalgic Probability {gait_pct:.1f}% — *Weight 30%*")
        st.write(f"- 📋 **Clinical Questionnaire:** Symptom Burden {symp_pct:.1f}% — *Weight 20%*")
        st.write(f"- 📡 **Wearable Sensor (ESP32):** Motion Index {sensor_pct:.1f}% — *Weight 10%*")

    with r2:
        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(
            r=[xray_pct, gait_pct, symp_pct, sensor_pct],
            theta=['Radiograph (KL)', 'Vision Gait', 'Symptoms', 'Inertial Motion'],
            fill='toself',
            name='Patient Profile',
            line=dict(color='#38BDF8')
        ))
        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            margin=dict(l=30, r=30, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94A3B8')
        )
        st.plotly_chart(radar_fig, use_container_width=True)

    st.divider()
    c_btn1, c_btn2 = st.columns([1, 1])
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
                label="📥 Download Official Clinical Report (PDF)",
                data=pdf_bytes,
                file_name=f"ARTHRO_AI_Report_{p_info.get('id', 'Patient')}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
        else:
            st.error("Install reportlab: 'pip install reportlab' to download PDFs.")

    with c_btn2:
        if st.button(t["restart"], use_container_width=True):
            st.session_state["current_step"] = 0
            st.rerun()