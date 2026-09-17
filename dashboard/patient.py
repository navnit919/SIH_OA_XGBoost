# dashboard/patient.py
import streamlit as st
from datetime import date

PATIENT_STRINGS = {
    "English": {
        "title": "👤 Patient Registration & Baseline Intake",
        "caption": "Enter patient demographics and clinical baseline metrics before starting diagnostic tests.",
        "demographics": "📝 Personal Demographics",
        "name": "Patient Full Name",
        "id": "Patient ID / MRN",
        "age": "Age (Years)",
        "sex": "Biological Sex",
        "sex_options": ("Male", "Female", "Other"),
        "date": "Examination Date",
        "vitals": "🩺 Clinical Vitals & Joint Target",
        "joint": "Primary Knee Joint Evaluated",
        "joint_options": ("Right Knee", "Left Knee", "Bilateral"),
        "height": "Height (cm)",
        "weight": "Weight (kg)",
        "bmi_title": "Body Mass Index (BMI)",
        "bmi_cats": {
            "under": "Underweight",
            "normal": "Normal Weight (Healthy)",
            "over": "Overweight (Elevated OA Risk)",
            "obese": "Obese (High OA Risk Factor)"
        }
    },
    "हिन्दी": {
        "title": "👤 रोगी पंजीकरण एवं प्रारंभिक विवरण",
        "caption": "जांच शुरू करने से पहले रोगी का विवरण और शारीरिक माप दर्ज करें।",
        "demographics": "📝 व्यक्तिगत विवरण (Demographics)",
        "name": "रोगी का पूरा नाम",
        "id": "रोगी आईडी / MRN",
        "age": "उम्र (वर्ष)",
        "sex": "लिंग",
        "sex_options": ("पुरुष", "महिला", "अन्य"),
        "date": "जांच की तारीख",
        "vitals": "🩺 शारीरिक माप एवं लक्षित जोड़",
        "joint": "लक्षित घुटने का जोड़",
        "joint_options": ("दायां घुटना", "बायां घुटना", "दोनों घुटने"),
        "height": "कद / ऊंचाई (cm)",
        "weight": "वजन (kg)",
        "bmi_title": "बॉडी मास इंडेक्स (BMI)",
        "bmi_cats": {
            "under": "कम वजन (Underweight)",
            "normal": "सामान्य वजन (स्वस्थ)",
            "over": "अधिक वजन (गठिया का बढ़ा जोखिम)",
            "obese": "मोटापा (गठिया का उच्च जोखिम)"
        }
    },
    "অসমীয়া": {
        "title": "👤 ৰোগীৰ পঞ্জীয়ন আৰু প্ৰাৰম্ভিক তথ্য",
        "caption": "পৰীক্ষা আৰম্ভ কৰাৰ পূৰ্বে ৰোগীৰ ব্যক্তিগত আৰু শাৰীৰিক তথ্যসমূহ প্ৰৱিষ্ট কৰক।",
        "demographics": "📝 সাধাৰণ তথ্য",
        "name": "ৰোগীৰ সম্পূৰ্ণ নাম",
        "id": "ৰোগীৰ পৰিচয় নং (ID)",
        "age": "বয়স (বছৰ)",
        "sex": "লিংগ",
        "sex_options": ("পুৰুষ", "মহিলা", "অন্যান্য"),
        "date": "পৰীক্ষাৰ তাৰিখ",
        "vitals": "🩺 শাৰীৰিক মাপ আৰু লক্ষ্য আঁঠু",
        "joint": "পৰীক্ষা কৰিবলগীয়া আঁঠুৰ জোৰা",
        "joint_options": ("সোঁ আঁঠু", "বাওঁ আঁঠু", "দুয়োটা আঁঠু"),
        "height": "উচ্চতা (cm)",
        "weight": "ওজন (kg)",
        "bmi_title": "বডি মাছ ইনডেক্স (BMI)",
        "bmi_cats": {
            "under": "স্বাভাৱিকতকৈ কম ওজন",
            "normal": "স্বাভাৱিক ওজন (সুস্থ)",
            "over": "অধিক ওজন (বাতবিষৰ ঝুঁকি)",
            "obese": "অতি ওজন (বাতবিষৰ উচ্চ ঝুঁকি)"
        }
    },
    "বাংলা": {
        "title": "👤 রোগীর নিবন্ধন ও প্রাথমিক ক্লিনিকাল তথ্য",
        "caption": "ডায়াগনস্টিক পরীক্ষা শুরুর পূর্বে রোগীর সাধারণ ও শারীরিক পরিমাপ লিপিবদ্ধ করুন।",
        "demographics": "📝 রোগীর সাধারণ বিবরণ",
        "name": "রোগীর সম্পূর্ণ নাম",
        "id": "রোগীর আইডি / MRN",
        "age": "বয়স (বছর)",
        "sex": "লিঙ্গ",
        "sex_options": ("পুরুষ", "মহিলা", "অন্যান্য"),
        "date": "পরীক্ষার তারিখ",
        "vitals": "🩺 শারীরিক পরিমাপ ও আক্রান্ত সন্ধি",
        "joint": "প্রধান পরীক্ষাধীন হাঁটুর সন্ধি",
        "joint_options": ("ডান হাঁটু", "বাম হাঁটু", "উভয় হাঁটু"),
        "height": "উচ্চতা (cm)",
        "weight": "ওজন (kg)",
        "bmi_title": "বডি মাস ইনডেক্স (BMI)",
        "bmi_cats": {
            "under": "স্বাভাবিকের চেয়ে কম ওজন",
            "normal": "স্বাভাবিক ওজন (সুস্থ)",
            "over": "অতিরিক্ত ওজন (অস্টিওআর্থারাইটিসের ঝুঁকি)",
            "obese": "স্থূলতা (উচ্চ ঝুঁকি)"
        }
    }
}

def patient_module():
    lang = st.session_state.get("language", "English")
    t = PATIENT_STRINGS.get(lang, PATIENT_STRINGS["English"])
    p_data = st.session_state.get("patient_data", {})

    st.markdown(f"<h2 style='font-size: 2.2rem; font-weight: 800; color: #1E1B4B; margin-bottom: 4px;'>{t['title']}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #64748B; font-size: 1.05rem; margin-bottom: 25px;'>{t['caption']}</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown(f"""
        <div style="background: #FFFFFF; border-radius: 20px; padding: 25px; box-shadow: 0 10px 25px rgba(91, 81, 216, 0.06); border: 1px solid #EAE7FB; margin-bottom: 20px;">
            <h4 style="color: #5B51D8; font-size: 1.25rem; font-weight: 700; margin-top: 0; margin-bottom: 15px;">{t['demographics']}</h4>
        """, unsafe_allow_html=True)
        name = st.text_input(t["name"], value=p_data.get("name", "Navnit Kumar"))
        patient_id = st.text_input(t["id"], value=p_data.get("id", "OA-2026-NITN"))
        
        c_age, c_gen = st.columns(2)
        with c_age:
            age = st.number_input(t["age"], min_value=18, max_value=100, value=int(p_data.get("age", 58)))
        with c_gen:
            gender = st.selectbox(t["sex"], t["sex_options"], index=0)
        
        exam_date = st.date_input(t["date"], value=date.today())
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background: #FFFFFF; border-radius: 20px; padding: 25px; box-shadow: 0 10px 25px rgba(91, 81, 216, 0.06); border: 1px solid #EAE7FB; margin-bottom: 20px;">
            <h4 style="color: #5B51D8; font-size: 1.25rem; font-weight: 700; margin-top: 0; margin-bottom: 15px;">{t['vitals']}</h4>
        """, unsafe_allow_html=True)
        joint = st.radio(t["joint"], t["joint_options"], index=0, horizontal=True)
        
        c_ht, c_wt = st.columns(2)
        with c_ht:
            height = st.number_input(t["height"], min_value=120.0, max_value=220.0, value=float(p_data.get("height", 170.0)))
        with c_wt:
            weight = st.number_input(t["weight"], min_value=30.0, max_value=180.0, value=float(p_data.get("weight", 74.0)))

        height_m = height / 100.0
        bmi = weight / (height_m ** 2)
        
        cats = t["bmi_cats"]
        if bmi < 18.5:
            bmi_label, bmi_col, bmi_bg = cats["under"], "#0284C7", "#E0F2FE"
        elif bmi < 25:
            bmi_label, bmi_col, bmi_bg = cats["normal"], "#059669", "#DCFCE7"
        elif bmi < 30:
            bmi_label, bmi_col, bmi_bg = cats["over"], "#D97706", "#FEF3C7"
        else:
            bmi_label, bmi_col, bmi_bg = cats["obese"], "#DC2626", "#FEE2E2"

        st.markdown(f"""
        <div style="background: {bmi_bg}; border-radius: 16px; padding: 18px 22px; margin-top: 18px; border: 1.5px solid {bmi_col}; display: flex; align-items: center; justify-content: space-between;">
            <div>
                <span style="color: {bmi_col}; font-size: 0.85rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.06em;">{t['bmi_title']}</span><br>
                <span style="color: #1E1B4B; font-size: 2.1rem; font-weight: 800;">{bmi:.1f} <span style="font-size: 1.05rem; font-weight: 600; color: #475569;">kg/m²</span></span>
            </div>
            <div style="background: #FFFFFF; border-radius: 30px; padding: 8px 16px; font-weight: 700; color: {bmi_col}; font-size: 0.9rem; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                {bmi_label}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.session_state["patient_data"] = {
        "name": name,
        "id": patient_id,
        "age": age,
        "gender": gender,
        "joint": joint,
        "height": height,
        "weight": weight,
        "bmi": bmi,
        "exam_date": str(exam_date)
    }