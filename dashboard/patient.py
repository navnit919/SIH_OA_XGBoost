# dashboard/patient.py
import streamlit as st
from datetime import date

PATIENT_STRINGS = {
    "English": {
        "title": "👤 Step 1: Patient Registration & Clinical Intake",
        "caption": "Enter patient demographics and clinical baseline metrics before starting diagnostic tests.",
        "demographics": "📝 Demographics",
        "name": "Patient Full Name",
        "id": "Patient ID / MRN",
        "age": "Age (Years)",
        "sex": "Biological Sex",
        "sex_options": ["Male", "Female", "Other"],
        "date": "Examination Date",
        "vitals": "🩺 Clinical Vitals & Joint Target",
        "joint": "Primary Knee Joint Evaluated",
        "joint_options": ["Right Knee", "Left Knee", "Bilateral"],
        "height": "Height (cm)",
        "weight": "Weight (kg)",
        "bmi_title": "Calculated Body Mass Index (BMI)",
        "bmi_cats": {
            "under": "Underweight",
            "normal": "Normal Weight",
            "over": "Overweight (Elevated OA Risk)",
            "obese": "Obese (High OA Risk Factor)"
        }
    },
    "हिन्दी": {
        "title": "👤 चरण 1: रोगी पंजीकरण एवं क्लिनिकल विवरण",
        "caption": "जांच शुरू करने से पहले रोगी का विवरण और प्राथमिक शारीरिक माप दर्ज करें।",
        "demographics": "📝 जनसांख्यिकी (Demographics)",
        "name": "रोगी का पूरा नाम",
        "id": "रोगी आईडी / MRN",
        "age": "उम्र (वर्ष)",
        "sex": "लिंग",
        "sex_options": ["पुरुष", "महिला", "अन्य"],
        "date": "जांच की तारीख",
        "vitals": "🩺 शारीरिक माप एवं लक्षित जोड़",
        "joint": "लक्षित घुटने का जोड़",
        "joint_options": ["दायां घुटना", "बायां घुटना", "दोनों घुटने"],
        "height": "कद / ऊंचाई (cm)",
        "weight": "वजन (kg)",
        "bmi_title": "बॉडी मास इंडेक्स (BMI)",
        "bmi_cats": {
            "under": "कम वजन (Underweight)",
            "normal": "सामान्य वजन (Normal)",
            "over": "अधिक वजन (गठिया का बढ़ा जोखिम)",
            "obese": "मोटापा (गठिया का उच्च जोखिम)"
        }
    },
    "অসমীয়া": {
        "title": "👤 স্তৰ ১: ৰোগীৰ পঞ্জীয়ন আৰু প্ৰাৰম্ভিক তথ্য",
        "caption": "পৰীক্ষা আৰম্ভ কৰাৰ পূৰ্বে ৰোগীৰ ব্যক্তিগত আৰু শাৰীৰিক তথ্যসমূহ প্ৰৱিষ্ট কৰক।",
        "demographics": "📝 সাধাৰণ তথ্য",
        "name": "ৰোগীৰ সম্পূৰ্ণ নাম",
        "id": "ৰোগীৰ পৰিচয় নং (ID)",
        "age": "বয়স (বছৰ)",
        "sex": "লিংগ",
        "sex_options": ["পুৰুষ", "মহিলা", "অন্যান্য"],
        "date": "পৰীক্ষাৰ তাৰিখ",
        "vitals": "🩺 শাৰীৰিক মাপ আৰু লক্ষ্য আঁঠু",
        "joint": "পৰীক্ষা কৰিবলগীয়া আঁঠুৰ জোৰা",
        "joint_options": ["সোঁ আঁঠু", "বাওঁ আঁঠু", "দুয়োটা আঁঠু"],
        "height": "উচ্চতা (cm)",
        "weight": "ওজন (kg)",
        "bmi_title": "বডি মাছ ইনডেক্স (BMI)",
        "bmi_cats": {
            "under": "স্বাভাৱিকতকৈ কম ওজন",
            "normal": "স্বাভাৱিক ওজন",
            "over": "অধিক ওজন (বাতবিষৰ ঝুঁকি)",
            "obese": "অতি ওজন (বাতবিষৰ উচ্চ ঝুঁকি)"
        }
    },
    "বাংলা": {
        "title": "👤 ধাপ ১: রোগীর নিবন্ধন ও প্রাথমিক ক্লিনিকাল তথ্য",
        "caption": "ডায়াগনস্টিক পরীক্ষা শুরুর পূর্বে রোগীর সাধারণ ও শারীরিক পরিমাপ লিপিবদ্ধ করুন।",
        "demographics": "📝 রোগীর বিবরণ",
        "name": "রোগীর সম্পূর্ণ নাম",
        "id": "রোগীর আইডি / MRN",
        "age": "বয়স (বছর)",
        "sex": "লিঙ্গ",
        "sex_options": ["পুরুষ", "মহিলা", "অন্যান্য"],
        "date": "পরীক্ষার তারিখ",
        "vitals": "🩺 শারীরিক পরিমাপ ও আক্রান্ত সন্ধি",
        "joint": "প্রধান পরীক্ষাধীন হাঁটুর সন্ধি",
        "joint_options": ["ডান হাঁটু", "বাম হাঁটু", "উভয় হাঁটু"],
        "height": "উচ্চতা (cm)",
        "weight": "ওজন (kg)",
        "bmi_title": "বডি মাস ইনডেক্স (BMI)",
        "bmi_cats": {
            "under": "স্বাভাবিকের চেয়ে কম ওজন",
            "normal": "স্বাভাবিক ওজন",
            "over": "অতিরিক্ত ওজন (অস্টিওআর্থারাইটিসের ঝুঁকি)",
            "obese": "স্থূলতা (উচ্চ ঝুঁকি)"
        }
    }
}

def patient_module():
    lang = st.session_state.get("language", "English")
    t = PATIENT_STRINGS.get(lang, PATIENT_STRINGS["English"])
    p_data = st.session_state.get("patient_data", {})

    st.subheader(t["title"])
    st.caption(t["caption"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"##### {t['demographics']}")
        name = st.text_input(t["name"], value=p_data.get("name", "Navnit Kumar"))
        patient_id = st.text_input(t["id"], value=p_data.get("id", "OA-2026-NITN"))
        
        c_age, c_gen = st.columns(2)
        with c_age:
            age = st.number_input(t["age"], min_value=18, max_value=100, value=int(p_data.get("age", 58)))
        with c_gen:
            gender = st.selectbox(t["sex"], t["sex_options"], index=0)
        
        exam_date = st.date_input(t["date"], value=date.today())

    with col2:
        st.markdown(f"##### {t['vitals']}")
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
            bmi_cat = (cats["under"], "#38BDF8")
        elif bmi < 25:
            bmi_cat = (cats["normal"], "#10B981")
        elif bmi < 30:
            bmi_cat = (cats["over"], "#F59E0B")
        else:
            bmi_cat = (cats["obese"], "#EF4444")

        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.75); border-radius: 12px; padding: 16px 20px; margin-top: 10px; border-left: 4px solid {bmi_cat[1]}; border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
            <span style="color: #94A3B8; font-size: 0.8rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;">{t['bmi_title']}</span><br>
            <span style="color: #F8FAFC; font-size: 1.5rem; font-weight: 800;">{bmi:.1f} kg/m²</span> 
            <span style="color: {bmi_cat[1]}; font-size: 0.9rem; font-weight: 600; margin-left: 8px;">({bmi_cat[0]})</span>
        </div>
        """, unsafe_allow_html=True)

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