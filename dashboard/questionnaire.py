# dashboard/questionnaire.py
import streamlit as st

QUEST_STRINGS = {
    "English": {
        "header": "📋 Clinical Symptom Assessment (WOMAC-Aligned)",
        "caption": "Quantifies subjective patient pain, joint stiffness, and daily activity restrictions.",
        "pain_sec": "Joint Pain Severity (0 = None, 10 = Severe)",
        "q1": "Pain walking on flat ground",
        "q2": "Pain ascending or descending stairs",
        "q3": "Pain at rest or in bed at night",
        "stiff_sec": "Joint Stiffness & Functional Limits",
        "q4": "Morning joint stiffness duration/severity",
        "q5": "Difficulty bending or squatting down",
        "q6": "Limitation in performing routine duties",
        "index_title": "Normalized Clinical Burden Score",
        "moderate_status": "Moderate to Severe Limitation",
        "mild_status": "Mild Symptomatic Burden"
    },
    "हिन्दी": {
        "header": "📋 क्लिनिकल लक्षण मूल्यांकन (WOMAC मानक)",
        "caption": "घुटने के दर्द, अकड़न और दैनिक कार्यों में कठिनाई का मानकीकृत मूल्यांकन।",
        "pain_sec": "जोड़ों के दर्द की गंभीरता (0 = बिल्कुल नहीं, 10 = अत्यधिक)",
        "q1": "समतल जमीन पर चलने में दर्द",
        "q2": "सीढ़ियां चढ़ने या उतरने में दर्द",
        "q3": "आराम करते समय या रात में बिस्तर पर दर्द",
        "stiff_sec": "जोड़ों की अकड़न एवं दैनिक कार्य सीमाएं",
        "q4": "सुबह उठने पर घुटने की अकड़न की अवधि/गंभीरता",
        "q5": "घुटने मोड़ने या नीचे बैठने में कठिनाई",
        "q6": "दैनिक घरेलू या कार्यस्थल के कार्यों में रुकावट",
        "index_title": "मानकीकृत लक्षण सूचकांक",
        "moderate_status": "मध्यम से गंभीर कार्यिक रुकावट",
        "mild_status": "हल्के / शुरुआती लक्षण"
    },
    "অসমীয়া": {
        "header": "📋 ৰোগৰ লক্ষণ মূল্যায়ন (WOMAC নিৰ্দেশনা)",
        "caption": "আঁঠুৰ বিষ, টান ভাব আৰু দৈনন্দিন কামত হোৱা অসুবিধাৰ পৰিমাণ নিৰ্ধাৰণ।",
        "pain_sec": "আঁঠুৰ বিষৰ মাত্ৰা (০ = বিষ নাই, ১০ = তীব্ৰ বিষ)",
        "q1": "সমান মাটিত খোজ কঢ়াৰ সময়ত বিষ",
        "q2": "চিৰিৰে উঠা বা নমাৰ সময়ত বিষ",
        "q3": "বিশ্রামৰ সময়ত বা ৰাতি বিছনাত বিষ",
        "stiff_sec": "আঁঠুৰ টান ভাব আৰু কাৰ্যক্ষমতা",
        "q4": "ৰাতিপুৱা আঁঠুৰ টান ভাবৰ স্থায়িত্ব/তীব্ৰতা",
        "q5": "আঁঠু ভাঁজ কৰা বা বহাত অসুবিধা",
        "q6": "দৈনন্দিন কাম-কাজ কৰাত অসুবিধা",
        "index_title": "মুঠ লক্ষণৰ সূচকাঙ্ক",
        "moderate_status": "মধ্যমৰ পৰা গুৰুতৰ সমস্যা",
        "mild_status": "কম মাত্ৰাৰ লক্ষণ"
    },
    "বাংলা": {
        "header": "📋 ক্লিনিকাল লক্ষণ মূল্যায়ন (WOMAC ভিত্তিক)",
        "caption": "হাঁটুর ব্যথা, আড়ষ্টতা এবং দৈনন্দিন কাজের সীমাবদ্ধতার পরিমাণ নির্ধারণ।",
        "pain_sec": "সন্ধির ব্যথার তীব্রতা (০ = ব্যথা নেই, ১০ = চরম ব্যথা)",
        "q1": "সমতল স্থানে হাঁটার সময় ব্যথা",
        "q2": "সিঁড়ি দিয়ে ওঠার বা নামার সময় ব্যথা",
        "q3": "বিশ্রামের সময় বা রাতে বিছানায় ব্যথা",
        "stiff_sec": "হাঁটুর আড়ষ্টতা ও কার্যিক সীমাবদ্ধতা",
        "q4": "সকালে ঘুম থেকে ওঠার পর হাঁটুর আড়ষ্টতার তীব্রতা",
        "q5": "হাঁটু বাঁকানো বা নিচে বসতে অসুবিধা",
        "q6": "দৈনন্দিন রুটিনমাফিক কাজে বাধার সম্মুখীন হওয়া",
        "index_title": "লক্ষণ মাত্রা সূচক",
        "moderate_status": "মাঝারি থেকে তীব্র সীমাবদ্ধতা",
        "mild_status": "মৃদু লক্ষণ"
    }
}

def questionnaire_module():
    lang = st.session_state.get("language", "English")
    t = QUEST_STRINGS.get(lang, QUEST_STRINGS["English"])

    st.markdown(f"<h2 style='font-size: 2.2rem; font-weight: 800; color: #1E1B4B; margin-bottom: 4px;'>{t['header']}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #64748B; font-size: 1.05rem; margin-bottom: 25px;'>{t['caption']}</p>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f"""
        <div style="background: #FFFFFF; border-radius: 20px; padding: 25px; box-shadow: 0 10px 25px rgba(91, 81, 216, 0.06); border: 1px solid #EAE7FB; margin-bottom: 20px;">
            <h4 style="color: #5B51D8; font-size: 1.2rem; font-weight: 700; margin-top: 0; margin-bottom: 15px;">{t['pain_sec']}</h4>
        """, unsafe_allow_html=True)
        q1 = st.slider(t["q1"], 0, 10, 4)
        q2 = st.slider(t["q2"], 0, 10, 6)
        q3 = st.slider(t["q3"], 0, 10, 3)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="background: #FFFFFF; border-radius: 20px; padding: 25px; box-shadow: 0 10px 25px rgba(91, 81, 216, 0.06); border: 1px solid #EAE7FB; margin-bottom: 20px;">
            <h4 style="color: #5B51D8; font-size: 1.2rem; font-weight: 700; margin-top: 0; margin-bottom: 15px;">{t['stiff_sec']}</h4>
        """, unsafe_allow_html=True)
        q4 = st.slider(t["q4"], 0, 10, 5)
        q5 = st.slider(t["q5"], 0, 10, 7)
        q6 = st.slider(t["q6"], 0, 10, 5)
        st.markdown("</div>", unsafe_allow_html=True)

    symptom_total = ((q1 + q2 + q3 + q4 + q5 + q6) / 60.0) * 100.0
    st.session_state["symptom_score"] = symptom_total

    status_txt = t["moderate_status"] if symptom_total > 50 else t["mild_status"]
    status_color = "#D97706" if symptom_total > 50 else "#059669"
    status_bg = "#FEF3C7" if symptom_total > 50 else "#DCFCE7"

    st.markdown(f"""
    <div style="background: #FFFFFF; border-radius: 20px; padding: 24px 30px; box-shadow: 0 10px 25px rgba(91, 81, 216, 0.06); border: 1px solid #EAE7FB; margin-top: 15px; display: flex; align-items: center; justify-content: space-between;">
        <div>
            <span style="color: #64748B; font-size: 0.9rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">{t['index_title']}</span><br>
            <span style="color: #1E1B4B; font-size: 2.6rem; font-weight: 800;">{symptom_total:.1f}%</span>
        </div>
        <div style="background: {status_bg}; color: {status_color}; font-size: 1.05rem; font-weight: 700; padding: 10px 20px; border-radius: 40px; border: 1.5px solid {status_color};">
            {status_txt}
        </div>
    </div>
    """, unsafe_allow_html=True)
    return symptom_total