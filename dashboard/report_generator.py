# dashboard/report_generator.py
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_clinical_pdf(patient_data, kl_grade, xray_conf, gait_prob, symptom_score, sensor_score=54.0):
    """
    Generates an official, print-ready Clinical PDF report in memory.
    Returns bytes suitable for Streamlit's st.download_button.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#64748B')
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#334155')
    )
    bold_style = ParagraphStyle(
        'BodyDarkBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("ARTHRO-AI | CLINICAL DIAGNOSTIC REPORT", title_style))
    story.append(Paragraph("Multimodal Osteoarthritis Screening & Quantitative Risk Assessment", subtitle_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M:%S')} | Institutional Node: NITN-CPS-2026", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#0284C7'), spaceBefore=2, spaceAfter=12))

    # 2. Patient Clinical Demographics
    story.append(Paragraph("I. PATIENT CLINICAL DEMOGRAPHICS", section_heading))
    
    name = patient_data.get("name", "N/A")
    pid = patient_data.get("id", "N/A")
    age = patient_data.get("age", 58)
    gender = patient_data.get("gender", "Male")
    joint = patient_data.get("joint", "Right Knee")
    bmi = patient_data.get("bmi", 25.6)
    ht = patient_data.get("height", 170.0)
    wt = patient_data.get("weight", 74.0)

    patient_table_data = [
        [Paragraph("<b>Patient Name:</b>", body_style), Paragraph(name, body_style),
         Paragraph("<b>Medical Record No (MRN):</b>", body_style), Paragraph(pid, bold_style)],
        [Paragraph("<b>Age / Gender:</b>", body_style), Paragraph(f"{age} Years / {gender}", body_style),
         Paragraph("<b>Target Joint:</b>", body_style), Paragraph(joint, bold_style)],
        [Paragraph("<b>Height / Weight:</b>", body_style), Paragraph(f"{ht:.1f} cm / {wt:.1f} kg", body_style),
         Paragraph("<b>Body Mass Index (BMI):</b>", body_style), Paragraph(f"{bmi:.1f} kg/m²", bold_style)],
    ]
    
    patient_table = Table(patient_table_data, colWidths=[110, 155, 135, 140])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 10))

    # 3. Composite Diagnostic Summary Box
    xray_pct = (kl_grade / 4.0) * 100.0
    gait_pct = gait_prob * 100.0
    symp_pct = symptom_score
    sens_pct = sensor_score
    composite_index = (0.40 * xray_pct) + (0.30 * gait_pct) + (0.20 * symp_pct) + (0.10 * sens_pct)

    if composite_index >= 60:
        tier_title = "HIGH RISK / RADIOGRAPHICALLY CONFIRMED OSTEOARTHRITIS"
        tier_bg = colors.HexColor('#FEE2E2')
        tier_color = colors.HexColor('#DC2626')
    elif composite_index >= 35:
        tier_title = "MODERATE RISK / EARLY STAGE MILD OSTEOARTHRITIS"
        tier_bg = colors.HexColor('#FEF3C7')
        tier_color = colors.HexColor('#D97706')
    else:
        tier_title = "LOW RISK / PHYSIOLOGICAL AGE-APPROPRIATE JOINT"
        tier_bg = colors.HexColor('#DCFCE7')
        tier_color = colors.HexColor('#16A34A')

    summary_box_data = [
        [Paragraph(f"<b>COMPOSITE OA DIAGNOSTIC INDEX: {composite_index:.1f} / 100</b>", ParagraphStyle('W1', parent=bold_style, fontSize=11, textColor=tier_color))],
        [Paragraph(f"<b>CLINICAL TIER:</b> {tier_title}", ParagraphStyle('W2', parent=bold_style, fontSize=9, textColor=colors.HexColor('#1E293B')))]
    ]
    summary_table = Table(summary_box_data, colWidths=[540])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), tier_bg),
        ('BOX', (0,0), (-1,-1), 1.5, tier_color),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10))

    # 4. Multimodal Pillar Findings Table
    story.append(Paragraph("II. INDEPENDENT MODALITY EVALUATION", section_heading))
    
    kl_labels = {
        0: "Grade 0 (Normal - No Radiographic OA)",
        1: "Grade 1 (Doubtful JSN, Minute Osteophytes)",
        2: "Grade 2 (Mild - Definite Osteophytes)",
        3: "Grade 3 (Moderate - Multiple Osteophytes, Definite JSN)",
        4: "Grade 4 (Severe - Marked JSN, Sclerosis, Bone Deformity)"
    }

    modality_table_data = [
        [Paragraph("<b>Diagnostic Modality</b>", bold_style), 
         Paragraph("<b>Clinical AI Engine</b>", bold_style), 
         Paragraph("<b>Quantitative Findings</b>", bold_style),
         Paragraph("<b>Weight</b>", bold_style)],
        
        [Paragraph("🩻 <b>Radiographic X-Ray</b>", body_style),
         Paragraph("DenseNet-121 (CNN)", body_style),
         Paragraph(f"<b>{kl_labels.get(kl_grade, 'Grade ' + str(kl_grade))}</b><br/>Confidence: {xray_conf*100:.1f}%", body_style),
         Paragraph("40%", body_style)],
        
        [Paragraph("🚶 <b>Gait Kinematics</b>", body_style),
         Paragraph("MediaPipe + XGBoost V5", body_style),
         Paragraph(f"OA Pattern Risk: <b>{gait_pct:.1f}%</b><br/>Pattern: {'Antalgic / Asymmetric Gait' if gait_pct > 50 else 'Normal Biomechanical Gait'}", body_style),
         Paragraph("30%", body_style)],
        
        [Paragraph("📋 <b>Clinical Questionnaire</b>", body_style),
         Paragraph("Standardized WOMAC Scale", body_style),
         Paragraph(f"Symptom Burden: <b>{symp_pct:.1f}%</b><br/>Status: {'Moderate/Severe Functional Limitation' if symp_pct > 50 else 'Mild Symptom Burden'}", body_style),
         Paragraph("20%", body_style)],

        [Paragraph("📡 <b>Wearable Inertial IMU</b>", body_style),
         Paragraph("ESP32 + 3× MPU6050", body_style),
         Paragraph(f"Motion Restriction Index: <b>{sens_pct:.1f}%</b><br/>6-DOF Dynamic Axial Profile", body_style),
         Paragraph("10%", body_style)],
    ]

    modality_table = Table(modality_table_data, colWidths=[125, 120, 235, 60])
    modality_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')])
    ]))
    story.append(modality_table)
    story.append(Spacer(1, 10))

    # 5. Clinical Recommendations
    story.append(Paragraph("III. RECOMMENDATIONS & CLINICAL MANAGEMENT PLAN", section_heading))
    if composite_index >= 60:
        plan_text = """
        • <b>Orthopedic Consultation:</b> Prompt referral for joint preservation therapy or surgical evaluation.<br/>
        • <b>Pharmacological Therapy:</b> Topical NSAIDs, oral analgesics, or intra-articular interventions as indicated.<br/>
        • <b>Physical Therapy:</b> Targeted quadriceps strengthening and unloader knee bracing to reduce medial joint compartment pressure.<br/>
        • <b>Weight Management:</b> Caloric optimization to reduce cumulative axial joint impact.
        """
    elif composite_index >= 35:
        plan_text = """
        • <b>Conservative Therapy:</b> Structured physical therapy focused on hamstring/quadriceps flexibility and gait re-education.<br/>
        • <b>Symptom Surveillance:</b> Repeat multimodal radiographic and kinematic assessment in 6 months.<br/>
        • <b>Low-Impact Activity:</b> Swimming or stationary cycling to preserve synovial joint lubrication.
        """
    else:
        plan_text = """
        • <b>Preventative Guidance:</b> Maintain routine physical conditioning and baseline joint flexibility.<br/>
        • <b>Periodic Follow-up:</b> Annual screening or re-evaluation if symptomatic changes occur.
        """
    story.append(Paragraph(plan_text, body_style))
    story.append(Spacer(1, 14))

    # 6. Attending Sign-Off Table
    sign_table_data = [
        [Paragraph("<b>Evaluating Clinician:</b> ___________________________", body_style),
         Paragraph("<b>Signature & Institutional Seal:</b> ___________________________", body_style)],
        [Paragraph("ARTHRO-AI Multimodal Clinical Workstation", subtitle_style),
         Paragraph("Confidential Medical Screening Document", subtitle_style)]
    ]
    sign_table = Table(sign_table_data, colWidths=[270, 270])
    sign_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
        ('TOPPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(sign_table)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()