\# 🦴 AI-Powered Multimodal Osteoarthritis Assessment System



An AI-powered multimodal system for Osteoarthritis (OA) screening and assessment, integrating X-ray analysis, camera-based gait analysis, patient questionnaires, and wearable motion sensing.



\## 🚀 Overview



The system combines four complementary sources of information:



\* 🩻 X-ray analysis using DenseNet-121

\* 🚶 Camera-based gait analysis using MediaPipe Pose and XGBoost

\* 📋 Patient questionnaire and symptom assessment

\* 📡 ESP32 + MPU6050 motion sensing



All components are integrated into a unified dashboard.



> \*\*Disclaimer:\*\* This project is intended for research, education, and prototype screening purposes. It is not a substitute for professional medical diagnosis.



\---



\## 🧠 System Architecture



```text

&#x20;                        PATIENT

&#x20;                           │

&#x20;                           ▼

&#x20;                   ┌───────────────┐

&#x20;                   │   DASHBOARD   │

&#x20;                   │ dashboard.py  │

&#x20;                   └───────┬───────┘

&#x20;                           │

&#x20;         ┌─────────────────┼─────────────────┐

&#x20;         │                 │                 │

&#x20;         ▼                 ▼                 ▼

&#x20;   QUESTIONNAIRE          X-RAY             GAIT

&#x20;questionnaire.py        xray.py           gait.py

&#x20;         │                 │                 │

&#x20;         │                 ▼                 ▼

&#x20;         │            DenseNet-121       Camera Input

&#x20;         │                 │                 │

&#x20;         │                 ▼                 ▼

&#x20;         │       best\_densenet121.pth   MediaPipe Pose

&#x20;         │                                   │

&#x20;         │                                   ▼

&#x20;         │                             Gait Features

&#x20;         │                                   │

&#x20;         │                                   ▼

&#x20;         │                              XGBoost V5

&#x20;         │                                   │

&#x20;         │                                   ▼

&#x20;         │                         gait\_model\_v5.pkl

&#x20;         │

&#x20;         └─────────────────┬─────────────────┘

&#x20;                           │

&#x20;                           ▼

&#x20;                      HARDWARE

&#x20;                    hardware.py

&#x20;                           │

&#x20;                           ▼

&#x20;                    ESP32 + MPU6050

&#x20;                           │

&#x20;                           ▼

&#x20;                    Motion Data

&#x20;                           │

&#x20;                           ▼

&#x20;                 INTEGRATED ASSESSMENT

```



\---



\## 🤖 AI Components



\### 1. X-ray Analysis



The X-ray component uses a trained DenseNet-121 model.



```text

X-ray Image

&#x20;    ↓

DenseNet-121

&#x20;    ↓

OA Prediction

```



Main files:



```text

xray/

├── xray\_prediction.py

└── evaluate\_xray.py

```



Model:



```text

models/xray/best\_densenet121.pth

```



The trained model is excluded from the public repository because of file size and distribution considerations.



\---



\### 2. Camera Gait Analysis



The gait system uses MediaPipe Pose Landmarker to extract body landmarks and XGBoost to classify gait-related features.



```text

Camera

&#x20;  ↓

MediaPipe Pose

&#x20;  ↓

Body Landmarks

&#x20;  ↓

Gait Features

&#x20;  ↓

XGBoost V5

&#x20;  ↓

Gait Prediction

```



Main components:



```text

camera/pose\_landmarker\_lite.task

dashboard/gait.py

models/gait/oa\_xgboost\_gait\_model\_v5.pkl

```



V5 is the current/latest gait model.



\---



\### 3. ESP32 + MPU6050



The hardware subsystem is designed around an ESP32 and multiple MPU6050 inertial sensors.



```text

MPU6050 #1 ─┐

MPU6050 #2 ─┼── ESP32 ─── Python Dashboard

MPU6050 #3 ─┘

```



Each MPU6050 provides accelerometer and gyroscope measurements.



Python integration:



```text

dashboard/hardware.py

```



\---



\### 4. Questionnaire



The questionnaire component collects patient-reported symptoms and functional information.



```text

dashboard/questionnaire.py

```



\---



\## 🖥️ Dashboard



The main application is:



```text

dashboard/dashboard.py

```



It integrates:



\* Patient information

\* Questionnaire

\* X-ray assessment

\* Camera gait assessment

\* Hardware sensor integration

\* Assessment results



\---



\## 📁 Project Structure



```text

SIH\_OA\_XGBoost/

│

├── dashboard/

│   ├── dashboard.py

│   ├── patient.py

│   ├── questionnaire.py

│   ├── xray.py

│   ├── gait.py

│   ├── gait\_backup.py

│   └── hardware.py

│

├── camera/

│   ├── live\_oa\_prediction.py

│   └── pose\_landmarker\_lite.task

│

├── xray/

│   ├── xray\_prediction.py

│   └── evaluate\_xray.py

│

├── gait\_analysis/

│   ├── gait.py

│   └── gait\_camera.py

│

├── models/

│   ├── gait/

│   └── xray/

│

├── data/

│   └── gait/

│

├── fusion/

├── questionnaire/

├── reports/

├── tests/

│

├── requirements.txt

├── README.md

└── .gitignore

```



\---



\## ⚙️ Installation



Clone the repository:



```bash

git clone https://github.com/YOUR\_USERNAME/SIH\_OA\_XGBoost.git

cd SIH\_OA\_XGBoost

```



Create a virtual environment:



```powershell

python -m venv .venv

.venv\\Scripts\\activate

```



Install dependencies:



```powershell

pip install -r requirements.txt

```



\---



\## 📦 Required Model Files



Place the trained models locally at:



```text

models/xray/best\_densenet121.pth

models/gait/oa\_xgboost\_gait\_model\_v5.pkl

```



The MediaPipe pose model should be located at:



```text

camera/pose\_landmarker\_lite.task

```



\---



\## ▶️ Running the Dashboard



From the project root:



```powershell

streamlit run dashboard/dashboard.py

```



\---



\## 🔬 Training and Research



The repository contains scripts used during gait-model development and evaluation, including:



```text

train\_gait.py

train\_gait\_xgboost\_v2.py

train\_gait\_xgboost\_v4.py

train\_gait\_xgboost\_v5.py

extract\_gait\_features\_v5.py

compare\_gait\_models.py

fair\_compare\_v4\_v5.py

recover\_bilateral\_gait.py

```



These are development/research scripts and are not required for normal dashboard operation.



\---



\## 🔐 Privacy



Do not commit:



\* Patient-identifiable information

\* Private medical images

\* Raw patient recordings

\* Passwords

\* API keys

\* `.env` files

\* Private datasets



\---



\## ⚠️ Medical Disclaimer



This project is a research and prototype system for AI-assisted Osteoarthritis assessment.



Its predictions should not be considered a definitive medical diagnosis. Clinical decisions should be made by qualified healthcare professionals.



\---



\## 🔮 Future Work



\* Multimodal model fusion

\* Improved gait classification

\* Larger and more diverse datasets

\* Real-time wearable sensing

\* Automated assessment reports

\* Longitudinal patient monitoring

\* Explainable AI

\* Clinical validation

\* Secure patient-data management



\---



\## 👥 Project



\*\*SIH — AI-Powered Multimodal Osteoarthritis Assessment System\*\*



A multimodal AI-assisted platform combining medical imaging, computer vision, machine learning, questionnaires, and wearable sensing.



