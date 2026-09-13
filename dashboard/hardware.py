# dashboard/hardware.py
import time
import serial
import numpy as np
import streamlit as st

HW_STRINGS = {
    "English": {
        "title": "📡 Step 5: Wearable Inertial Motion Capture (ESP32 + 3× MPU6050)",
        "caption": "6-DOF Real-Time Inertial Stream | CSV Protocol: accX,accY,accZ,gyroX,gyroY,gyroZ via Serial GPIO 21/22",
        "imu1_name": "IMU 1: Thigh Acceleration (Y)",
        "imu1_sub": "Axial Impact Load",
        "imu2_name": "IMU 2: Shank Gyroscope (Z)",
        "imu2_sub": "Sagittal Angular Velocity",
        "imu3_name": "IMU 3: Ankle Shock Absorption",
        "imu3_sub": "Ground Reaction Shock Profile",
        "chart_title": "Real-Time 6-DOF Inertial Dynamics Stream"
    },
    "हिन्दी": {
        "title": "📡 चरण 5: वियरेबल इनर्शियल मोशन सेंसर (ESP32 + 3× MPU6050)",
        "caption": "6-DOF रीयल-टाइम डेटा स्ट्रीम | CSV प्रोटोकॉल: accX,accY,accZ,gyroX,gyroY,gyroZ (सीरियल GPIO 21/22)",
        "imu1_name": "सेंसर 1: जांघ का त्वरण (Thigh Accel Y)",
        "imu1_sub": "अक्षीय भार प्रभाव (Axial Impact Load)",
        "imu2_name": "सेंसर 2: पिंडली का जायरोस्कोप (Shank Gyro Z)",
        "imu2_sub": "घुटने की कोणीय गति (Angular Velocity)",
        "imu3_name": "सेंसर 3: टखने का शॉक अवशोषण (Ankle Shock)",
        "imu3_sub": "जमीन से लगने वाला झटका (Ground Reaction)",
        "chart_title": "रीयल-टाइम 6-DOF मोशन सेंसर वेवफॉर्म"
    },
    "অসমীয়া": {
        "title": "📡 স্তৰ ৫: ৱিয়েৰেবল মোচন চেন্সৰ (ESP32 + ৩× MPU6050)",
        "caption": "৬-DOF ৰিয়েল-টাইম গতিবিধি | CSV প্ৰট'কল (GPIO 21/22 চিৰিয়েল ডাটা)",
        "imu1_name": "চেন্সৰ ১: উৰুৰ ত্বৰণ (Thigh Accel)",
        "imu1_sub": "ওপৰৰ পৰা পৰা চাপ (Impact Load)",
        "imu2_name": "চেন্সৰ ২: ভৰিৰ কোণিক বেগ (Shank Gyro)",
        "imu2_sub": "আঁঠু ভাঁজ হোৱাৰ গতি",
        "imu3_name": "চেন্সৰ ৩: গোৰোহাৰ প্ৰভাৱ (Ankle Shock)",
        "imu3_sub": "মাটিত ভৰি দিয়াৰ সময়ৰ কম্পন",
        "chart_title": "ৰিয়েল-টাইম মোচন চেন্সৰ তৰংগ"
    },
    "বাংলা": {
        "title": "📡 ধাপ ৫: পরিধানযোগ্য মোশন সেন্সর (ESP32 + ৩টি MPU6050)",
        "caption": "৬-DOF রিয়েল-টাইম ডেটা স্ট্রিম | CSV প্রোটোকল (GPIO 21/22 সিরিয়াল যোগাযোগ)",
        "imu1_name": "সেন্সর ১: উরুর ত্বরণ (Thigh Accel Y)",
        "imu1_sub": "উল্লম্ব লোড প্রভাব (Axial Impact)",
        "imu2_name": "সেন্সর ২: পায়ের নিম্নভাগের জাইরো (Shank Gyro)",
        "imu2_sub": "সন্ধির ঘূর্ণন গতি (Angular Velocity)",
        "imu3_name": "সেন্সর ৩: গোড়ালির কম্পন শোষণ (Ankle Shock)",
        "imu3_sub": "মাটিতে পা ফেলার প্রতিক্রিয়া",
        "chart_title": "রিয়েল-টাইম মোশন সেন্সর ওয়েভফর্ম"
    }
}

class ESP32SerialReader:
    def __init__(self, port="COM4", baud_rate=115200):
        self.port = port
        self.baud_rate = baud_rate
        self.ser = None
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=0.1)
            time.sleep(1.0)
        except Exception:
            self.ser = None

    def is_connected(self):
        return self.ser is not None and self.ser.is_open

    def read_packet(self):
        if self.is_connected():
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    parts = [float(x.strip()) for x in line.split(",")]
                    if len(parts) >= 6:
                        return {
                            "accX": parts[0], "accY": parts[1], "accZ": parts[2],
                            "gyroX": parts[3], "gyroY": parts[4], "gyroZ": parts[5], "live": True
                        }
            except Exception:
                pass
        t = time.time()
        return {
            "accX": float(0.12 * np.sin(2.0 * t)),
            "accY": float(1.0 + 0.25 * np.cos(2.0 * t)),
            "accZ": float(0.06 * np.sin(t)),
            "gyroX": float(15.0 * np.sin(3.0 * t)),
            "gyroY": float(48.0 * np.cos(2.0 * t)),
            "gyroZ": float(8.0 * np.sin(t)),
            "live": False
        }