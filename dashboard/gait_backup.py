from pathlib import Path
from collections import deque

import time

import cv2
import numpy as np
import streamlit as st
import joblib

import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


PROJECT_ROOT = Path(__file__).resolve().parents[1]

XGBOOST_PATH = (
    PROJECT_ROOT
    / "models"
    / "gait"
    / "oa_xgboost_gait_model_v2.pkl"
)

POSE_MODEL_PATH = (
    PROJECT_ROOT
    / "camera"
    / "pose_landmarker_lite.task"
)


@st.cache_resource
def load_gait_model():

    return joblib.load(
        XGBOOST_PATH
    )


@st.cache_resource
def load_pose_model():

    base_options = python.BaseOptions(
        model_asset_path=str(
            POSE_MODEL_PATH
        )
    )

    options = (
        vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=(
                vision.RunningMode.IMAGE
            ),
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
    )

    return (
        vision.PoseLandmarker
        .create_from_options(options)
    )


def calculate_angle(
    hip,
    knee,
    ankle
):

    hip = np.array(hip)

    knee = np.array(knee)

    ankle = np.array(ankle)

    vector_a = hip - knee

    vector_b = ankle - knee

    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )

    if denominator == 0:

        return None

    cosine = (
        np.dot(
            vector_a,
            vector_b
        )
        / denominator
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    internal_angle = np.degrees(
        np.arccos(cosine)
    )

    flexion = (
        180.0 - internal_angle
    )

    return float(flexion)


def extract_features(
    angles
):

    values = np.array(
        angles,
        dtype=float
    )

    return {
        "knee_rom": float(
            np.max(values)
            - np.min(values)
        ),

        "knee_mean": float(
            np.mean(values)
        ),

        "knee_std": float(
            np.std(values)
        ),

        "knee_max_flex": float(
            np.max(values)
        ),

        "knee_min_flex": float(
            np.min(values)
        ),

        "knee_median": float(
            np.median(values)
        )
    }


def run_camera_assessment():

    try:

        gait_model = load_gait_model()

        pose_landmarker = (
            load_pose_model()
        )

    except Exception as e:

        st.error(
            f"Unable to load gait models: {e}"
        )

        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        st.error(
            "Unable to access the webcam."
        )

        return

    angles = deque(
        maxlen=300
    )

    frame_placeholder = st.empty()

    status_placeholder = st.empty()

    progress_placeholder = st.empty()

    required_frames = 40

    start_time = time.time()

    timeout = 20

    try:

        while True:

            success, frame = (
                cap.read()
            )

            if not success:

                break

            frame = cv2.flip(
                frame,
                1
            )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=(
                    mp.ImageFormat.SRGB
                ),
                data=rgb
            )

            result = (
                pose_landmarker.detect(
                    mp_image
                )
            )

            if result.pose_landmarks:

                landmarks = (
                    result.pose_landmarks[0]
                )

                hip = landmarks[24]

                knee = landmarks[26]

                ankle = landmarks[28]

                if (
                    hip.visibility >= 0.50
                    and knee.visibility >= 0.50
                    and ankle.visibility >= 0.50
                ):

                    angle = calculate_angle(
                        [hip.x, hip.y],
                        [knee.x, knee.y],
                        [ankle.x, ankle.y]
                    )

                    if angle is not None:

                        angles.append(
                            angle
                        )

                        cv2.putText(
                            frame,
                            f"Knee: {angle:.1f} deg",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

            frame_placeholder.image(
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                ),
                channels="RGB"
            )

            current_frames = len(
                angles
            )

            progress = min(
                current_frames
                / required_frames,
                1.0
            )

            progress_placeholder.progress(
                progress
            )

            status_placeholder.write(
                f"Valid frames: "
                f"{current_frames}/"
                f"{required_frames}"
            )

            if current_frames >= required_frames:

                break

            if (
                time.time()
                - start_time
                > timeout
            ):

                break

    finally:

        cap.release()

    if len(angles) < required_frames:

        st.error(
            "Not enough valid pose frames were captured. "
            "Please make sure your full body and right leg "
            "are visible to the camera."
        )

        return

    features = extract_features(
        angles
    )

    feature_order = [
        "knee_rom",
        "knee_mean",
        "knee_std",
        "knee_max_flex",
        "knee_min_flex",
        "knee_median"
    ]

    X = np.array([
        features[name]
        for name in feature_order
    ]).reshape(
        1,
        -1
    )

    try:

        probability = float(
            gait_model
            .predict_proba(X)[0][1]
        )

        prediction = int(
            gait_model.predict(X)[0]
        )

    except Exception as e:

        st.error(
            f"Gait prediction failed: {e}"
        )

        return

    st.session_state[
        "camera_prediction"
    ] = probability

    st.session_state[
        "camera_features"
    ] = features

    st.success(
        "Camera gait assessment completed."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Gait OA-Pattern Score",
            f"{probability * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Model Output",
            (
                "OA-pattern"
                if prediction == 1
                else "Non-OA-pattern"
            )
        )

    st.subheader(
        "Extracted Gait Features"
    )

    for name, value in features.items():

        st.write(
            f"**{name}:** {value:.2f}"
        )

    st.warning(
        "This is an experimental gait-model output. "
        "It represents an OA-pattern screening signal "
        "from the trained research model and is not a "
        "personal diagnostic probability."
    )


def gait_module():

    st.title("🚶 Camera Gait Analysis")

    st.write(
        "MediaPipe estimates pose landmarks from the "
        "camera, while the trained XGBoost model evaluates "
        "the extracted gait features."
    )

    if not XGBOOST_PATH.exists():

        st.error(
            "XGBoost gait model not found."
        )

        return

    if not POSE_MODEL_PATH.exists():

        st.error(
            "MediaPipe pose model not found."
        )

        return

    st.info(
        "Stand far enough from the camera so that your "
        "body and legs remain visible."
    )

    if st.button(
        "📷 Start Camera Assessment",
        type="primary"
    ):

        run_camera_assessment()

    probability = st.session_state.get(
        "camera_prediction"
    )

    features = st.session_state.get(
        "camera_features"
    )

    if (
        probability is not None
        and features is not None
    ):

        st.divider()

        st.subheader(
            "Latest Gait Result"
        )

        st.metric(
            "Gait OA-Pattern Score",
            f"{probability * 100:.2f}%"
        )