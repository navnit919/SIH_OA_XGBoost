
import cv2
import math
import joblib
import numpy as np
import mediapipe as mp
from collections import deque


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = r"D:\osteoarthrits\SIH_OA_XGBoost\models\gait\oa_xgboost_gait_model_v2.pkl"
POSE_MODEL = r"D:\osteoarthrits\SIH_OA_XGBoost\camera\pose_landmarker_lite.task"


# ============================================================
# LOAD XGBOOST MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

print("=" * 70)
print("LIVE OA GAIT PREDICTION")
print("=" * 70)

print("Model loaded:")
print(MODEL_PATH)

print("\nExpected features:")
print([
    "knee_rom",
    "knee_mean",
    "knee_std",
    "knee_max_flex",
    "knee_min_flex",
    "knee_median"
])


# ============================================================
# MEDIAPIPE POSE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=POSE_MODEL
    ),
    running_mode=VisionRunningMode.IMAGE
)

landmarker = PoseLandmarker.create_from_options(options)


# ============================================================
# RIGHT LEG LANDMARKS
# ============================================================

RIGHT_HIP = 24
RIGHT_KNEE = 26
RIGHT_ANKLE = 28


# ============================================================
# KNEE ANGLE CALCULATION
# ============================================================

def calculate_knee_angle(hip, knee, ankle):

    hip_point = np.array([
        hip.x,
        hip.y,
        hip.z
    ], dtype=float)

    knee_point = np.array([
        knee.x,
        knee.y,
        knee.z
    ], dtype=float)

    ankle_point = np.array([
        ankle.x,
        ankle.y,
        ankle.z
    ], dtype=float)

    vector_1 = hip_point - knee_point
    vector_2 = ankle_point - knee_point

    denominator = (
        np.linalg.norm(vector_1) *
        np.linalg.norm(vector_2)
    )

    if denominator == 0:
        return None

    cosine_angle = (
        np.dot(vector_1, vector_2)
        / denominator
    )

    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    internal_angle = math.degrees(
        math.acos(cosine_angle)
    )

    # Same flexion definition used during training
    flexion = 180.0 - internal_angle

    return float(flexion)


# ============================================================
# FEATURE CALCULATION
# ============================================================

def calculate_features(angles):

    angles = np.asarray(
        angles,
        dtype=float
    )

    if len(angles) < 20:
        return None

    knee_rom = (
        np.max(angles)
        - np.min(angles)
    )

    knee_mean = np.mean(angles)

    knee_std = np.std(angles)

    knee_max_flex = np.max(angles)

    knee_min_flex = np.min(angles)

    knee_median = np.median(angles)

    features = np.array([[
        knee_rom,
        knee_mean,
        knee_std,
        knee_max_flex,
        knee_min_flex,
        knee_median
    ]], dtype=float)

    return features


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("\nERROR: Could not open webcam.")

    landmarker.close()

    raise SystemExit


# Try to use a reasonable camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


print("\nCamera opened successfully.")
print("Stand with your RIGHT side visible.")
print("Walk naturally through the camera view.")
print("Collect at least 60 valid frames.")
print("Press Q to quit.")
print("Press R to reset.\n")


# ============================================================
# STORAGE
# ============================================================

angle_buffer = deque(
    maxlen=300
)

prediction_probability = None
prediction_label = "Collecting data..."

current_features = None

last_printed_frame_count = -1


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read camera frame.")

        break


    # Mirror display
    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # MEDIAPIPE INPUT
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(
        mp_image
    )


    # --------------------------------------------------------
    # LANDMARK PROCESSING
    # --------------------------------------------------------

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        if len(landmarks) >= 29:

            hip = landmarks[RIGHT_HIP]
            knee = landmarks[RIGHT_KNEE]
            ankle = landmarks[RIGHT_ANKLE]

            visibility_values = [
                hip.visibility,
                knee.visibility,
                ankle.visibility
            ]

            if min(visibility_values) >= 0.50:

                angle = calculate_knee_angle(
                    hip,
                    knee,
                    ankle
                )

                if angle is not None:

                    # Reject impossible numerical values
                    if 0 <= angle <= 180:

                        angle_buffer.append(
                            angle
                        )


    # ========================================================
    # PREDICTION
    # ========================================================

    if len(angle_buffer) >= 60:

        features = calculate_features(
            angle_buffer
        )

        if features is not None:

            current_features = features[0]

            probability = model.predict_proba(
                features
            )[0][1]

            prediction_probability = (
                float(probability) * 100.0
            )

            if probability >= 0.50:

                prediction_label = (
                    "OA RISK INDICATION"
                )

            else:

                prediction_label = (
                    "LOWER OA RISK INDICATION"
                )


            # ------------------------------------------------
            # PRINT FEATURES TO TERMINAL
            # ------------------------------------------------

            # Print once whenever another 30 frames are added
            current_count = len(angle_buffer)

            if (
                current_count != last_printed_frame_count
                and current_count % 30 == 0
            ):

                print("\n" + "-" * 60)
                print("LIVE GAIT FEATURES")
                print("-" * 60)

                print(
                    f"Knee ROM      : "
                    f"{current_features[0]:.2f}"
                )

                print(
                    f"Knee Mean     : "
                    f"{current_features[1]:.2f}"
                )

                print(
                    f"Knee Std      : "
                    f"{current_features[2]:.2f}"
                )

                print(
                    f"Max Flexion   : "
                    f"{current_features[3]:.2f}"
                )

                print(
                    f"Min Flexion   : "
                    f"{current_features[4]:.2f}"
                )

                print(
                    f"Knee Median   : "
                    f"{current_features[5]:.2f}"
                )

                print(
                    f"OA Probability: "
                    f"{prediction_probability:.2f}%"
                )

                print(
                    f"Result        : "
                    f"{prediction_label}"
                )

                print("-" * 60)

                last_printed_frame_count = current_count


    # ========================================================
    # CAMERA DISPLAY
    # ========================================================

    cv2.putText(
        frame,
        "OA GAIT SCREENING",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Valid frames: {len(angle_buffer)}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # FEATURES ON SCREEN
    # --------------------------------------------------------

    if current_features is not None:

        feature_names = [
            "Knee ROM",
            "Knee Mean",
            "Knee Std",
            "Max Flex",
            "Min Flex",
            "Median"
        ]

        y_position = 125

        for name, value in zip(
            feature_names,
            current_features
        ):

            cv2.putText(
                frame,
                f"{name}: {value:.2f}",
                (30, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            y_position += 27


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"OA probability: "
            f"{prediction_probability:.1f}%",
            (30, y_position + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            prediction_label,
            (30, y_position + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2
        )


    else:

        cv2.putText(
            frame,
            "Collecting gait data...",
            (30, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )


    # --------------------------------------------------------
    # INSTRUCTIONS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "Q = Quit | R = Reset",
        (30, frame.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # SHOW CAMERA
    # --------------------------------------------------------

    cv2.imshow(
        "OA Gait Screening",
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break

    elif key == ord("r"):

        angle_buffer.clear()

        prediction_probability = None

        prediction_label = (
            "Collecting data..."
        )

        current_features = None

        last_printed_frame_count = -1

        print("\nMeasurement reset.")


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()

print("\nCamera session ended.")
