import cv2
import mediapipe as mp
import math


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    """
    Calculate angle ABC using three points.
    """
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    print("Check that your webcam is connected and not being used by another application.")
    raise SystemExit


with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    print("Camera started successfully.")
    print("Stand where your full body is visible.")
    print("Press Q to quit.")

    while True:

        success, frame = cap.read()

        if not success:
            print("ERROR: Could not read camera frame.")
            break

        # Mirror the camera view
        frame = cv2.flip(frame, 1)

        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose detection
        results = pose.process(rgb_frame)

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            # MediaPipe landmarks
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

            h, w, _ = frame.shape

            # Pixel coordinates
            lh = (int(left_hip.x * w), int(left_hip.y * h))
            lk = (int(left_knee.x * w), int(left_knee.y * h))
            la = (int(left_ankle.x * w), int(left_ankle.y * h))

            rh = (int(right_hip.x * w), int(right_hip.y * h))
            rk = (int(right_knee.x * w), int(right_knee.y * h))
            ra = (int(right_ankle.x * w), int(right_ankle.y * h))

            # Knee angles
            left_knee_angle = calculate_angle(lh, lk, la)
            right_knee_angle = calculate_angle(rh, rk, ra)

            # Draw skeleton
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # Display angles
            cv2.putText(
                frame,
                f"Left Knee: {left_knee_angle:.1f} deg",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Right Knee: {right_knee_angle:.1f} deg",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "No person detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        cv2.imshow("OsteoCare - Camera Gait Analysis", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()