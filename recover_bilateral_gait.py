
from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

SOURCE_ROOT = Path(
    r"D:\sih2026\kneeOA_age_IMU\Marker_data"
)

OUTPUT_DIR = Path(
    r"D:\osteoarthrits\SIH_OA_XGBoost\data\gait"
)

OUTPUT_FILE = OUTPUT_DIR / "gait_marker_cohort_right_leg.csv"
SKIPPED_FILE = OUTPUT_DIR / "skipped_gait_trials.csv"


# ============================================================
# GROUP / LABEL INFORMATION
# ============================================================

# label:
#   0 = Non-OA
#   1 = OA

GROUP_LABELS = {
    "SSub_01": ("Y", 0),
    "SSub_03": ("Y", 0),
    "SSub_04": ("Y", 0),

    "SSub_05": ("OH", 0),
    "SSub_06": ("OH", 0),
    "SSub_07": ("Y", 0),
    "SSub_08": ("OH", 0),
    "SSub_09": ("OH", 0),

    "SSub_10": ("OA", 1),
    "SSub_11": ("OH", 0),
    "SSub_12": ("OA", 1),
    "SSub_13": ("OH", 0),
    "SSub_14": ("Y", 0),
    "SSub_15": ("Y", 0),
    "SSub_16": ("Y", 0),
    "SSub_17": ("Y", 0),
    "SSub_18": ("OH", 0),
    "SSub_19": ("Y", 0),
    "SSub_21": ("OH", 0),
    "SSub_22": ("OA", 1),
    "SSub_23": ("OH", 0),
    "SSub_24": ("OA", 1),
    "SSub_25": ("OA", 1),
    "SSub_26": ("OA", 1),
    "SSub_27": ("OA", 1),
    "SSub_28": ("OA", 1),
    "SSub_29": ("OA", 1),
}


# ============================================================
# REQUIRED MARKERS
# ============================================================

# For right-leg gait extraction:
#
# HIP:
#   Preferred = midpoint(LASIS, RASIS)
#   Fallback  = RASIS
#
# KNEE:
#   MKNEE
#
# ANKLE:
#   MANK

REQUIRED_MARKERS = [
    "RASIS",
    "MKNEE",
    "MANK",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_angle(a, b, c):
    """
    Calculate the internal angle at point b.

    a, b, c are 3D points.

    Returns angle in degrees.
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    if not (
        np.all(np.isfinite(a))
        and np.all(np.isfinite(b))
        and np.all(np.isfinite(c))
    ):
        return np.nan

    vector_1 = a - b
    vector_2 = c - b

    norm_1 = np.linalg.norm(vector_1)
    norm_2 = np.linalg.norm(vector_2)

    if norm_1 == 0 or norm_2 == 0:
        return np.nan

    cosine_value = np.dot(vector_1, vector_2) / (norm_1 * norm_2)

    # Prevent floating-point errors
    cosine_value = np.clip(cosine_value, -1.0, 1.0)

    internal_angle = np.degrees(
        np.arccos(cosine_value)
    )

    return internal_angle


def get_point(row, marker):
    """
    Return marker XYZ position from one TRC row.

    Returns None if any coordinate is missing/invalid.
    """

    columns = [
        f"{marker}_X",
        f"{marker}_Y",
        f"{marker}_Z",
    ]

    if not all(column in row.index for column in columns):
        return None

    values = row[columns].to_numpy(dtype=float)

    if not np.all(np.isfinite(values)):
        return None

    return values


def midpoint(a, b):
    """
    Return midpoint of two 3D points.
    """

    return (np.asarray(a) + np.asarray(b)) / 2.0


# ============================================================
# TRC READER
# ============================================================

def read_trc(path):
    """
    Read a TRC file and return marker dataframe.

    The function detects the marker names from the TRC header
    and creates columns such as:

        LASIS_X
        LASIS_Y
        LASIS_Z
        RASIS_X
        RASIS_Y
        RASIS_Z
        ...

    """

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if len(lines) < 6:
        raise ValueError("TRC file is too short")

    # --------------------------------------------------------
    # Find the marker-name header line.
    #
    # Standard TRC structure:
    #
    # line 4 -> marker names
    # line 5 -> X1 Y1 Z1 X2 Y2 Z2 ...
    # --------------------------------------------------------

    marker_line_index = None

    for i, line in enumerate(lines):
        if (
            "Frame#" in line
            and "Time" in line
            and i + 1 < len(lines)
        ):
            marker_line_index = i
            break

    if marker_line_index is None:
        raise ValueError("Could not find TRC marker header")

    marker_line = lines[marker_line_index].rstrip("\n").split("\t")

    # Remove empty fields
    marker_line = [
        x.strip()
        for x in marker_line
        if x.strip() != ""
    ]

    # --------------------------------------------------------
    # Marker names normally begin after Frame# and Time.
    # --------------------------------------------------------

    markers = []

    for item in marker_line:
        if item in ["Frame#", "Frame", "Time"]:
            continue

        # Ignore coordinate labels if present
        if item in ["X", "Y", "Z"]:
            continue

        markers.append(item)

    # --------------------------------------------------------
    # Remove accidental duplicates while preserving order
    # --------------------------------------------------------

    clean_markers = []

    for marker in markers:
        if marker not in clean_markers:
            clean_markers.append(marker)

    markers = clean_markers

    # --------------------------------------------------------
    # Locate numeric data.
    # --------------------------------------------------------

    data_start = marker_line_index + 2

    rows = []

    for line in lines[data_start:]:

        line = line.strip()

        if not line:
            continue

        parts = line.split("\t")

        # Remove empty trailing fields
        parts = [
            p.strip()
            for p in parts
            if p.strip() != ""
        ]

        if len(parts) < 2:
            continue

        try:
            frame_number = int(float(parts[0]))
            time_value = float(parts[1])
        except ValueError:
            continue

        values = parts[2:]

        row = {
            "Frame": frame_number,
            "Time": time_value,
        }

        for i, marker in enumerate(markers):

            base = i * 3

            if base + 2 < len(values):

                try:
                    x = float(values[base])
                    y = float(values[base + 1])
                    z = float(values[base + 2])
                except ValueError:
                    x = np.nan
                    y = np.nan
                    z = np.nan

            else:
                x = np.nan
                y = np.nan
                z = np.nan

            row[f"{marker}_X"] = x
            row[f"{marker}_Y"] = y
            row[f"{marker}_Z"] = z

        rows.append(row)

    if not rows:
        raise ValueError("No numeric TRC frames found")

    return pd.DataFrame(rows)


# ============================================================
# PROCESS ONE TRIAL
# ============================================================

def process_trial(path, subject, group, label):
    """
    Extract right-leg knee-angle features from one TRC trial.
    """

    df = read_trc(path)

    # --------------------------------------------------------
    # Check required markers
    # --------------------------------------------------------

    missing = []

    for marker in REQUIRED_MARKERS:

        required_columns = [
            f"{marker}_X",
            f"{marker}_Y",
            f"{marker}_Z",
        ]

        for column in required_columns:
            if column not in df.columns:
                missing.append(column)

    if missing:
        return None, (
            f"Missing required marker columns: "
            f"{', '.join(missing)}"
        )

    # --------------------------------------------------------
    # Calculate knee flexion angle for every frame
    # --------------------------------------------------------

    flexion_angles = []

    for _, row in df.iterrows():

        # ----------------------------------------------------
        # HIP
        #
        # Preferred:
        #     midpoint(LASIS, RASIS)
        #
        # If LASIS is completely/partially missing:
        #     use RASIS directly.
        #
        # This specifically allows trials such as:
        # SSub_15 / Trial07_OS.trc
        # ----------------------------------------------------

        lasis = get_point(row, "LASIS")
        rasis = get_point(row, "RASIS")

        if lasis is not None and rasis is not None:
            hip = midpoint(lasis, rasis)

        elif rasis is not None:
            hip = rasis

        else:
            continue

        # ----------------------------------------------------
        # RIGHT KNEE
        # ----------------------------------------------------

        knee = get_point(row, "MKNEE")

        # ----------------------------------------------------
        # RIGHT ANKLE
        # ----------------------------------------------------

        ankle = get_point(row, "MANK")

        if knee is None or ankle is None:
            continue

        # ----------------------------------------------------
        # Internal anatomical angle
        # ----------------------------------------------------

        internal_angle = calculate_angle(
            hip,
            knee,
            ankle
        )

        if not np.isfinite(internal_angle):
            continue

        # ----------------------------------------------------
        # Convert to flexion convention used by camera model
        #
        # Flexion = 180 - internal angle
        # ----------------------------------------------------

        flexion = 180.0 - internal_angle

        if not np.isfinite(flexion):
            continue

        flexion_angles.append(flexion)

    # --------------------------------------------------------
    # Convert to numpy array
    # --------------------------------------------------------

    angles = np.asarray(
        flexion_angles,
        dtype=float
    )

    # Remove any accidental invalid values
    angles = angles[np.isfinite(angles)]

    # --------------------------------------------------------
    # Need enough frames
    #
    # Camera pipeline uses >=60 frames.
    # Keep same threshold here.
    # --------------------------------------------------------

    if len(angles) < 60:

        return None, (
            f"No valid knee-angle frames "
            f"(valid={len(angles)})"
        )

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    knee_rom = np.max(angles) - np.min(angles)

    knee_mean = np.mean(angles)

    knee_std = np.std(angles)

    knee_max_flex = np.max(angles)

    knee_min_flex = np.min(angles)

    knee_median = np.median(angles)

    # --------------------------------------------------------
    # Return feature row
    # --------------------------------------------------------

    result = {
        "subject": subject,
        "trial": path.name,
        "group": group,
        "label": label,

        "knee_rom": knee_rom,
        "knee_mean": knee_mean,
        "knee_std": knee_std,
        "knee_max_flex": knee_max_flex,
        "knee_min_flex": knee_min_flex,
        "knee_median": knee_median,

        "n_frames": len(angles),
    }

    return result, None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RECOVER RIGHT-LEG GAIT FEATURES FROM TRC")
    print("=" * 70)
    print()
    print("Hip   : midpoint(LASIS, RASIS)")
    print("        fallback = RASIS if LASIS unavailable")
    print("Knee  : MKNEE")
    print("Ankle : MANK")
    print("Angle : 180 - internal angle")
    print()

    # --------------------------------------------------------
    # Check source folder
    # --------------------------------------------------------

    if not SOURCE_ROOT.exists():

        raise FileNotFoundError(
            f"Source directory not found:\n"
            f"{SOURCE_ROOT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find TRC files
    # --------------------------------------------------------

    trc_files = sorted(
        SOURCE_ROOT.rglob("*.trc")
    )

    print(
        f"TRC files found: {len(trc_files)}"
    )

    results = []
    skipped = []

    # --------------------------------------------------------
    # Process every TRC
    # --------------------------------------------------------

    for path in trc_files:

        subject = path.parent.name
        filename_lower = path.name.lower()

        # ----------------------------------------------------
        # Ignore static calibration trials
        # ----------------------------------------------------

        if "static" in filename_lower:
            continue

        # Only process gait trials
        if "trial" not in filename_lower:
            continue

        # ----------------------------------------------------
        # Check subject label
        # ----------------------------------------------------

        if subject not in GROUP_LABELS:

            skipped.append({
                "subject": subject,
                "trial": path.name,
                "reason": "Subject not present in GROUP_LABELS",
            })

            continue

        group, label = GROUP_LABELS[subject]

        # ----------------------------------------------------
        # Process
        # ----------------------------------------------------

        try:

            result, reason = process_trial(
                path,
                subject,
                group,
                label
            )

        except Exception as e:

            result = None
            reason = (
                f"{type(e).__name__}: {e}"
            )

        # ----------------------------------------------------
        # Save result or skip
        # ----------------------------------------------------

        if result is not None:

            results.append(result)

        else:

            print(
                f"[SKIP] {subject} / {path.name} "
                f"→ {reason}"
            )

            skipped.append({
                "subject": subject,
                "trial": path.name,
                "reason": reason,
            })

    # ========================================================
    # CREATE OUTPUT DATAFRAME
    # ========================================================

    columns = [
        "subject",
        "trial",
        "group",
        "label",
        "knee_rom",
        "knee_mean",
        "knee_std",
        "knee_max_flex",
        "knee_min_flex",
        "knee_median",
        "n_frames",
    ]

    output_df = pd.DataFrame(
        results,
        columns=columns
    )

    # --------------------------------------------------------
    # Sort consistently
    # --------------------------------------------------------

    if not output_df.empty:

        output_df = output_df.sort_values(
            ["subject", "trial"]
        ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save main CSV
    # --------------------------------------------------------

    output_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Save skipped-trial report
    # --------------------------------------------------------

    skipped_df = pd.DataFrame(
        skipped,
        columns=[
            "subject",
            "trial",
            "reason",
        ]
    )

    skipped_df.to_csv(
        SKIPPED_FILE,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"Rows saved: {len(output_df)}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Label distribution
    # --------------------------------------------------------

    print()
    print("Label distribution:")

    if not output_df.empty:
        print(
            output_df["label"].value_counts()
        )
    else:
        print("No rows")

    # --------------------------------------------------------
    # Group distribution
    # --------------------------------------------------------

    print()
    print("Group distribution:")

    if not output_df.empty:
        print(
            output_df["group"].value_counts()
        )
    else:
        print("No rows")

    # --------------------------------------------------------
    # Feature summary
    # --------------------------------------------------------

    print()
    print("Feature summary:")

    feature_columns = [
        "knee_rom",
        "knee_mean",
        "knee_std",
        "knee_max_flex",
        "knee_min_flex",
        "knee_median",
    ]

    if not output_df.empty:

        print(
            output_df[feature_columns].describe()
        )

    else:

        print("No features available")

    # --------------------------------------------------------
    # Frame-count summary
    # --------------------------------------------------------

    print()
    print("Frame-count summary:")

    if not output_df.empty:

        print(
            output_df["n_frames"].describe()
        )

    # --------------------------------------------------------
    # Skipped trials
    # --------------------------------------------------------

    print()
    print(
        f"Skipped trials: {len(skipped_df)}"
    )

    print(
        f"Skipped-trial report: {SKIPPED_FILE}"
    )

    if not skipped_df.empty:

        for _, row in skipped_df.iterrows():

            print(
                f"  {row['subject']} / "
                f"{row['trial']} → "
                f"{row['reason']}"
            )

    print()
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
