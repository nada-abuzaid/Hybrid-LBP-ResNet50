import os
import uuid

import cv2
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from skimage.feature import local_binary_pattern


# ============================================================
# Application Configuration
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "model",
    "paper_fingerprint_hybrid_lbp_resnet50_final.keras"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# Constants
# ============================================================

DOCUMENT_WIDTH = 1200
DOCUMENT_HEIGHT = 1600

PATCH_SIZE = 224
PATCHES_PER_IMG = 8

CANDIDATES_PER_PATCH = 100

MIN_TEXTURE_STD = 8
TEXT_PENALTY = 50

THRESHOLD = 0.5

# Prevent nearly identical patches from being selected
MIN_PATCH_DISTANCE = 180

# Fixed seed for reproducible web inference
RANDOM_SEED = 42


# ============================================================
# LBP Settings
# ============================================================

LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"

LBP_FEATURE_SIZE = LBP_POINTS + 2


# ============================================================
# Allowed Extensions
# ============================================================

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "bmp",
    "tif",
    "tiff"
}


# ============================================================
# Load Model
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("\n===== MODEL INFORMATION =====")
print("Model loaded successfully.")
print("Model name:", model.name)

print("\nModel inputs:")
for inp in model.inputs:
    print(
        f"  {inp.name}: {inp.shape}"
    )

print(
    "Model output:",
    model.output_shape
)


# ============================================================
# Validate File
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# Document Preprocessing
# ============================================================

def preprocess_document(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        (
            DOCUMENT_WIDTH,
            DOCUMENT_HEIGHT
        ),
        interpolation=cv2.INTER_LINEAR
    )

    return gray


# ============================================================
# Text Mask
# ============================================================

def build_text_mask(gray):

    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU
    )

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=2
    )

    return mask


# ============================================================
# LBP Feature Extraction
# ============================================================

def extract_lbp_features(gray_patch):

    gray_patch = np.asarray(
        gray_patch,
        dtype=np.uint8
    )

    lbp = local_binary_pattern(
        gray_patch,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method=LBP_METHOD
    )

    histogram, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(
            0,
            LBP_FEATURE_SIZE + 1
        ),
        range=(
            0,
            LBP_FEATURE_SIZE
        )
    )

    histogram = histogram.astype(
        np.float32
    )

    histogram /= (
        histogram.sum()
        + 1e-7
    )

    return histogram


# ============================================================
# Distance Between Candidate Patches
# ============================================================

def is_far_enough(
    x,
    y,
    selected_locations
):

    for (
        selected_x,
        selected_y
    ) in selected_locations:

        distance = np.sqrt(
            (x - selected_x) ** 2
            +
            (y - selected_y) ** 2
        )

        if (
            distance
            < MIN_PATCH_DISTANCE
        ):
            return False

    return True


# ============================================================
# Extract One Diverse Texture Patch
# ============================================================

def extract_texture_patch(
    gray,
    mask,
    rng,
    selected_locations
):

    height, width = gray.shape

    if (
        width < PATCH_SIZE
        or height < PATCH_SIZE
    ):
        raise ValueError(
            "Document is too small for patch extraction."
        )

    candidates = []

    # --------------------------------------------------------
    # Exactly 100 candidates per required patch
    # --------------------------------------------------------

    for _ in range(
        CANDIDATES_PER_PATCH
    ):

        x = int(
            rng.integers(
                0,
                width - PATCH_SIZE + 1
            )
        )

        y = int(
            rng.integers(
                0,
                height - PATCH_SIZE + 1
            )
        )

        patch = gray[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ]

        patch_mask = mask[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ]

        texture_std = float(
            patch.std()
        )

        if (
            texture_std
            < MIN_TEXTURE_STD
        ):
            continue

        text_ratio = float(
            np.mean(
                patch_mask > 0
            )
        )

        score = (
            texture_std
            -
            text_ratio * TEXT_PENALTY
        )

        candidates.append(
            (
                score,
                x,
                y,
                patch.copy()
            )
        )

    if not candidates:
        return None, None

    # Best candidate first
    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    # --------------------------------------------------------
    # Prefer a candidate far away from previous patches
    # --------------------------------------------------------

    for (
        score,
        x,
        y,
        patch
    ) in candidates:

        if is_far_enough(
            x,
            y,
            selected_locations
        ):

            return patch, (x, y)

    # --------------------------------------------------------
    # Fallback:
    # use best candidate if no sufficiently distant one exists
    # --------------------------------------------------------

    best = candidates[0]

    return (
        best[3],
        (
            best[1],
            best[2]
        )
    )


# ============================================================
# Convert One Patch to Model Inputs
# ============================================================

def prepare_patch_for_model(
    gray_patch
):

    if gray_patch.shape != (
        PATCH_SIZE,
        PATCH_SIZE
    ):

        gray_patch = cv2.resize(
            gray_patch,
            (
                PATCH_SIZE,
                PATCH_SIZE
            ),
            interpolation=cv2.INTER_LINEAR
        )

    gray_patch = np.clip(
        gray_patch,
        0,
        255
    ).astype(
        np.uint8
    )

    # LBP branch
    lbp_features = extract_lbp_features(
        gray_patch
    )

    # ResNet50 branch
    #
    # IMPORTANT:
    # Keep original 0-255 range.
    # ResNet50 preprocess_input() is inside the model.
    rgb_patch = cv2.cvtColor(
        gray_patch,
        cv2.COLOR_GRAY2RGB
    ).astype(
        np.float32
    )

    return (
        rgb_patch,
        lbp_features
    )


# ============================================================
# Direct Prediction for a 224x224 Patch
# ============================================================

def predict_single_patch_direct(
    gray_patch
):

    rgb_patch, lbp_features = (
        prepare_patch_for_model(
            gray_patch
        )
    )

    prediction = model.predict(
        {
            "image_input":
                np.expand_dims(
                    rgb_patch,
                    axis=0
                ),

            "lbp_input":
                np.expand_dims(
                    lbp_features,
                    axis=0
                )
        },
        verbose=0
    ).reshape(-1)[0]

    return float(
        prediction
    )


# ============================================================
# Extract Document Patches
# ============================================================

def extract_document_patches(
    image_path
):

    image = cv2.imread(
        image_path,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "The uploaded image could not be read."
        )

    original_height, original_width = (
        image.shape[:2]
    )

    # ========================================================
    # SPECIAL CASE:
    # Uploaded file is already a saved 224x224 texture patch
    # ========================================================

    if (
        original_height == PATCH_SIZE
        and original_width == PATCH_SIZE
    ):

        gray_patch = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        rgb_patch, lbp_features = (
            prepare_patch_for_model(
                gray_patch
            )
        )

        return (
            np.expand_dims(
                rgb_patch,
                axis=0
            ).astype(np.float32),

            np.expand_dims(
                lbp_features,
                axis=0
            ).astype(np.float32)
        )

    # ========================================================
    # Full document mode
    # ========================================================

    gray = preprocess_document(
        image
    )

    text_mask = build_text_mask(
        gray
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    image_inputs = []
    lbp_inputs = []

    selected_locations = []

    # ========================================================
    # Extract up to 8 spatially diverse patches
    # ========================================================

    for patch_index in range(
        PATCHES_PER_IMG
    ):

        patch, location = (
            extract_texture_patch(
                gray,
                text_mask,
                rng,
                selected_locations
            )
        )

        if (
            patch is None
            or location is None
        ):
            continue

        selected_locations.append(
            location
        )

        rgb_patch, lbp_features = (
            prepare_patch_for_model(
                patch
            )
        )

        image_inputs.append(
            rgb_patch
        )

        lbp_inputs.append(
            lbp_features
        )

        print(
            f"Patch {patch_index + 1}: "
            f"x={location[0]}, "
            f"y={location[1]}, "
            f"std={patch.std():.2f}"
        )

    if len(image_inputs) == 0:

        raise ValueError(
            "No suitable texture patches "
            "could be extracted from this document."
        )

    images_array = np.stack(
        image_inputs,
        axis=0
    ).astype(
        np.float32
    )

    lbp_array = np.stack(
        lbp_inputs,
        axis=0
    ).astype(
        np.float32
    )

    return (
        images_array,
        lbp_array
    )


# ============================================================
# Predict Document
# ============================================================

def predict_document(
    image_path
):

    image_patches, lbp_features = (
        extract_document_patches(
            image_path
        )
    )

    predictions = model.predict(
        {
            "image_input":
                image_patches,

            "lbp_input":
                lbp_features
        },
        verbose=0
    ).reshape(-1)

    # ========================================================
    # Same document aggregation used in methodology:
    # arithmetic mean
    # ========================================================

    forged_probability = float(
        np.mean(
            predictions
        )
    )

    authentic_probability = (
        1.0
        - forged_probability
    )

    if (
        forged_probability
        >= THRESHOLD
    ):

        result = "Forged"
        confidence = (
            forged_probability
        )

    else:

        result = "Authentic"
        confidence = (
            authentic_probability
        )

    return {
        "result":
            result,

        "confidence":
            confidence * 100.0,

        "forged_probability":
            forged_probability * 100.0,

        "authentic_probability":
            authentic_probability * 100.0,

        "patches_count":
            len(predictions),

        "patch_predictions":
            predictions.tolist()
    }


# ============================================================
# Flask Route
# ============================================================

@app.route(
    "/",
    methods=[
        "GET",
        "POST"
    ]
)
def index():

    result = None
    confidence = None
    patches_count = None
    image_url = None
    error = None

    if (
        request.method
        == "POST"
    ):

        if (
            "document"
            not in request.files
        ):

            error = (
                "Please select "
                "a document image."
            )

            return render_template(
                "index.html",
                error=error
            )

        file = request.files[
            "document"
        ]

        if (
            file.filename
            == ""
        ):

            error = (
                "Please select "
                "a document image."
            )

            return render_template(
                "index.html",
                error=error
            )

        if not allowed_file(
            file.filename
        ):

            error = (
                "Supported formats: "
                "PNG, JPG, JPEG, "
                "BMP, TIF and TIFF."
            )

            return render_template(
                "index.html",
                error=error
            )

        original_filename = (
            secure_filename(
                file.filename
            )
        )

        extension = (
            original_filename
            .rsplit(
                ".",
                1
            )[1]
            .lower()
        )

        unique_filename = (
            f"{uuid.uuid4().hex}"
            f".{extension}"
        )

        file_path = os.path.join(
            app.config[
                "UPLOAD_FOLDER"
            ],
            unique_filename
        )

        file.save(
            file_path
        )

        try:

            prediction = (
                predict_document(
                    file_path
                )
            )

            result = prediction[
                "result"
            ]

            confidence = prediction[
                "confidence"
            ]

            patches_count = prediction[
                "patches_count"
            ]

            image_url = (
                f"uploads/"
                f"{unique_filename}"
            )

            print(
                "\n"
                "=============================="
            )

            print(
                "===== PREDICTION RESULT ====="
            )

            print(
                "=============================="
            )

            print(
                "Result:",
                prediction[
                    "result"
                ]
            )

            print(
                "Confidence:",
                f"{prediction['confidence']:.2f}%"
            )

            print(
                "Authentic probability:",
                f"{prediction['authentic_probability']:.2f}%"
            )

            print(
                "Forged probability:",
                f"{prediction['forged_probability']:.2f}%"
            )

            print(
                "Patches used:",
                prediction[
                    "patches_count"
                ]
            )

            print(
                "Patch predictions:",
                np.round(
                    prediction[
                        "patch_predictions"
                    ],
                    4
                ).tolist()
            )

        except Exception as exception:

            print(
                "Prediction error:",
                exception
            )

            error = str(
                exception
            )

    return render_template(
        "index.html",

        result=result,

        confidence=confidence,

        patches_count=
            patches_count,

        image_url=image_url,

        error=error
    )


# ============================================================
# Run Application
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )