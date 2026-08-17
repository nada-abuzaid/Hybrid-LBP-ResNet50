import os
from pathlib import Path

# Force CPU — Streamlit Cloud does not need CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from skimage.feature import local_binary_pattern


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "paper_fingerprint_hybrid_lbp_resnet50_final.keras"
)

IMG_SIZE = 224


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Hybrid LBP–ResNet50",
    page_icon="📄",
    layout="centered",
)


# =========================================================
# MODEL
# =========================================================

@st.cache_resource(show_spinner="Loading forensic model...")
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


# =========================================================
# LBP PREPROCESSING
# =========================================================

def extract_lbp(image_bytes):

    # Convert uploaded bytes to OpenCV image
    arr = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        arr,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            "Could not read the uploaded image."
        )

    # Resize to model input size
    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    # Local Binary Pattern
    lbp = local_binary_pattern(
        image,
        P=8,
        R=1,
        method="uniform"
    )

    # Normalize LBP
    lbp = cv2.normalize(
        lbp,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Convert grayscale → RGB
    lbp = cv2.cvtColor(
        lbp,
        cv2.COLOR_GRAY2RGB
    )

    # Normalize to [0,1]
    lbp = lbp.astype(np.float32) / 255.0

    return lbp


# =========================================================
# PREDICTION
# =========================================================

def predict_document(image_bytes):

    model = load_model()

    # Extract LBP features
    processed = extract_lbp(image_bytes)

    # Add batch dimension
    x = np.expand_dims(
        processed,
        axis=0
    )

    # Prediction
    prediction = model.predict(
        x,
        verbose=0
    )

    probability = float(
        np.squeeze(prediction)
    )

    # Binary classification
    forged_probability = probability
    authentic_probability = 1.0 - probability

    if forged_probability >= 0.5:

        label = "FORGED"
        confidence = forged_probability

    else:

        label = "AUTHENTIC"
        confidence = authentic_probability

    return (
        label,
        confidence,
        authentic_probability,
        forged_probability
    )


# =========================================================
# UI
# =========================================================

st.title(
    "Hybrid LBP–ResNet50"
)

st.subheader(
    "Forensic Document Authentication"
)

st.write(
    """
Upload a scanned document image to analyze
its paper-texture fingerprint using Local
Binary Pattern (LBP) features and a
ResNet50-based deep learning model.
"""
)


# =========================================================
# UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a document image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "tif",
        "tiff"
    ]
)


if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    image = Image.open(
        uploaded_file
    )

    st.image(
        image,
        caption="Uploaded document",
        use_container_width=True
    )


    # =====================================================
    # ANALYZE
    # =====================================================

    if st.button(
        "Analyze Document",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Analyzing document fingerprint..."
            ):

                (
                    label,
                    confidence,
                    authentic_probability,
                    forged_probability
                ) = predict_document(
                    image_bytes
                )


            # =================================================
            # RESULT
            # =================================================

            if label == "AUTHENTIC":

                st.success(
                    f"Result: {label}"
                )

            else:

                st.error(
                    f"Result: {label}"
                )


            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Authentic Probability",
                    f"{authentic_probability * 100:.2f}%"
                )


            with col2:

                st.metric(
                    "Forged Probability",
                    f"{forged_probability * 100:.2f}%"
                )


            st.info(
                "This result is an automated model prediction "
                "and should be considered forensic assistance, "
                "not definitive proof."
            )


        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )