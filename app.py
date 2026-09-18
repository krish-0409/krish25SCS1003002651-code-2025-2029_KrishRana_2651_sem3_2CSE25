"""
Image Recognition System
=========================
A Flask web application that lets users upload an image and get back
a predicted object/category with a confidence score, powered by a
pretrained TensorFlow/Keras deep learning model (MobileNetV2 trained
on ImageNet, 1000 everyday object/animal/scene categories).

Author: Generated for Krish
"""

import os
import io
import traceback
from datetime import datetime

import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)

# --------------------------------------------------------------------------
# App configuration
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB upload limit
IMG_SIZE = 224  # MobileNetV2 expected input size
TOP_K = 3  # how many predictions to return

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# --------------------------------------------------------------------------
# Load the deep learning model ONCE at startup (not per-request)
# --------------------------------------------------------------------------
print("[INFO] Loading MobileNetV2 model (ImageNet weights)...")
model = MobileNetV2(weights="imagenet")
print("[INFO] Model loaded successfully.")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    """Check that the uploaded file has an allowed image extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def preprocess_image_from_bytes(file_bytes: bytes):
    """
    Decode raw image bytes using OpenCV, validate it, resize/convert it
    to the format the model expects, and return a batch-ready numpy array
    plus the original (BGR) image for reference.

    Raises ValueError with a human-readable message on any problem.
    """
    # Decode bytes -> OpenCV BGR image
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError(
            "Could not read the file as an image. It may be corrupted "
            "or not a supported image format."
        )

    h, w = img_bgr.shape[:2]
    if h < 10 or w < 10:
        raise ValueError("Image dimensions are too small to process.")

    # OpenCV loads as BGR -> convert to RGB (models expect RGB)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize to the size MobileNetV2 expects
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # Convert to float32 array and expand to a batch of size 1
    img_array = np.expand_dims(img_resized.astype(np.float32), axis=0)

    # Apply MobileNetV2-specific preprocessing (scales pixel values)
    img_preprocessed = preprocess_input(img_array)

    return img_preprocessed, img_bgr


def humanize_label(label: str) -> str:
    """Turn an ImageNet class label like 'golden_retriever' into 'Golden Retriever'."""
    return label.replace("_", " ").title()


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    """Serve the main upload page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts an uploaded image file, runs it through the classification
    model, and returns JSON with the top predictions and confidence
    scores. Handles invalid uploads and internal errors gracefully.
    """
    try:
        # ---- Validate that a file was actually sent ----
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file part in the request."}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"success": False, "error": "No file was selected."}), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": f"Unsupported file type. Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            }), 400

        filename = secure_filename(file.filename)
        file_bytes = file.read()

        if len(file_bytes) == 0:
            return jsonify({"success": False, "error": "Uploaded file is empty."}), 400

        # ---- Preprocess with OpenCV ----
        try:
            processed_image, original_bgr = preprocess_image_from_bytes(file_bytes)
        except ValueError as ve:
            return jsonify({"success": False, "error": str(ve)}), 400

        # ---- Run prediction ----
        try:
            raw_predictions = model.predict(processed_image, verbose=0)
            decoded = decode_predictions(raw_predictions, top=TOP_K)[0]
        except Exception as pred_err:
            traceback.print_exc()
            return jsonify({
                "success": False,
                "error": "The model failed to generate a prediction for this image."
            }), 500

        results = [
            {
                "label": humanize_label(label),
                "raw_label": label,
                "confidence": round(float(score) * 100, 2),
            }
            for (_, label, score) in decoded
        ]

        # Optionally save the uploaded file (timestamped, to avoid collisions)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{timestamp}_{filename}")
        try:
            cv2.imwrite(save_path, original_bgr)
        except Exception:
            pass  # Saving is a nice-to-have, not critical to the response

        return jsonify({
            "success": True,
            "top_prediction": results[0],
            "predictions": results,
            "image_size": f"{original_bgr.shape[1]}x{original_bgr.shape[0]}",
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "An unexpected server error occurred while processing your image."
        }), 500


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({
        "success": False,
        "error": "File is too large. Maximum allowed size is 8 MB."
    }), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Route not found."}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": "Internal server error."}), 500


if __name__ == "__main__":
    # debug=True is convenient for development; set to False in production
    app.run(host="0.0.0.0", port=5000, debug=True)
