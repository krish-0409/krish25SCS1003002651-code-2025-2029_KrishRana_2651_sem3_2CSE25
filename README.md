# AI Image Recognition System

A Flask web application that classifies uploaded images using a pretrained
deep learning model (MobileNetV2, trained on ImageNet — 1000 object/animal/
scene categories), with OpenCV handling image decoding and preprocessing.

## Features
- Drag-and-drop or click-to-upload image interface
- Image preprocessing with OpenCV (decode, color conversion, resize)
- Classification with TensorFlow/Keras MobileNetV2
- Top-3 predictions with confidence scores, shown with an animated bar
- Robust error handling: invalid files, oversized files, corrupted images,
  unsupported formats, and model/prediction failures all return clear
  messages instead of crashing
- Responsive UI (works on desktop and mobile)

## Project Structure
```
image_recognition_app/
├── app.py                 # Flask backend + model loading + /predict endpoint
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Main page
├── static/
│   ├── style.css          # Styling
│   └── script.js          # Upload/preview/API logic
└── uploads/                # Uploaded images are saved here (auto-created)
```

## Setup

1. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   python app.py
   ```
   The first run will download the MobileNetV2 ImageNet weights
   (~14 MB) automatically — this requires an internet connection once.

4. **Open in your browser**
   ```
   http://localhost:5000
   ```

## How It Works

1. The user selects or drags an image onto the page.
2. The image is sent to the `/predict` Flask endpoint as `multipart/form-data`.
3. OpenCV (`cv2.imdecode`) decodes the raw bytes into an image array,
   validates it, converts BGR→RGB, and resizes it to 224×224 (the input
   size MobileNetV2 expects).
4. The array is preprocessed with Keras's `preprocess_input` and passed to
   the MobileNetV2 model for inference.
5. `decode_predictions` converts the raw output vector into human-readable
   labels with probabilities; the top 3 are returned as JSON.
6. The frontend displays the top prediction with a confidence bar plus two
   runner-up guesses.

## Using Your Own Custom-Trained Model

This project ships with MobileNetV2 pretrained on ImageNet so it works out
of the box with no training step. To swap in your own trained model
(e.g. for a custom set of classes like specific products or plant species):

1. Train and save your model as `model.h5` or a SavedModel directory.
2. In `app.py`, replace:
   ```python
   model = MobileNetV2(weights="imagenet")
   ```
   with:
   ```python
   model = tf.keras.models.load_model("path/to/your_model.h5")
   ```
3. Replace `decode_predictions` with your own logic that maps output
   indices to your class names (e.g. a `class_names` list you index with
   `np.argmax`).
4. Adjust `IMG_SIZE` and any preprocessing in `preprocess_image_from_bytes`
   to match what your model was trained on.

## Notes
- Max upload size is 8 MB (configurable via `MAX_CONTENT_LENGTH` in `app.py`).
- Allowed file types: PNG, JPG, JPEG, BMP, WEBP.
- Set `debug=False` in `app.py` before deploying to production, and serve
  with a production WSGI server (e.g. `gunicorn app:app`) rather than the
  Flask dev server.
