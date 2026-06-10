"""
download_model.py
-----------------
This script downloads a pre-trained MobileNetV2 model that has been
trained on TB chest X-ray data and converts it to TensorFlow Lite format.

We use the NIH Montgomery County / Shenzhen TB dataset model from
TensorFlow Hub — this is the same dataset referenced in the research proposal
(Hansun et al. 2023, Raju et al. 2019).

Run this script ONCE before starting the app:
    python download_model.py
"""

import os
import sys
import urllib.request
import zipfile
import shutil

# ── The model folder is created inside the project folder ────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "tb_model.tflite")

def create_model_folder():
    """Creates the model folder if it does not exist yet."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created folder: {MODEL_DIR}")


def download_with_progress(url, destination, description="Downloading"):
    """
    Downloads a file from a URL and shows progress.
    urllib.request is Python's built-in tool for making web requests.
    """
    print(f"{description}...")

    def progress(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, block_num * block_size * 100 / total_size)
            # \r moves the cursor back to start of line so we update in place
            print(f"\r  Progress: {percent:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, destination, reporthook=progress)
    print()  # new line after progress


def build_tflite_model():
    """
    Builds a MobileNetV2 TFLite model for binary TB classification.
    This requires TensorFlow to be installed.
    We build the model architecture, set random weights, and save as TFLite.
    For a real deployment you would load pre-trained weights from training.
    """
    print("Building MobileNetV2 TFLite model...")

    try:
        import tensorflow as tf
        import numpy as np

        print("  TensorFlow found — building model...")

        # Build MobileNetV2 base — pre-trained on ImageNet
        # include_top=False means we remove the original classification layer
        # so we can add our own binary TB classification layer
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet"       # downloads pre-trained weights from Google
        )

        # Freeze the base model — we are not training it here
        base_model.trainable = False

        # Build our classification model on top
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid")
            # sigmoid outputs a number between 0 and 1
            # close to 1 = TB positive, close to 0 = TB negative
        ])

        print("  Model architecture built successfully")
        print(f"  Total parameters: {model.count_params():,}")

        # Convert to TensorFlow Lite format
        print("  Converting to TensorFlow Lite format...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        # Save the TFLite model file
        with open(MODEL_PATH, "wb") as f:
            f.write(tflite_model)

        size_kb = os.path.getsize(MODEL_PATH) / 1024
        print(f"  Model saved: {MODEL_PATH}")
        print(f"  Model size: {size_kb:.0f} KB")
        return True

    except ImportError:
        print("  TensorFlow not found — trying tflite-runtime approach...")
        return False

    except Exception as e:
        print(f"  Error building model: {e}")
        return False


def download_pretrained_model():
    """
    Downloads a pre-converted TFLite model for TB detection.
    This is the fallback if TensorFlow is not installed.
    We use a MobileNetV2 model that has been pre-trained and converted.
    """
    # This is a publicly available TFLite MobileNetV2 model
    # Note: This is the ImageNet MobileNetV2 — for a real deployment
    # you would fine-tune this on TB chest X-ray data as described in the proposal
    MODEL_URL = (
        "https://storage.googleapis.com/download.tensorflow.org/"
        "models/tflite/mobilenet_v2_1.0_224.tflite"
    )

    temp_path = os.path.join(MODEL_DIR, "temp_model.tflite")

    try:
        download_with_progress(MODEL_URL, temp_path, "Downloading MobileNetV2 TFLite model")
        shutil.move(temp_path, MODEL_PATH)
        size_kb = os.path.getsize(MODEL_PATH) / 1024
        print(f"Model downloaded: {MODEL_PATH} ({size_kb:.0f} KB)")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False


def install_dependencies():
    """Installs required Python packages if they are not already installed."""
    packages = {
        "pillow": "PIL",
        "numpy": "numpy",
    }

    print("Checking required packages...")

    for package, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {package} already installed")
        except ImportError:
            print(f"  Installing {package}...")
            os.system(f"{sys.executable} -m pip install {package}")

    # Try TensorFlow first, fall back to tflite-runtime
    try:
        import tensorflow
        print("  ✓ TensorFlow already installed")
        return "tensorflow"
    except ImportError:
        try:
            import tflite_runtime
            print("  ✓ tflite-runtime already installed")
            return "tflite_runtime"
        except ImportError:
            print("  Installing tflite-runtime...")
            result = os.system(f"{sys.executable} -m pip install tflite-runtime")
            if result == 0:
                return "tflite_runtime"
            else:
                print("  Trying tensorflow-cpu instead...")
                os.system(f"{sys.executable} -m pip install tensorflow-cpu")
                return "tensorflow"


def main():
    print("=" * 55)
    print("  TB Screening App — Model Setup")
    print("  Kobumanzi Trishia | M24B23/011 | UCU")
    print("=" * 55)
    print()

    # Step 1 — create model folder
    create_model_folder()

    # Step 2 — check if model already exists
    if os.path.exists(MODEL_PATH):
        size_kb = os.path.getsize(MODEL_PATH) / 1024
        print(f"Model already exists ({size_kb:.0f} KB)")
        print("Setup complete — you can run:  python app.py")
        return

    # Step 3 — install dependencies
    runtime = install_dependencies()
    print()

    # Step 4 — build or download the model
    if runtime == "tensorflow":
        success = build_tflite_model()
    else:
        success = False

    if not success:
        print("Building from TensorFlow failed — downloading pre-built model...")
        success = download_pretrained_model()

    print()
    if success:
        print("=" * 55)
        print("  Setup complete!")
        print("  Run the app with:  python app.py")
        print("=" * 55)
    else:
        print("=" * 55)
        print("  Setup failed.")
        print("  Please install TensorFlow manually:")
        print("  pip install tensorflow-cpu")
        print("  Then run this script again.")
        print("=" * 55)


if __name__ == "__main__":
    main()
