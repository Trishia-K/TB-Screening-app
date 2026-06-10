# TB Screening App — Edge AI Demo
**Kobumanzi Trishia**

A desktop application that uses a MobileNetV2 deep learning model to screen chest X-rays for tuberculosis and it runs offline.

## What This App Does

1. You upload a chest X-ray image (JPG or PNG)
2. The app runs it through an AI model (MobileNetV2)
3. It shows you: TB Positive or TB Negative + confidence score


---

## Setup — Do This Once

### Step 1 — Make sure Python is installed
Open VS Code terminal and type: python --version

### Step 2 — Open the project folder in VS Code
File → Open Folder → select the `tb_screening_app` folder

### Step 3 — Open the terminal in VS Code
Terminal → New Terminal

### Step 4 — Install required packages
```
pip install pillow numpy
```
Then install TensorFlow:
```
pip install tensorflow-cpu
```

### Step 5 — Download and build the AI model
```
python download_model.py
```
This creates a `model/tb_model.tflite` file.
Wait for it to finish — it will say "Setup complete!"

### Step 6 — Run the app
```
python app.py
```


## Where to Get Sample X-Rays for Testing

Download free TB chest X-ray images from:
- Montgomery County Dataset:
  https://openi.nlm.nih.gov/faq#collection
- Kaggle TB dataset:
  https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen

---

## Project Structure

```
tb_screening_app/
├── app.py              ← main application (run this)
├── download_model.py   ← downloads/builds the AI model (run once)
├── requirements.txt    ← list of packages needed
├── README.md           ← this file
└── model/
    └── tb_model.tflite ← AI model file (created by download_model.py)
```

---

## Technologies Used

| Tool | Purpose |
|------|---------|
| Python | Programming language |
| Tkinter | Desktop GUI (comes with Python) |
| Pillow (PIL) | Loading and processing images |
| NumPy | Working with image arrays |
| MobileNetV2 | The AI model architecture |
| TensorFlow Lite | Running the model efficiently |

---

## Research Context

This app is the demo prototype for the research proposal:
**"Edge AI Software System for Tuberculosis Screening
in Rural Health Centres in Uganda"**

Submitted to the Department of Computing & Technology,
Uganda Christian University, Easter 2026 Semester.

This is a proof-of-concept. It uses a MobileNetV2 model with
ImageNet pre-trained weights. A fully trained version would
use fine-tuned weights from TB chest X-ray data as described
in the proposal methodology.
