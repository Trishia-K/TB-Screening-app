"""
TB Screening App — Edge AI Demo
Author: Kobumanzi Trishia
Description: A desktop application that uses a MobileNetV2 deep learning
model to screen chest X-rays for signs of tuberculosis, completely offline.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import os
import threading

# We try to import tflite. If it is not installed yet, we show a helpful message.
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        # On some systems TensorFlow Lite comes bundled with full TensorFlow
        import tensorflow as tf
        tflite = tf.lite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False


# ─── COLOURS AND FONTS ────────────────────────────────────────────────────────
# These match the red and blue colour scheme from the research poster

BG_DARK      = "#0D1B2A"   # dark navy background
BG_CARD      = "#1B2D42"   # slightly lighter for cards
RED          = "#C0392B"   # red from poster
BLUE         = "#2E86C1"   # blue from poster
GREEN        = "#1ABC9C"   # for positive result highlight
WHITE        = "#FFFFFF"
LIGHT_GREY   = "#BDC3C7"
DARK_GREY    = "#2C3E50"
YELLOW       = "#F39C12"   # for warning/pending state

FONT_TITLE   = ("Calibri", 20, "bold")
FONT_HEADING = ("Calibri", 13, "bold")
FONT_BODY    = ("Calibri", 11)
FONT_SMALL   = ("Calibri", 9)
FONT_RESULT  = ("Calibri", 28, "bold")
FONT_CONF    = ("Calibri", 14)


# ─── MODEL PATH ───────────────────────────────────────────────────────────────
# This is where the app looks for the downloaded model file.
# The download_model.py script saves it here automatically.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "tb_model.tflite")


# ─── MAIN APPLICATION CLASS ───────────────────────────────────────────────────

class TBScreeningApp:
    """
    This is the main class for the application.
    A class is like a blueprint — it holds all the code for the app in one place.
    When Python runs TBScreeningApp(), it creates one instance of the app.
    """

    def __init__(self, root):
        """
        __init__ is called automatically when the app starts.
        'root' is the main window that Tkinter gives us.
        Everything the app needs is set up here.
        """
        self.root = root
        self.root.title("TB Screening System — Edge AI Demo")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("900x700")
        self.root.minsize(800, 600)

        # These variables will hold the image and the loaded model
        self.current_image_path = None   # path to the X-ray file the user uploads
        self.interpreter = None          # the TFLite model interpreter
        self.photo_image = None          # the image displayed on screen

        # Load the model when the app starts
        self._load_model()

        # Build the user interface
        self._build_ui()


    def _load_model(self):
        """
        This method loads the TFLite model from the model folder.
        It is called once when the app starts.
        If the model file is not found, the app still opens but
        shows a message telling the user to run the download script.
        """
        if not TFLITE_AVAILABLE:
            self.model_status = "missing_library"
            return

        if not os.path.exists(MODEL_PATH):
            self.model_status = "missing_model"
            return

        try:
            # Create an interpreter — this is the object that runs the model
            if TFLITE_AVAILABLE and hasattr(tflite, 'Interpreter'):
                self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
            else:
                self.interpreter = tflite.lite.Interpreter(model_path=MODEL_PATH)

            # Allocate tensors — this prepares the model to receive input data
            self.interpreter.allocate_tensors()

            # Get the input and output details so we know what shape to feed in
            self.input_details  = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            self.model_status = "ready"

        except Exception as e:
            self.model_status = f"error: {str(e)}"


    def _build_ui(self):
        """
        This method builds everything you see on screen.
        It uses Tkinter widgets — Frame, Label, Button, Canvas — to create the layout.
        """

        # ── TOP HEADER BAR ─────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=RED, height=70)
        header.pack(fill="x")
        header.pack_propagate(False)  # prevents the frame from shrinking

        tk.Label(
            header,
            text="TB Screening System",
            font=FONT_TITLE,
            bg=RED,
            fg=WHITE
        ).pack(side="left", padx=20, pady=15)

        tk.Label(
            header,
            text="Offlin System",
            font=FONT_SMALL,
            bg=RED,
            fg="#E8E8E8"
        ).pack(side="right", padx=20, pady=15)

        # Model status badge — shows in the header
        status_text, status_colour = self._get_status_display()
        self.status_badge = tk.Label(
            header,
            text=status_text,
            font=FONT_SMALL,
            bg=status_colour,
            fg=WHITE,
            padx=8,
            pady=4
        )
        self.status_badge.pack(side="right", padx=10, pady=18)

        # ── MAIN BODY ──────────────────────────────────────────────────────────
        # The body is split into two columns: left for the image, right for results
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=16, pady=12)

        # Left column — image upload and display
        left = tk.Frame(body, bg=BG_CARD, bd=0)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        # Right column — results and info
        right = tk.Frame(body, bg=BG_DARK)
        right.pack(side="right", fill="both", expand=True, padx=(8, 0))

        # ── LEFT COLUMN: X-RAY DISPLAY ─────────────────────────────────────────
        tk.Label(
            left,
            text="Chest X-Ray",
            font=FONT_HEADING,
            bg=BG_CARD,
            fg=BLUE
        ).pack(pady=(12, 4))

        # The canvas is where the X-ray image will be displayed
        # Canvas is like a drawing board — we can put images on it
        self.canvas = tk.Canvas(
            left,
            bg="#111111",
            highlightthickness=1,
            highlightbackground=BLUE
        )
        self.canvas.pack(fill="both", expand=True, padx=12, pady=4)

        # Placeholder text shown before any image is uploaded
        self.canvas_placeholder = self.canvas.create_text(
            200, 200,
            text="No X-ray loaded\n\nClick 'Upload X-Ray' to begin",
            fill=LIGHT_GREY,
            font=FONT_BODY,
            justify="center"
        )

        # Upload button — blue, at the bottom of the left column
        tk.Button(
            left,
            text="Upload X-Ray",
            font=FONT_HEADING,
            bg=BLUE,
            fg=WHITE,
            activebackground="#1A5276",
            activeforeground=WHITE,
            relief="flat",
            cursor="hand2",
            padx=20,
            pady=10,
            command=self._upload_image
        ).pack(pady=12)

        # ── RIGHT COLUMN: RESULTS ──────────────────────────────────────────────

        # File name display
        self.filename_label = tk.Label(
            right,
            text="No file selected",
            font=FONT_SMALL,
            bg=BG_DARK,
            fg=LIGHT_GREY,
            wraplength=280
        )
        self.filename_label.pack(pady=(0, 8))

        # Result card — this big box shows the screening result
        self.result_card = tk.Frame(right, bg=BG_CARD, bd=0)
        self.result_card.pack(fill="x", pady=4)

        tk.Label(
            self.result_card,
            text="Screening Result",
            font=FONT_HEADING,
            bg=BG_CARD,
            fg=LIGHT_GREY
        ).pack(pady=(14, 4))

        # This label shows TB POSITIVE or TB NEGATIVE in large text
        self.result_label = tk.Label(
            self.result_card,
            text="—",
            font=FONT_RESULT,
            bg=BG_CARD,
            fg=LIGHT_GREY
        )
        self.result_label.pack(pady=8)

        # This label shows the confidence percentage
        self.confidence_label = tk.Label(
            self.result_card,
            text="",
            font=FONT_CONF,
            bg=BG_CARD,
            fg=LIGHT_GREY
        )
        self.confidence_label.pack(pady=(0, 14))

        # Screen button — RED, runs the model
        self.screen_button = tk.Button(
            right,
            text="Screen for TB",
            font=FONT_HEADING,
            bg=RED,
            fg=WHITE,
            activebackground="#922B21",
            activeforeground=WHITE,
            relief="flat",
            cursor="hand2",
            padx=20,
            pady=12,
            state="disabled",  # disabled until an image is uploaded
            command=self._run_screening
        )
        self.screen_button.pack(fill="x", pady=8)

        # Progress label — shows "Analysing..." while the model is running
        self.progress_label = tk.Label(
            right,
            text="",
            font=FONT_BODY,
            bg=BG_DARK,
            fg=YELLOW
        )
        self.progress_label.pack(pady=4)

        # Info box — explains what the result means
        info_frame = tk.Frame(right, bg=BG_CARD)
        info_frame.pack(fill="x", pady=8)

        tk.Label(
            info_frame,
            text="About This System",
            font=FONT_HEADING,
            bg=BG_CARD,
            fg=BLUE
        ).pack(pady=(10, 4))

        info_text = (
            "This offline system uses a MobileNetV2 deep learning\n"
            "model to analyse chest X-rays for signs of\n"
            "tuberculosis.\n\n"
            "This is a research prototype. Results should\n"
            "always be confirmed by a trained health worker."
        )
        tk.Label(
            info_frame,
            text=info_text,
            font=FONT_SMALL,
            bg=BG_CARD,
            fg=LIGHT_GREY,
            justify="left",
            wraplength=260
        ).pack(padx=14, pady=(0, 12))

        # Reset button
        tk.Button(
            right,
            text="Reset",
            font=FONT_BODY,
            bg=DARK_GREY,
            fg=WHITE,
            activebackground="#1A252F",
            relief="flat",
            cursor="hand2",
            padx=12,
            pady=6,
            command=self._reset
        ).pack(pady=4)

        # ── BOTTOM STATUS BAR ──────────────────────────────────────────────────
        status_bar = tk.Frame(self.root, bg=DARK_GREY, height=28)
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)

        self.status_bar_label = tk.Label(
            status_bar,
            text="Ready — upload a chest X-ray to begin",
            font=FONT_SMALL,
            bg=DARK_GREY,
            fg=LIGHT_GREY
        )
        self.status_bar_label.pack(side="left", padx=12, pady=4)

        tk.Label(
            status_bar,
            text="Kobumanzi Trishia  |  M24B23/011  |  UCU",
            font=FONT_SMALL,
            bg=DARK_GREY,
            fg=LIGHT_GREY
        ).pack(side="right", padx=12, pady=4)

        # Show a warning if model or library is missing
        if self.model_status != "ready":
            self._show_setup_warning()


    def _get_status_display(self):
        """Returns the text and colour for the model status badge."""
        if self.model_status == "ready":
            return "● Model Ready", GREEN
        elif self.model_status == "missing_model":
            return "⚠ Model Not Found", YELLOW
        elif self.model_status == "missing_library":
            return "⚠ Library Missing", RED
        else:
            return "⚠ Error", RED


    def _show_setup_warning(self):
        """Shows a popup if something is not set up correctly."""
        if self.model_status == "missing_library":
            msg = (
                "TensorFlow Lite is not installed.\n\n"
                "Please open a terminal and run:\n\n"
                "    pip install tflite-runtime\n\n"
                "Then restart the app."
            )
        elif self.model_status == "missing_model":
            msg = (
                "The AI model file was not found.\n\n"
                "Please run the download script first:\n\n"
                "    python download_model.py\n\n"
                "Then restart the app."
            )
        else:
            msg = f"Error loading model:\n{self.model_status}"

        messagebox.showwarning("Setup Required", msg)


    def _upload_image(self):
        """
        Opens a file dialog so the user can choose an X-ray image.
        Accepts JPG, PNG, and BMP formats.
        Once a file is chosen, it displays it on the canvas.
        """
        # filetypes tells the dialog what file types to accept
        file_path = filedialog.askopenfilename(
            title="Select Chest X-Ray Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )

        # If the user cancelled the dialog, file_path will be empty
        if not file_path:
            return

        self.current_image_path = file_path

        # Show the filename in the label
        filename = os.path.basename(file_path)
        self.filename_label.config(text=f"File: {filename}", fg=WHITE)

        # Load and display the image on the canvas
        self._display_image(file_path)

        # Enable the Screen button now that we have an image
        self.screen_button.config(state="normal")

        # Reset any previous result
        self._clear_result()

        self._update_status(f"X-ray loaded: {filename} — click 'Screen for TB' to analyse")


    def _display_image(self, path):
        """
        Loads an image from the given path and displays it on the canvas.
        PIL (Pillow) is used to open and resize the image.
        ImageTk converts it to a format Tkinter can show.
        """
        try:
            # Open the image using Pillow
            img = Image.open(path)

            # Convert to greyscale for display — X-rays are greyscale
            img = img.convert("L")

            # Get the canvas size so we can fit the image inside it
            self.canvas.update()  # make sure canvas has rendered its size
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()

            # Use a fallback size if canvas hasn't rendered yet
            if canvas_w < 10:
                canvas_w = 380
            if canvas_h < 10:
                canvas_h = 380

            # Resize the image to fit the canvas while keeping proportions
            img.thumbnail((canvas_w - 20, canvas_h - 20), Image.LANCZOS)

            # Convert to a format Tkinter can display
            self.photo_image = ImageTk.PhotoImage(img)

            # Clear the canvas and show the image centred
            self.canvas.delete("all")
            cx = canvas_w // 2
            cy = canvas_h // 2
            self.canvas.create_image(cx, cy, image=self.photo_image, anchor="center")

        except Exception as e:
            messagebox.showerror("Image Error", f"Could not load image:\n{str(e)}")


    def _run_screening(self):
        """
        This is called when the user clicks 'Screen for TB'.
        It runs the model in a separate thread so the app does not freeze
        while the model is processing. Threading means doing two things
        at the same time — the UI stays responsive while the model runs.
        """
        if not self.current_image_path:
            return

        if self.model_status != "ready":
            self._show_setup_warning()
            return

        # Disable the button and show progress message
        self.screen_button.config(state="disabled", text="Analysing...")
        self.progress_label.config(text="Analysing chest X-ray...")
        self.result_label.config(text="...", fg=YELLOW)
        self.confidence_label.config(text="")
        self._update_status("Running AI model — please wait...")

        # Run the model in a background thread
        # This prevents the app window from freezing
        thread = threading.Thread(target=self._analyse_image)
        thread.daemon = True  # thread closes when the app closes
        thread.start()


    def _analyse_image(self):
        """
        This method does the actual AI analysis.
        It runs in a background thread.

        Steps:
        1. Open the image with Pillow
        2. Resize it to 224x224 pixels (what MobileNetV2 expects)
        3. Normalise pixel values from 0-255 to 0.0-1.0
        4. Add a batch dimension (the model expects a batch of images)
        5. Feed it into the TFLite model
        6. Read the output probability
        7. Update the UI with the result
        """
        try:
            # Step 1 — open the image
            img = Image.open(self.current_image_path)

            # Step 2 — convert to RGB (MobileNetV2 expects 3 colour channels)
            img = img.convert("RGB")

            # Step 3 — resize to 224x224 pixels
            img = img.resize((224, 224), Image.LANCZOS)

            # Step 4 — convert to a numpy array of numbers
            # numpy is a library for working with arrays of numbers efficiently
            img_array = np.array(img, dtype=np.float32)

            # Step 5 — normalise: divide by 255 to get values between 0 and 1
            img_array = img_array / 255.0

            # Step 6 — add a batch dimension
            # The model expects input shape (1, 224, 224, 3)
            # meaning: 1 image, 224 pixels tall, 224 pixels wide, 3 colour channels
            img_array = np.expand_dims(img_array, axis=0)

            # Step 7 — set the input tensor (feed the image into the model)
            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                img_array
            )

            # Step 8 — run the model
            self.interpreter.invoke()

            # Step 9 — get the output (a probability between 0 and 1)
            output = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )

            # The output is a 2D array like [[0.87]] — we get the single number
            probability = float(output[0][0])

            # Step 10 — update the UI with the result
            # We use root.after to safely update the UI from the background thread
            self.root.after(0, self._show_result, probability)

        except Exception as e:
            self.root.after(0, self._show_error, str(e))


    def _show_result(self, probability):
        """
        Updates the UI with the screening result.
        probability is a number between 0 and 1.
        A probability above 0.5 means the model thinks TB is present.
        """
        # Convert probability to a percentage
        confidence = probability * 100

        # Decide: TB Positive or TB Negative
        # The threshold is 0.5 — above this means TB signs detected
        if probability >= 0.5:
            result_text   = "TB POSITIVE"
            result_colour = RED
            conf_text     = f"Confidence: {confidence:.1f}%"
            status_msg    = "Result: TB Positive signs detected — please refer to a specialist"
        else:
            result_text   = "TB NEGATIVE"
            result_colour = GREEN
            conf_text     = f"Confidence: {(100 - confidence):.1f}%"
            status_msg    = "Result: No TB signs detected"

        # Update the result label
        self.result_label.config(text=result_text, fg=result_colour)
        self.result_card.config(bg=BG_CARD)

        # Update the confidence label
        self.confidence_label.config(
            text=conf_text,
            fg=result_colour,
            font=FONT_CONF
        )

        # Re-enable the button
        self.screen_button.config(state="normal", text="Screen for TB")
        self.progress_label.config(text="")

        self._update_status(status_msg)


    def _show_error(self, error_msg):
        """Called if something goes wrong during analysis."""
        self.result_label.config(text="ERROR", fg=YELLOW)
        self.confidence_label.config(text=error_msg, fg=YELLOW, font=FONT_SMALL)
        self.screen_button.config(state="normal", text="Screen for TB")
        self.progress_label.config(text="")
        self._update_status(f"Error during analysis: {error_msg}")


    def _clear_result(self):
        """Resets the result display."""
        self.result_label.config(text="—", fg=LIGHT_GREY)
        self.confidence_label.config(text="")
        self.progress_label.config(text="")


    def _reset(self):
        """Resets the entire app back to the starting state."""
        self.current_image_path = None
        self.photo_image = None
        self.canvas.delete("all")
        self.canvas.create_text(
            200, 200,
            text="No X-ray loaded\n\nClick 'Upload X-Ray' to begin",
            fill=LIGHT_GREY,
            font=FONT_BODY,
            justify="center"
        )
        self.filename_label.config(text="No file selected", fg=LIGHT_GREY)
        self.screen_button.config(state="disabled", text="Screen for TB")
        self._clear_result()
        self._update_status("Reset — upload a chest X-ray to begin")


    def _update_status(self, message):
        """Updates the status bar at the bottom of the screen."""
        self.status_bar_label.config(text=message)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
# This is where Python starts running when you open the file.
# It creates the main Tkinter window, creates the app, and starts the loop.

if __name__ == "__main__":
    root = tk.Tk()
    app = TBScreeningApp(root)
    root.mainloop()   # mainloop keeps the window open and listens for clicks
