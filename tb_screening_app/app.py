"""
TB Screening App v2 — Edge AI Demo
Author: Kobumanzi Trishia | M24B23/011 | UCU
Features added in v2:
  - Patient details form (name, ID, age, gender, notes)
  - Screening history saved to CSV file
  - History tab showing all past screenings in a table
  - Print / Save report as a text file that can be printed
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk
import os
import threading
import csv
import datetime

# Try importing TensorFlow Lite
try:
    import tensorflow as tf
    tflite = tf.lite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False

# ── COLOURS ──────────────────────────────────────────────────────────────────
BG_DARK    = "#0D1B2A"
BG_CARD    = "#1B2D42"
BG_MID     = "#162232"
RED        = "#C0392B"
BLUE       = "#2E86C1"
GREEN      = "#1ABC9C"
WHITE      = "#FFFFFF"
LIGHT_GREY = "#BDC3C7"
DARK_GREY  = "#2C3E50"
YELLOW     = "#F39C12"

# ── FONTS ─────────────────────────────────────────────────────────────────────
FONT_TITLE   = ("Calibri", 18, "bold")
FONT_HEADING = ("Calibri", 12, "bold")
FONT_BODY    = ("Calibri", 10)
FONT_SMALL   = ("Calibri", 9)
FONT_RESULT  = ("Calibri", 24, "bold")
FONT_CONF    = ("Calibri", 13)

# ── FILE PATHS ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "tb_model.tflite")
HISTORY_CSV = os.path.join(BASE_DIR, "screening_history.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


class TBScreeningApp:

    def __init__(self, root):
        self.root = root
        self.root.title("TB Screening System — Edge AI  |  UCU")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1100x750")
        self.root.minsize(1000, 680)

        # These will be set when a patient is screened
        self.current_image_path = None
        self.interpreter        = None
        self.photo_image        = None
        self.last_result        = None

        # Create folders and files needed
        os.makedirs(REPORTS_DIR, exist_ok=True)
        self._init_csv()

        # Load the AI model
        self._load_model()

        # Build the full UI
        self._build_ui()


    # =========================================================================
    # CSV HISTORY FUNCTIONS
    # These functions handle saving and reading the screening history
    # =========================================================================

    def _init_csv(self):
        """
        Creates the CSV file if it does not exist yet.
        CSV stands for Comma Separated Values — it is a simple spreadsheet
        that can be opened in Excel. Each row is one screening.
        """
        if not os.path.exists(HISTORY_CSV):
            with open(HISTORY_CSV, "w", newline="") as f:
                writer = csv.writer(f)
                # Write the header row — these become the column names
                writer.writerow([
                    "Date", "Time", "Patient Name", "Patient ID",
                    "Age", "Gender", "Result", "Confidence (%)", "Notes"
                ])

    def _save_to_history(self, record):
        """
        Appends one row to the CSV file.
        'record' is a dictionary with all the patient and result details.
        We open the file in append mode ("a") so we add to the end
        without deleting what is already there.
        """
        with open(HISTORY_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                record["date"],       record["time"],
                record["name"],       record["patient_id"],
                record["age"],        record["gender"],
                record["result"],     record["confidence"],
                record["notes"]
            ])

    def _load_history(self):
        """
        Reads all rows from the CSV and returns them as a list of dicts.
        csv.DictReader reads each row as a dictionary where the keys
        are the column headers we set in _init_csv.
        """
        rows = []
        if os.path.exists(HISTORY_CSV):
            with open(HISTORY_CSV, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
        return rows


    # =========================================================================
    # MODEL LOADING
    # =========================================================================

    def _load_model(self):
        """
        Loads the TFLite model file from the model/ folder.
        The Interpreter object is what actually runs the model.
        allocate_tensors() prepares memory for the model's input and output.
        get_input_details() tells us what shape the model expects as input.
        get_output_details() tells us what shape the output will be.
        """
        if not TFLITE_AVAILABLE:
            self.model_status = "missing_library"; return
        if not os.path.exists(MODEL_PATH):
            self.model_status = "missing_model"; return
        try:
            self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
            self.interpreter.allocate_tensors()
            self.input_details  = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.model_status   = "ready"
        except Exception as e:
            self.model_status   = f"error: {str(e)}"


    # =========================================================================
    # UI BUILDER
    # =========================================================================

    def _build_ui(self):
        """Builds the full interface. Called once when the app starts."""

        # ── HEADER BAR ────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=RED, height=60)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header, text="TB Screening System",
            font=FONT_TITLE, bg=RED, fg=WHITE
        ).pack(side="left", padx=20, pady=10)

        tk.Label(
            header,
            text="Edge AI  |  Offline  |  Uganda Christian University",
            font=FONT_SMALL, bg=RED, fg="#E8E8E8"
        ).pack(side="right", padx=20)

        status_text, status_colour = self._get_status_display()
        tk.Label(
            header, text=status_text,
            font=FONT_SMALL, bg=status_colour, fg=WHITE,
            padx=8, pady=4
        ).pack(side="right", padx=10, pady=14)

        # ── TABS ──────────────────────────────────────────────────────────────
        # ttk.Notebook is the Tkinter widget that creates tabs.
        # We add two frames (tab_screen and tab_history) to it.
        # Each frame is a separate page that shows when you click its tab.

        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook",     background=BG_DARK, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_CARD, foreground=WHITE,
                         font=FONT_HEADING, padding=[16, 6])
        style.map("TNotebook.Tab",
                  background=[("selected", BLUE)],
                  foreground=[("selected", WHITE)])

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        self.tab_screen  = tk.Frame(self.notebook, bg=BG_DARK)
        self.tab_history = tk.Frame(self.notebook, bg=BG_DARK)

        self.notebook.add(self.tab_screen,  text="  Screen Patient  ")
        self.notebook.add(self.tab_history, text="  Screening History  ")

        self._build_screen_tab()
        self._build_history_tab()

        # ── BOTTOM STATUS BAR ─────────────────────────────────────────────────
        status_bar = tk.Frame(self.root, bg=DARK_GREY, height=26)
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)

        self.status_bar_label = tk.Label(
            status_bar,
            text="Ready — fill in patient details and upload a chest X-ray",
            font=FONT_SMALL, bg=DARK_GREY, fg=LIGHT_GREY
        )
        self.status_bar_label.pack(side="left", padx=12, pady=3)

        tk.Label(
            status_bar,
            text="Kobumanzi Trishia  |  M24B23/011  |  UCU  |  2026",
            font=FONT_SMALL, bg=DARK_GREY, fg=LIGHT_GREY
        ).pack(side="right", padx=12, pady=3)

        if self.model_status != "ready":
            self._show_setup_warning()


    # =========================================================================
    # TAB 1 — SCREEN PATIENT
    # =========================================================================

    def _build_screen_tab(self):

        body = tk.Frame(self.tab_screen, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=12, pady=10)

        # Three side-by-side columns
        left   = tk.Frame(body, bg=BG_CARD, padx=14, pady=10)
        middle = tk.Frame(body, bg=BG_CARD, padx=10, pady=10)
        right  = tk.Frame(body, bg=BG_DARK)

        left.pack(side="left",  fill="y", ipadx=4)
        middle.pack(side="left", fill="both", expand=True, padx=10)
        right.pack(side="right", fill="y",    ipadx=4)

        # ── LEFT COLUMN: PATIENT FORM ──────────────────────────────────────────

        tk.Label(
            left, text="Patient Details",
            font=FONT_HEADING, bg=BG_CARD, fg=BLUE
        ).pack(anchor="w", pady=(0, 10))

        def field(parent, label_text):
            """Helper that creates a labelled text entry field."""
            tk.Label(
                parent, text=label_text,
                font=FONT_SMALL, bg=BG_CARD, fg=LIGHT_GREY
            ).pack(anchor="w", pady=(6, 1))
            e = tk.Entry(
                parent, font=FONT_BODY, bg=BG_MID, fg=WHITE,
                insertbackground=WHITE, relief="flat", width=22
            )
            e.pack(anchor="w", ipady=5)
            return e

        # Patient form fields
        self.entry_name       = field(left, "Patient Name *")
        self.entry_patient_id = field(left, "Patient ID *")
        self.entry_age        = field(left, "Age")

        # Gender dropdown using OptionMenu
        # StringVar holds the selected value — we read it later with .get()
        tk.Label(
            left, text="Gender",
            font=FONT_SMALL, bg=BG_CARD, fg=LIGHT_GREY
        ).pack(anchor="w", pady=(6, 1))
        self.gender_var = tk.StringVar(value="Select")
        opt = tk.OptionMenu(left, self.gender_var,
                            "Select", "Male", "Female", "Other")
        opt.config(font=FONT_BODY, bg=BG_MID, fg=WHITE,
                   activebackground=BLUE, relief="flat", width=19)
        opt["menu"].config(bg=BG_MID, fg=WHITE, font=FONT_BODY)
        opt.pack(anchor="w")

        # Notes text area — Text widget allows multiple lines
        tk.Label(
            left, text="Notes (optional)",
            font=FONT_SMALL, bg=BG_CARD, fg=LIGHT_GREY
        ).pack(anchor="w", pady=(10, 1))
        self.text_notes = tk.Text(
            left, font=FONT_BODY, bg=BG_MID, fg=WHITE,
            insertbackground=WHITE, relief="flat",
            width=22, height=4, wrap="word"
        )
        self.text_notes.pack(anchor="w")

        tk.Frame(left, bg=DARK_GREY, height=1).pack(fill="x", pady=14)

        # Upload button
        tk.Button(
            left, text="Upload X-Ray",
            font=FONT_HEADING, bg=BLUE, fg=WHITE,
            activebackground="#1A5276", relief="flat",
            cursor="hand2", padx=12, pady=8, width=20,
            command=self._upload_image
        ).pack(pady=(0, 6))

        self.filename_label = tk.Label(
            left, text="No file selected",
            font=FONT_SMALL, bg=BG_CARD, fg=LIGHT_GREY, wraplength=180
        )
        self.filename_label.pack(anchor="w")

        # Screen button — disabled until image is loaded
        self.screen_button = tk.Button(
            left, text="Screen for TB",
            font=FONT_HEADING, bg=RED, fg=WHITE,
            activebackground="#922B21", relief="flat",
            cursor="hand2", padx=12, pady=10, width=20,
            state="disabled", command=self._run_screening
        )
        self.screen_button.pack(pady=(10, 4))

        self.progress_label = tk.Label(
            left, text="", font=FONT_SMALL, bg=BG_CARD, fg=YELLOW
        )
        self.progress_label.pack(anchor="w")

        tk.Button(
            left, text="Reset",
            font=FONT_BODY, bg=DARK_GREY, fg=WHITE,
            activebackground="#1A252F", relief="flat",
            cursor="hand2", padx=10, pady=5, width=20,
            command=self._reset
        ).pack(pady=(8, 0))

        # ── MIDDLE COLUMN: X-RAY IMAGE ──────────────────────────────────────

        tk.Label(
            middle, text="Chest X-Ray",
            font=FONT_HEADING, bg=BG_CARD, fg=BLUE
        ).pack(pady=(0, 6))

        # Canvas is like a drawing board — images and shapes go on it
        self.canvas = tk.Canvas(
            middle, bg="#111111",
            highlightthickness=1, highlightbackground=BLUE
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_text(
            200, 200,
            text="No X-ray loaded\n\nClick 'Upload X-Ray' to begin",
            fill=LIGHT_GREY, font=FONT_BODY, justify="center",
            tags="placeholder"
        )

        # ── RIGHT COLUMN: RESULT ────────────────────────────────────────────

        result_card = tk.Frame(right, bg=BG_CARD, padx=14, pady=14)
        result_card.pack(fill="x", pady=(0, 8))

        tk.Label(
            result_card, text="Screening Result",
            font=FONT_HEADING, bg=BG_CARD, fg=LIGHT_GREY
        ).pack(pady=(0, 6))

        # These three labels get updated when a result comes back
        self.result_label = tk.Label(
            result_card, text="—",
            font=FONT_RESULT, bg=BG_CARD, fg=LIGHT_GREY
        )
        self.result_label.pack()

        self.confidence_label = tk.Label(
            result_card, text="",
            font=FONT_CONF, bg=BG_CARD, fg=LIGHT_GREY
        )
        self.confidence_label.pack(pady=(4, 0))

        self.date_label = tk.Label(
            result_card, text="",
            font=FONT_SMALL, bg=BG_CARD, fg=LIGHT_GREY
        )
        self.date_label.pack(pady=(2, 0))

        # Print button — enabled after a result is shown
        self.print_button = tk.Button(
            right, text="Print / Save Report",
            font=FONT_HEADING, bg=DARK_GREY, fg=WHITE,
            activebackground=BLUE, relief="flat",
            cursor="hand2", padx=12, pady=8,
            state="disabled", command=self._print_result
        )
        self.print_button.pack(fill="x", pady=(0, 6))

        # Info box
        info = tk.Frame(right, bg=BG_CARD, padx=12, pady=10)
        info.pack(fill="x")
        tk.Label(
            info, text="About This System",
            font=FONT_HEADING, bg=BG_CARD, fg=BLUE
        ).pack(anchor="w", pady=(0, 4))
        tk.Label(
            info,
            text=(
                "MobileNetV2 deep learning model\n"
                "deployed via TensorFlow Lite.\n\n"
                "Runs completely offline — no\n"
                "internet required at any point.\n\n"
                "Results should always be confirmed\n"
                "by a trained health worker.\n\n"
                "All results saved automatically to\n"
                "screening_history.csv"
            ),
            font=FONT_SMALL, bg=BG_CARD, fg=LIGHT_GREY, justify="left"
        ).pack(anchor="w")


    # =========================================================================
    # TAB 2 — HISTORY
    # =========================================================================

    def _build_history_tab(self):

        top = tk.Frame(self.tab_history, bg=BG_DARK)
        top.pack(fill="x", padx=12, pady=(10, 4))

        tk.Label(
            top, text="Screening History",
            font=FONT_HEADING, bg=BG_DARK, fg=BLUE
        ).pack(side="left")

        tk.Button(
            top, text="Refresh",
            font=FONT_SMALL, bg=BLUE, fg=WHITE,
            relief="flat", cursor="hand2", padx=10, pady=4,
            command=self._refresh_history
        ).pack(side="right")

        tk.Button(
            top, text="Open CSV in Excel",
            font=FONT_SMALL, bg=DARK_GREY, fg=WHITE,
            relief="flat", cursor="hand2", padx=10, pady=4,
            command=self._open_csv
        ).pack(side="right", padx=6)

        # Summary counts label
        self.history_summary = tk.Label(
            self.tab_history, text="",
            font=FONT_SMALL, bg=BG_DARK, fg=LIGHT_GREY
        )
        self.history_summary.pack(anchor="w", padx=12, pady=(0, 4))

        # Table frame with scrollbars
        table_frame = tk.Frame(self.tab_history, bg=BG_DARK)
        table_frame.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        scroll_y = ttk.Scrollbar(table_frame, orient="vertical")
        scroll_y.pack(side="right", fill="y")
        scroll_x = ttk.Scrollbar(table_frame, orient="horizontal")
        scroll_x.pack(side="bottom", fill="x")

        # Style the Treeview (table) to match our dark theme
        style = ttk.Style()
        style.configure("Treeview",
            background=BG_CARD, foreground=WHITE,
            fieldbackground=BG_CARD, font=FONT_BODY, rowheight=26)
        style.configure("Treeview.Heading",
            background=BLUE, foreground=WHITE, font=FONT_HEADING)
        style.map("Treeview", background=[("selected", DARK_GREY)])

        # Treeview is Tkinter's table widget
        # columns= defines the column names
        # show="headings" hides the default first empty column
        columns = ("Date","Time","Patient Name","Patient ID",
                   "Age","Gender","Result","Confidence (%)","Notes")

        self.tree = ttk.Treeview(
            table_frame, columns=columns, show="headings",
            yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set
        )

        widths = {"Date":90,"Time":70,"Patient Name":140,"Patient ID":90,
                  "Age":45,"Gender":65,"Result":110,
                  "Confidence (%)":110,"Notes":200}
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=widths.get(col,100), anchor="w")

        self.tree.pack(fill="both", expand=True)
        scroll_y.config(command=self.tree.yview)
        scroll_x.config(command=self.tree.xview)

        # Row colour tags — positive rows show in red, negative in green
        self.tree.tag_configure("positive", foreground="#E74C3C")
        self.tree.tag_configure("negative", foreground="#1ABC9C")

        self._refresh_history()


    def _refresh_history(self):
        """Clears the table and reloads all rows from the CSV."""
        for row in self.tree.get_children():
            self.tree.delete(row)

        rows     = self._load_history()
        total    = len(rows)
        positive = sum(1 for r in rows if "POSITIVE" in r.get("Result",""))
        negative = total - positive

        for row in rows:
            result = row.get("Result","")
            tag    = "positive" if "POSITIVE" in result else "negative"
            self.tree.insert("", "end", values=(
                row.get("Date",""),    row.get("Time",""),
                row.get("Patient Name",""), row.get("Patient ID",""),
                row.get("Age",""),     row.get("Gender",""),
                row.get("Result",""),  row.get("Confidence (%)",""),
                row.get("Notes","")
            ), tags=(tag,))

        self.history_summary.config(
            text=f"Total screenings: {total}    "
                 f"TB Positive: {positive}    "
                 f"TB Negative: {negative}"
        )


    def _open_csv(self):
        """Opens the CSV file in Excel or the default spreadsheet app."""
        if os.path.exists(HISTORY_CSV):
            os.startfile(HISTORY_CSV)
        else:
            messagebox.showinfo("No History",
                                "No screening history found yet.")


    # =========================================================================
    # IMAGE UPLOAD
    # =========================================================================

    def _upload_image(self):
        """Opens a file dialog and loads the chosen X-ray image."""
        path = filedialog.askopenfilename(
            title="Select Chest X-Ray Image",
            filetypes=[("Image files","*.jpg *.jpeg *.png *.bmp *.tiff"),
                       ("All files","*.*")]
        )
        if not path:
            return

        self.current_image_path = path
        self.filename_label.config(
            text=f"File: {os.path.basename(path)}", fg=WHITE)
        self._display_image(path)
        self.screen_button.config(state="normal")
        self._clear_result()
        self._update_status(f"X-ray loaded: {os.path.basename(path)}")


    def _display_image(self, path):
        """Loads the image with Pillow and draws it on the canvas."""
        try:
            img = Image.open(path).convert("L")  # L = greyscale
            self.canvas.update()
            w = max(self.canvas.winfo_width(),  380)
            h = max(self.canvas.winfo_height(), 380)
            img.thumbnail((w - 20, h - 20), Image.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(
                w // 2, h // 2,
                image=self.photo_image, anchor="center")
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not load image:\n{e}")


    # =========================================================================
    # SCREENING
    # =========================================================================

    def _run_screening(self):
        """Validates the form and starts the model in a background thread."""
        name = self.entry_name.get().strip()
        pid  = self.entry_patient_id.get().strip()

        if not name:
            messagebox.showwarning("Missing", "Please enter the patient name.")
            return
        if not pid:
            messagebox.showwarning("Missing", "Please enter the patient ID.")
            return
        if not self.current_image_path:
            messagebox.showwarning("No Image",
                                   "Please upload a chest X-ray first.")
            return
        if self.model_status != "ready":
            self._show_setup_warning()
            return

        self.screen_button.config(state="disabled", text="Analysing...")
        self.progress_label.config(text="Running AI model...")
        self.result_label.config(text="...", fg=YELLOW)
        self.confidence_label.config(text="")
        self._update_status("Analysing — please wait...")

        # Run model in background so window stays responsive
        t = threading.Thread(target=self._analyse_image)
        t.daemon = True
        t.start()


    def _analyse_image(self):
        """
        Runs the TFLite model. This is the core AI step.
        1. Open and resize image to 224x224 (what MobileNetV2 expects)
        2. Normalise pixels from 0-255 to 0.0-1.0
        3. Add batch dimension: shape becomes (1, 224, 224, 3)
        4. Feed into model with set_tensor
        5. Run model with invoke()
        6. Read output probability with get_tensor
        7. Send result back to the main thread with root.after()
        """
        try:
            img       = Image.open(self.current_image_path).convert("RGB")
            img       = img.resize((224, 224), Image.LANCZOS)
            arr       = np.array(img, dtype=np.float32) / 255.0
            arr       = np.expand_dims(arr, axis=0)          # add batch dim

            self.interpreter.set_tensor(
                self.input_details[0]["index"], arr)
            self.interpreter.invoke()
            output    = self.interpreter.get_tensor(
                self.output_details[0]["index"])
            prob      = float(output[0][0])

            # root.after(0, fn, arg) calls fn(arg) safely on the main thread
            self.root.after(0, self._show_result, prob)
        except Exception as e:
            self.root.after(0, self._show_error, str(e))


    def _show_result(self, probability):
        """Updates the UI, saves to CSV, and enables the print button."""
        confidence = probability * 100
        now        = datetime.datetime.now()

        if probability >= 0.5:
            result_text  = "TB POSITIVE"
            result_color = RED
            conf_text    = f"Confidence: {confidence:.1f}%"
        else:
            result_text  = "TB NEGATIVE"
            result_color = GREEN
            conf_text    = f"Confidence: {(100 - confidence):.1f}%"

        self.result_label.config(text=result_text, fg=result_color)
        self.confidence_label.config(text=conf_text, fg=result_color)
        self.date_label.config(
            text=now.strftime("%d %b %Y  %H:%M"), fg=LIGHT_GREY)

        # Build the record that gets saved to CSV
        record = {
            "date":       now.strftime("%Y-%m-%d"),
            "time":       now.strftime("%H:%M:%S"),
            "name":       self.entry_name.get().strip(),
            "patient_id": self.entry_patient_id.get().strip(),
            "age":        self.entry_age.get().strip(),
            "gender":     self.gender_var.get(),
            "result":     result_text,
            "confidence": (f"{confidence:.1f}" if probability >= 0.5
                           else f"{100-confidence:.1f}"),
            "notes":      self.text_notes.get("1.0", "end").strip()
        }
        self.last_result = record

        # Save to history and refresh the history tab
        self._save_to_history(record)
        self._refresh_history()

        # Enable print button
        self.print_button.config(state="normal", bg=BLUE)

        self.screen_button.config(state="normal", text="Screen for TB")
        self.progress_label.config(text="")
        self._update_status(
            f"Done — {record['name']}  |  {result_text}")


    # =========================================================================
    # PRINT / EXPORT
    # =========================================================================

    def _print_result(self):
        """
        Generates a formatted text report and opens it in Notepad.
        The user can then print it using File > Print in Notepad.
        The report is also saved in the reports/ folder for future reference.
        """
        if not self.last_result:
            messagebox.showinfo("No Result",
                                "Please run a screening first.")
            return

        r      = self.last_result
        border = "=" * 56

        report = (
            f"\n{border}\n"
            f"        TB SCREENING REPORT\n"
            f"        Edge AI System — Uganda Christian University\n"
            f"{border}\n\n"
            f"  Date & Time  :  {r['date']}   {r['time']}\n"
            f"  Patient Name :  {r['name']}\n"
            f"  Patient ID   :  {r['patient_id']}\n"
            f"  Age          :  {r['age'] or 'Not provided'}\n"
            f"  Gender       :  {r['gender']}\n\n"
            f"{border}\n\n"
            f"  SCREENING RESULT  :  {r['result']}\n"
            f"  Confidence        :  {r['confidence']}%\n\n"
            f"{border}\n\n"
            f"  Notes  :  {r['notes'] or 'None'}\n\n"
            f"  X-Ray File  :  "
            f"{os.path.basename(self.current_image_path or 'N/A')}\n\n"
            f"{border}\n\n"
            f"  DISCLAIMER\n"
            f"  This result was generated by an AI prototype system\n"
            f"  for research purposes only. It must be confirmed by\n"
            f"  a trained health worker before any clinical decision.\n\n"
            f"  System    : MobileNetV2 / TensorFlow Lite\n"
            f"  Developer : Kobumanzi Trishia | M24B23/011 | UCU 2026\n\n"
            f"{border}\n"
        )

        filename    = f"TB_Report_{r['patient_id']}_{r['date']}.txt"
        report_path = os.path.join(REPORTS_DIR, filename)

        with open(report_path, "w") as f:
            f.write(report)

        # os.startfile opens the file with its default application
        # On Windows this opens .txt files in Notepad
        try:
            os.startfile(report_path)
            messagebox.showinfo(
                "Report Saved",
                f"Report opened in Notepad.\n\n"
                f"Use  File > Print  to print it.\n\n"
                f"Saved to:\n{report_path}"
            )
        except Exception:
            messagebox.showinfo("Report Saved",
                                f"Report saved to:\n{report_path}")


    # =========================================================================
    # HELPERS
    # =========================================================================

    def _get_status_display(self):
        if self.model_status == "ready":
            return "● Model Ready", GREEN
        elif self.model_status == "missing_model":
            return "⚠ Model Not Found", YELLOW
        elif self.model_status == "missing_library":
            return "⚠ Library Missing", RED
        else:
            return "⚠ Error", RED

    def _show_setup_warning(self):
        if self.model_status == "missing_library":
            msg = "TensorFlow not installed.\n\nRun:\n  pip install tensorflow-cpu"
        elif self.model_status == "missing_model":
            msg = "Model file not found.\n\nRun:\n  python download_model.py"
        else:
            msg = f"Error: {self.model_status}"
        messagebox.showwarning("Setup Required", msg)

    def _clear_result(self):
        self.result_label.config(text="—", fg=LIGHT_GREY)
        self.confidence_label.config(text="")
        self.date_label.config(text="")
        self.progress_label.config(text="")
        self.print_button.config(state="disabled", bg=DARK_GREY)
        self.last_result = None

    def _reset(self):
        self.current_image_path = None
        self.photo_image        = None
        self.canvas.delete("all")
        self.canvas.create_text(
            200, 200,
            text="No X-ray loaded\n\nClick 'Upload X-Ray' to begin",
            fill=LIGHT_GREY, font=FONT_BODY, justify="center"
        )
        self.entry_name.delete(0, "end")
        self.entry_patient_id.delete(0, "end")
        self.entry_age.delete(0, "end")
        self.gender_var.set("Select")
        self.text_notes.delete("1.0", "end")
        self.filename_label.config(text="No file selected", fg=LIGHT_GREY)
        self.screen_button.config(state="disabled", text="Screen for TB")
        self._clear_result()
        self._update_status("Reset — fill in patient details and upload an X-ray")

    def _update_status(self, msg):
        self.status_bar_label.config(text=msg)


# ── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = TBScreeningApp(root)
    root.mainloop()