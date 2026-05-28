"""
Hair-Based Gender Classifier — Local ML (No API Required)
==========================================================
Uses DeepFace (local pre-trained models) for age & gender detection.
Uses OpenCV for hair length estimation.

Custom Classification Logic:
  - Age 20-30: long hair -> Female, short hair -> Male (overrides real gender)
  - Outside 20-30: standard gender prediction (hair ignored)

Install:
    pip install deepface opencv-python pillow numpy

Run:
    python Hair_local.py
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False


# ─────────────────────────────────────────────
#  HAIR LENGTH DETECTION  (OpenCV-based)
# ─────────────────────────────────────────────

def detect_hair_length(img_bgr: np.ndarray, face: dict) -> str:
    """
    Estimate hair length by analyzing the region below the detected face.
    Returns: 'short', 'medium', or 'long'
    """
    ih, iw = img_bgr.shape[:2]
    x      = int(face["x"])
    y      = int(face["y"])
    w      = int(face["w"])
    h      = int(face["h"])

    chin_y      = y + h
    face_height = h

    # Region of interest: from chin downward (up to 2x face height)
    roi_x1 = max(0, x - int(w * 0.3))
    roi_x2 = min(iw, x + w + int(w * 0.3))
    roi_y2 = min(ih, chin_y + int(face_height * 2.0))

    if chin_y >= roi_y2 or roi_x1 >= roi_x2:
        return "medium"

    roi = img_bgr[chin_y:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return "medium"

    roi_h, roi_w = roi.shape[:2]

    # ── Step 1: Skin mask (exclude skin-colored pixels) ──
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    skin_lower = np.array([0,  15,  60], dtype=np.uint8)
    skin_upper = np.array([25, 170, 255], dtype=np.uint8)
    skin_mask  = cv2.inRange(hsv, skin_lower, skin_upper)

    # ── Step 2: Dark pixel mask (hair tends to be darker than background) ──
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold to find dark regions
    _, dark_mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)

    # ── Step 3: Combine: dark AND not skin ──
    hair_mask = cv2.bitwise_and(dark_mask, cv2.bitwise_not(skin_mask))

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN,  kernel)

    # ── Step 4: Find lowest row with significant hair pixels ──
    row_coverage = np.sum(hair_mask > 0, axis=1) / roi_w
    threshold    = 0.12   # at least 12% of the row width must be hair

    significant_rows = np.where(row_coverage > threshold)[0]

    if len(significant_rows) == 0:
        return "short"

    lowest_hair_row  = significant_rows[-1]
    hair_extent_ratio = lowest_hair_row / max(face_height, 1)

    if hair_extent_ratio >= 0.85:
        return "long"
    elif hair_extent_ratio >= 0.30:
        return "medium"
    else:
        return "short"


# ─────────────────────────────────────────────
#  DEEPFACE ANALYSIS
# ─────────────────────────────────────────────

def analyze_image(image_path: str) -> dict:
    """
    Run DeepFace analysis + OpenCV hair detection.
    Returns a unified result dict.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError("Could not read image file.")

    # DeepFace: age + gender
    results = DeepFace.analyze(
        img_path    = image_path,
        actions     = ["age", "gender"],
        enforce_detection = True,
        detector_backend  = "opencv",
        silent      = True,
    )

    # DeepFace may return a list or a single dict
    result = results[0] if isinstance(results, list) else results

    age        = int(result["age"])
    gender_raw = result["dominant_gender"]           # 'Man' or 'Woman'
    bio_gender = "female" if gender_raw == "Woman" else "male"

    # Face region from DeepFace
    face_region = result.get("region", {})
    if not face_region:
        # fallback: use OpenCV face detector
        gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        face_casc = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_casc.detectMultiScale(gray, 1.1, 5)
        if len(faces) == 0:
            raise ValueError("No face detected in the image.")
        x, y, w, h = faces[0]
        face_region = {"x": x, "y": y, "w": w, "h": h}

    hair_length = detect_hair_length(img_bgr, face_region)

    return {
        "age":         age,
        "bio_gender":  bio_gender,
        "hair_length": hair_length,
    }


# ─────────────────────────────────────────────
#  CLASSIFICATION LOGIC
# ─────────────────────────────────────────────

def apply_logic(age: int, hair_length: str, bio_gender: str) -> dict:
    in_range = 20 <= age <= 30

    if in_range:
        if hair_length == "long":
            return {
                "predicted_gender": "Female",
                "rule": "Hair Override (Age 20-30)",
                "explanation": (
                    f"Age {age} is within 20-30.\n"
                    f"LONG hair detected -> predicted FEMALE\n"
                    f"(biological appearance: {bio_gender} is overridden)."
                ),
                "in_range": True,
            }
        elif hair_length == "short":
            return {
                "predicted_gender": "Male",
                "rule": "Hair Override (Age 20-30)",
                "explanation": (
                    f"Age {age} is within 20-30.\n"
                    f"SHORT hair detected -> predicted MALE\n"
                    f"(biological appearance: {bio_gender} is overridden)."
                ),
                "in_range": True,
            }
        else:
            gender = "Female" if bio_gender == "female" else "Male"
            return {
                "predicted_gender": gender,
                "rule": "Medium Hair Fallback (Age 20-30)",
                "explanation": (
                    f"Age {age} is within 20-30.\n"
                    f"MEDIUM hair — ambiguous for override.\n"
                    f"Falling back to biological appearance -> {gender}."
                ),
                "in_range": True,
            }
    else:
        gender    = "Female" if bio_gender == "female" else "Male"
        side_note = "below 20" if age < 20 else "above 30"
        return {
            "predicted_gender": gender,
            "rule": "Standard Prediction",
            "explanation": (
                f"Age {age} is {side_note} — outside 20-30 range.\n"
                f"Hair length is IGNORED.\n"
                f"Gender predicted from facial features -> {gender}."
            ),
            "in_range": False,
        }


# ─────────────────────────────────────────────
#  COLORS
# ─────────────────────────────────────────────
PINK   = "#D4537E"
BLUE   = "#185FA5"
GREEN  = "#0F6E56"
AMBER  = "#BA7517"
RED    = "#A32D2D"
BG     = "#F8F7F5"
CARD   = "#FFFFFF"
BORDER = "#DEDBD2"
TEXT   = "#1C1C1A"
MUTED  = "#6B6A66"


# ─────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────

class HairGenderApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hair-Based Gender Classifier (Local ML)")
        self.geometry("780x860")
        self.resizable(True, True)
        self.configure(bg=BG)
        self.image_path = None
        self.photo_img  = None
        self._build_ui()

    # ─── UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        canvas    = tk.Canvas(self, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.sf   = tk.Frame(canvas, bg=BG)
        self.sf.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.sf, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>",
            lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        f = self.sf
        
        # Header
        tk.Label(f, text="Hair-Based Gender Classifier",
                 font=("Helvetica", 20, "bold"), bg=BG, fg=TEXT
                 ).pack(anchor="w", padx=24, pady=(20, 2))
        
        self._logic_card(f)
        self._model_info_card(f)
        self._upload_section(f)

        # Preview
        self.preview_frame = tk.Frame(f, bg=BG)
        self.preview_frame.pack(fill="x", padx=24, pady=(0, 0))

        # Buttons
        btn_row = tk.Frame(f, bg=BG)
        btn_row.pack(anchor="w", padx=24, pady=(12, 0))
        self.analyze_btn = tk.Button(
            btn_row, text="  Analyze (Local Model)",
            font=("Helvetica", 12, "bold"), bg=BLUE, fg="white",
            relief="flat", padx=20, pady=8, cursor="hand2",
            command=self._start_analysis, state="disabled")
        self.analyze_btn.pack(side="left")
        tk.Button(btn_row, text="X  Clear",
                  font=("Helvetica", 11), bg=CARD, fg=MUTED,
                  relief="flat", padx=14, pady=8, cursor="hand2",
                  bd=1, highlightbackground=BORDER,
                  command=self._clear).pack(side="left", padx=(10, 0))

        # Status
        self.status_var = tk.StringVar(value="")
        tk.Label(f, textvariable=self.status_var,
                 font=("Helvetica", 11), bg=BG, fg=MUTED
                 ).pack(anchor="w", padx=24, pady=(8, 0))

        # Progress bar
        self.progress = ttk.Progressbar(f, mode="indeterminate", length=730)
        self.progress.pack(padx=24, pady=(4, 0))

        # Result
        self.result_frame = tk.Frame(f, bg=BG)
        self.result_frame.pack(fill="x", padx=24, pady=(14, 24))

    def _logic_card(self, parent):
        card = tk.Frame(parent, bg="#E6F1FB")
        card.pack(fill="x", padx=24, pady=(0, 10))
        inner = tk.Frame(card, bg="#E6F1FB")
        inner.pack(fill="x", padx=14, pady=10)
        tk.Label(inner, text="Classification Logic",
                 font=("Helvetica", 11, "bold"), bg="#E6F1FB", fg=BLUE
                 ).pack(anchor="w")
        rules = [
            ("Age 20-30:",          "Long hair -> Female  |  Short hair -> Male  (ignores biological gender)"),
            ("Outside 20-30:",      "Standard gender from facial features — hair length ignored"),
            ("Medium hair (20-30):", "Falls back to biological appearance"),
        ]
        for label, desc in rules:
            row = tk.Frame(inner, bg="#E6F1FB")
            row.pack(anchor="w", pady=1)
            tk.Label(row, text=label, font=("Helvetica", 10, "bold"),
                     bg="#E6F1FB", fg=TEXT).pack(side="left")
            tk.Label(row, text="  " + desc, font=("Helvetica", 10),
                     bg="#E6F1FB", fg=MUTED).pack(side="left")

    def _model_info_card(self, parent):
        card = tk.Frame(parent, bg="#F0F7F0")
        card.pack(fill="x", padx=24, pady=(0, 12))
        inner = tk.Frame(card, bg="#F0F7F0")
        inner.pack(fill="x", padx=14, pady=10)
        
    def _upload_section(self, parent):
        frame = tk.Frame(parent, bg=CARD, bd=1, relief="solid",
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill="x", padx=24, pady=(0, 12))
        inner = tk.Frame(frame, bg=CARD)
        inner.pack(pady=24)
        tk.Label(inner, text="[ Upload ]", font=("Helvetica", 22, "bold"),
                 bg=CARD, fg=MUTED).pack()
        tk.Label(inner, text="Click to select a photo",
                 font=("Helvetica", 13, "bold"), bg=CARD, fg=TEXT
                 ).pack(pady=(6, 2))
        tk.Label(inner, text="JPG  PNG  WEBP — one person per photo recommended",
                 font=("Helvetica", 10), bg=CARD, fg=MUTED).pack()
        tk.Button(inner, text="Browse Files",
                  font=("Helvetica", 11), bg=BG, fg=TEXT,
                  relief="flat", bd=1, highlightbackground=BORDER,
                  padx=16, pady=6, cursor="hand2",
                  command=self._browse_file).pack(pady=(12, 0))

    # ─── Actions ─────────────────────────────────────────────────────────

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp"),
                       ("All files", "*.*")])
        if path:
            self._load_image(path)

    def _load_image(self, path):
        self.image_path = path
        for w in self.preview_frame.winfo_children():
            w.destroy()
        for w in self.result_frame.winfo_children():
            w.destroy()
        self.status_var.set("")

        if PIL_AVAILABLE:
            img = Image.open(path)
            img.thumbnail((200, 200))
            self.photo_img = ImageTk.PhotoImage(img)
            row = tk.Frame(self.preview_frame, bg=BG)
            row.pack(anchor="w")
            tk.Label(row, image=self.photo_img, bg=BG,
                     relief="solid", bd=1,
                     highlightbackground=BORDER).pack(side="left")
            info = tk.Frame(row, bg=BG)
            info.pack(side="left", padx=(14, 0), anchor="n")
            fname   = os.path.basename(path)
            size_kb = os.path.getsize(path) // 1024
            tk.Label(info, text=fname, font=("Helvetica", 12, "bold"),
                     bg=BG, fg=TEXT, wraplength=400, justify="left"
                     ).pack(anchor="w")
            tk.Label(info, text=f"{size_kb} KB",
                     font=("Helvetica", 10), bg=BG, fg=MUTED
                     ).pack(anchor="w", pady=(4, 0))
        else:
            tk.Label(self.preview_frame,
                     text=f"Selected: {os.path.basename(path)}",
                     font=("Helvetica", 11), bg=BG, fg=MUTED
                     ).pack(anchor="w")

        self.analyze_btn.config(state="normal")

    def _clear(self):
        self.image_path = None
        self.photo_img  = None
        for w in self.preview_frame.winfo_children():
            w.destroy()
        for w in self.result_frame.winfo_children():
            w.destroy()
        self.status_var.set("")
        self.analyze_btn.config(state="disabled")

    def _start_analysis(self):
        if not self.image_path:
            return
        self.analyze_btn.config(state="disabled", text="Analyzing...")
        self.status_var.set("Running local ML models — this may take 10-30 sec on first run (downloading weights)...")
        for w in self.result_frame.winfo_children():
            w.destroy()
        self.progress.start(10)
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        try:
            data = analyze_image(self.image_path)
        except Exception as e:
            self.after(0, self._on_error, str(e))
            return
        self.after(0, self._on_result, data)

    def _on_error(self, msg):
        self.progress.stop()
        self.analyze_btn.config(state="normal", text="  Analyze (Local Model)")
        self.status_var.set("")
        messagebox.showerror("Analysis Failed",
            f"Error:\n{msg}\n\nTips:\n"
            "- Make sure the photo has a clear, visible face\n"
            "- Try a different image\n"
            "- Ensure deepface and opencv-python are installed")

    def _on_result(self, data):
        self.progress.stop()
        self.analyze_btn.config(state="normal", text="  Analyze (Local Model)")
        self.status_var.set("Analysis complete!")

        age    = data["age"]
        hair   = data["hair_length"]
        bio    = data["bio_gender"]
        logic  = apply_logic(age, hair, bio)
        self._render_result(age, hair, bio, logic)

    # ─── Result Rendering ─────────────────────────────────────────────────

    def _render_result(self, age, hair, bio, logic):
        f            = self.result_frame
        gender       = logic["predicted_gender"]
        gender_color = PINK if gender == "Female" else BLUE

        # Verdict banner
        banner = tk.Frame(f, bg=gender_color)
        banner.pack(fill="x", pady=(0, 12))
        inner = tk.Frame(banner, bg=gender_color)
        inner.pack(pady=14, padx=16, anchor="w")
        sex_icon = "[F]" if gender == "Female" else "[M]"
        tk.Label(inner, text=f"{sex_icon}  Predicted Gender: {gender}",
                 font=("Helvetica", 18, "bold"), bg=gender_color, fg="white"
                 ).pack(anchor="w")
        range_note = "In override range (20-30)" if logic["in_range"] else "Outside override range"
        tk.Label(inner, text=f"Estimated Age: {age} yrs  |  {range_note}",
                 font=("Helvetica", 11), bg=gender_color, fg="white"
                 ).pack(anchor="w", pady=(3, 0))

        # Metric cards
        grid = tk.Frame(f, bg=BG)
        grid.pack(fill="x", pady=(0, 12))
        metrics = [
            ("Estimated Age",      f"{age} yrs",          TEXT),
            ("Hair Length",        hair.capitalize(),      TEXT),
            ("Biological Gender",  bio.capitalize(),       TEXT),
            ("Rule Applied",       logic["rule"],          BLUE),
        ]
        for i, (lbl, val, vc) in enumerate(metrics):
            cell = tk.Frame(grid, bg=CARD, bd=1, relief="solid",
                            highlightbackground=BORDER, highlightthickness=1)
            cell.grid(row=0, column=i, padx=4, pady=0, sticky="nsew")
            grid.columnconfigure(i, weight=1)
            tk.Label(cell, text=lbl, font=("Helvetica", 9),
                     bg=CARD, fg=MUTED, wraplength=130, justify="center"
                     ).pack(pady=(10, 2), padx=6)
            tk.Label(cell, text=val, font=("Helvetica", 11, "bold"),
                     bg=CARD, fg=vc, wraplength=140, justify="center"
                     ).pack(pady=(0, 10), padx=6)

        # Logic explanation
        exp = tk.Frame(f, bg="#F0F4F8", bd=1, relief="solid",
                       highlightbackground=BORDER, highlightthickness=1)
        exp.pack(fill="x", pady=(0, 12))
        tk.Label(exp, text="Decision Path",
                 font=("Helvetica", 11, "bold"), bg="#F0F4F8", fg=TEXT
                 ).pack(anchor="w", padx=14, pady=(10, 2))
        tk.Label(exp, text=logic["explanation"],
                 font=("Helvetica", 11), bg="#F0F4F8", fg=MUTED,
                 wraplength=700, justify="left"
                 ).pack(anchor="w", padx=14, pady=(0, 10))

        # Hair length visual indicator
        hair_frame = tk.Frame(f, bg=BG)
        hair_frame.pack(fill="x", pady=(0, 8))
        tk.Label(hair_frame, text="Hair Length Scale:",
                 font=("Helvetica", 10), bg=BG, fg=MUTED
                 ).pack(anchor="w", pady=(0, 4))
        scale_row = tk.Frame(hair_frame, bg=BG)
        scale_row.pack(anchor="w")
        levels = [("Short", "short"), ("Medium", "medium"), ("Long", "long")]
        for label, val in levels:
            active = val == hair
            color  = gender_color if active else BORDER
            box    = tk.Frame(scale_row, bg=color, width=120, height=32)
            box.pack(side="left", padx=3)
            box.pack_propagate(False)
            fg = "white" if active else MUTED
            font_style = ("Helvetica", 10, "bold") if active else ("Helvetica", 10)
            tk.Label(box, text=label, font=font_style,
                     bg=color, fg=fg).place(relx=0.5, rely=0.5, anchor="center")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

def main():
    if not DEEPFACE_AVAILABLE:
        print("ERROR: 'deepface' not found.")
        print("Run:  pip install deepface opencv-python pillow numpy")
        input("Press Enter to exit...")
        return
    if not PIL_AVAILABLE:
        print("WARNING: Pillow not found — no image preview.")
        print("Run:  pip install pillow")

    app = HairGenderApp()
    app.mainloop()


if __name__ == "__main__":
    main()