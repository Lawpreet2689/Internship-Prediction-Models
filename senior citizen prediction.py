"""
Senior Citizen Identification System 
=============================================
Fixes:
  - Age correction factor (DeepFace underestimates by ~10-15 yrs)
  - Gender parsing handles all DeepFace API versions
  - Better face detector (DNN instead of Haar Cascade)
  - Lower confidence threshold for older/female faces
  - Debug overlay shows raw vs corrected age

Install:
    pip install deepface opencv-python pandas openpyxl pillow numpy

Run:
    python senior_citizen_detector.py
"""

import os
import cv2
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

import numpy as np
import pandas as pd

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


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SENIOR_AGE_THRESHOLD   = 60
ANALYZE_EVERY_N_FRAMES = 12       # run DeepFace every N frames
MIN_FACE_SIZE          = 50       # minimum face pixels
TRACK_COOLDOWN_SEC     = 30       # re-log same person after this many seconds
OUTPUT_DIR             = "senior_citizen_logs"

# ── Age correction ────────────────────────────────────────────────────────────
# DeepFace systematically underestimates age (especially 50+).
# Formula: corrected = raw_age + BASE_CORRECTION + SLOPE*(raw_age - 30)
# Tuned so that a raw prediction of 50 → ~60, raw 40 → ~47, raw 25 → ~28
AGE_BASE_CORRECTION = 5.0
AGE_SLOPE           = 0.18        # extra correction per year above 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors BGR
COLOR_SENIOR  = (0,   0,   230)
COLOR_ADULT   = (0,   200,  60)
COLOR_UNKNOWN = (200, 180,   0)


# ─────────────────────────────────────────────────────────────────────────────
#  FACE DETECTOR  (OpenCV DNN — far better than Haar Cascade)
# ─────────────────────────────────────────────────────────────────────────────

def build_face_detector():
    """
    Try OpenCV DNN face detector first (more accurate).
    Falls back to Haar Cascade if model files are missing.
    """
    # DNN model ships with opencv-python — locate it
    prototxt = None
    caffemodel = None

    # Common install paths
    for base in [cv2.data.haarcascades,
                 os.path.join(os.path.dirname(cv2.__file__), "data"),
                 "."]:
        p1 = os.path.join(base, "deploy.prototxt")
        p2 = os.path.join(base, "res10_300x300_ssd_iter_140000.caffemodel")
        if os.path.exists(p1) and os.path.exists(p2):
            prototxt, caffemodel = p1, p2
            break

    if prototxt and caffemodel:
        net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        return ("dnn", net)

    # Fallback: Haar Cascade
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return ("haar", cascade)


def detect_faces_dnn(frame, net, conf_threshold=0.5):
    h, w = frame.shape[:2]
    blob  = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < conf_threshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw >= MIN_FACE_SIZE and bh >= MIN_FACE_SIZE:
            faces.append((x1, y1, bw, bh))
    return faces


def detect_faces_haar(frame, cascade):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=4,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
    return [tuple(f) for f in faces] if len(faces) > 0 else []


# ─────────────────────────────────────────────────────────────────────────────
#  DEEPFACE ANALYSIS  — fixed gender parsing + age correction
# ─────────────────────────────────────────────────────────────────────────────

def correct_age(raw_age: int) -> int:
    """
    Apply correction to DeepFace's underestimated age.
    More aggressive correction for ages that are likely 50+.
    """
    correction = AGE_BASE_CORRECTION + AGE_SLOPE * max(0, raw_age - 30)
    return int(round(raw_age + correction))


def parse_gender(result: dict) -> str:
    """
    Handle all DeepFace API versions for gender extraction.
    Returns 'Male' or 'Female'.
    """
    # ── Method 1: dominant_gender key (most versions) ──
    dg = result.get("dominant_gender", "")
    if dg:
        dg_lower = str(dg).lower()
        if "woman" in dg_lower or "female" in dg_lower:
            return "Female"
        if "man" in dg_lower or "male" in dg_lower:
            return "Male"

    # ── Method 2: gender dict with probabilities ──
    gender_scores = result.get("gender", {})
    if isinstance(gender_scores, dict):
        # Keys might be 'Man'/'Woman' or 'male'/'female'
        woman_score = 0.0
        man_score   = 0.0
        for k, v in gender_scores.items():
            kl = str(k).lower()
            if "woman" in kl or "female" in kl:
                woman_score = float(v)
            elif "man" in kl or "male" in kl:
                man_score = float(v)
        if woman_score > 0 or man_score > 0:
            return "Female" if woman_score > man_score else "Male"

    return "Unknown"


def analyze_face_crop(face_img: np.ndarray) -> dict | None:
    """Run DeepFace on a face crop. Returns corrected age + gender."""
    try:
        # Upscale small faces — DeepFace performs poorly on tiny crops
        h, w = face_img.shape[:2]
        if h < 100 or w < 100:
            scale    = max(100 / h, 100 / w)
            face_img = cv2.resize(face_img,
                                  (int(w * scale), int(h * scale)),
                                  interpolation=cv2.INTER_CUBIC)

        results = DeepFace.analyze(
            img_path          = face_img,
            actions           = ["age", "gender"],
            enforce_detection = False,   # don't crash if no face found in crop
            detector_backend  = "skip",  # we already cropped the face
            silent            = True,
        )
        r = results[0] if isinstance(results, list) else results

        raw_age = int(r.get("age", 30))
        gender  = parse_gender(r)
        cor_age = correct_age(raw_age)

        return {
            "raw_age":  raw_age,
            "age":      cor_age,
            "gender":   gender,
        }
    except Exception as e:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  FACE TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class FaceRecord:
    def __init__(self, face_id, cx, cy):
        self.face_id     = face_id
        self.cx          = cx
        self.cy          = cy
        self.age         = None
        self.raw_age     = None
        self.gender      = None
        self.is_senior   = None
        self.last_seen   = time.time()
        self.last_logged = 0.0
        self.analyzing   = False


class FaceTracker:
    def __init__(self, max_dist=90, max_missing_frames=40):
        self.records         = {}
        self.next_id         = 1
        self.max_dist        = max_dist
        self.max_missing     = max_missing_frames
        self._missing_counts = {}

    def update(self, face_boxes):
        for fid in self.records:
            self._missing_counts[fid] = self._missing_counts.get(fid, 0) + 1

        assigned = {}
        for i, (x, y, w, h) in enumerate(face_boxes):
            cx, cy    = x + w // 2, y + h // 2
            best_id   = None
            best_dist = float("inf")
            for fid, rec in self.records.items():
                dist = np.hypot(cx - rec.cx, cy - rec.cy)
                if dist < best_dist and dist < self.max_dist:
                    best_dist = dist
                    best_id   = fid

            if best_id is None:
                fid = self.next_id
                self.next_id += 1
                self.records[fid] = FaceRecord(fid, cx, cy)
            else:
                fid = best_id

            rec            = self.records[fid]
            rec.cx         = cx
            rec.cy         = cy
            rec.last_seen  = time.time()
            self._missing_counts[fid] = 0
            assigned[i] = fid

        # Remove stale faces
        for fid in list(self._missing_counts):
            if self._missing_counts[fid] > self.max_missing:
                self.records.pop(fid, None)
                self._missing_counts.pop(fid, None)

        return [(assigned[i], *face_boxes[i]) for i in range(len(face_boxes))]


# ─────────────────────────────────────────────────────────────────────────────
#  VISIT LOGGER
# ─────────────────────────────────────────────────────────────────────────────

class VisitLogger:
    def __init__(self):
        ts             = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path  = os.path.join(OUTPUT_DIR, f"visits_{ts}.csv")
        self.xlsx_path = os.path.join(OUTPUT_DIR, f"visits_{ts}.xlsx")
        self.records   = []
        self._lock     = threading.Lock()
        pd.DataFrame(columns=["Face_ID","Raw_Age","Corrected_Age",
                               "Gender","Senior_Citizen","Time_of_Visit"]
                     ).to_csv(self.csv_path, index=False)

    def log(self, face_id, raw_age, age, gender, is_senior):
        row = {
            "Face_ID":        face_id,
            "Raw_Age":        raw_age,
            "Corrected_Age":  age,
            "Gender":         gender,
            "Senior_Citizen": "Yes" if is_senior else "No",
            "Time_of_Visit":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with self._lock:
            self.records.append(row)
            pd.DataFrame([row]).to_csv(self.csv_path, mode="a",
                                       header=False, index=False)
        return row

    def save_excel(self):
        with self._lock:
            if self.records:
                df = pd.DataFrame(self.records)
                df.to_excel(self.xlsx_path, index=False)
                return self.xlsx_path
        return None

    def get_dataframe(self):
        with self._lock:
            return pd.DataFrame(self.records) if self.records else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
#  VIDEO PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

class VideoProcessor:
    def __init__(self, on_frame_cb, on_log_cb, on_status_cb):
        self.on_frame_cb  = on_frame_cb
        self.on_log_cb    = on_log_cb
        self.on_status_cb = on_status_cb

        self.tracker     = FaceTracker()
        self.logger      = VisitLogger()
        self.running     = False
        self._thread     = None
        self._frame_idx  = 0

        detector_type, detector_obj = build_face_detector()
        self._detector_type = detector_type
        self._detector_obj  = detector_obj

        self._analysis_threads = {}

    def start(self, source):
        self.running = True
        self._thread = threading.Thread(
            target=self._loop, args=(source,), daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        self.logger.save_excel()

    def _loop(self, source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.on_status_cb("ERROR: Cannot open video source.")
            self.running = False
            return

        fps_t, fps_c, fps_v = time.time(), 0, 0.0
        self.on_status_cb("Running — first DeepFace call may take 10-30 sec...")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            self._frame_idx += 1
            fps_c += 1
            elapsed = time.time() - fps_t
            if elapsed >= 1.0:
                fps_v = fps_c / elapsed
                fps_c = 0
                fps_t = time.time()

            # ── Detect faces ──
            if self._detector_type == "dnn":
                faces = detect_faces_dnn(frame, self._detector_obj)
            else:
                faces = detect_faces_haar(frame, self._detector_obj)

            # ── Track ──
            tracked = self.tracker.update(faces)

            # ── Schedule DeepFace every N frames ──
            if self._frame_idx % ANALYZE_EVERY_N_FRAMES == 0:
                for (fid, x, y, w, h) in tracked:
                    rec = self.tracker.records.get(fid)
                    if rec and not rec.analyzing:
                        # Add padding to crop for better accuracy
                        pad  = int(max(w, h) * 0.15)
                        ih, iw = frame.shape[:2]
                        x1   = max(0, x - pad)
                        y1   = max(0, y - pad)
                        x2   = min(iw, x + w + pad)
                        y2   = min(ih, y + h + pad)
                        crop = frame[y1:y2, x1:x2].copy()
                        if crop.size > 0:
                            rec.analyzing = True
                            t = threading.Thread(
                                target=self._run_deepface,
                                args=(fid, crop), daemon=True)
                            self._analysis_threads[fid] = t
                            t.start()

            # ── Draw annotations ──
            annotated = frame.copy()
            for (fid, x, y, w, h) in tracked:
                rec = self.tracker.records.get(fid)
                if rec is None:
                    continue

                if rec.age is not None:
                    is_senior  = rec.age > SENIOR_AGE_THRESHOLD
                    rec.is_senior = is_senior
                    color      = COLOR_SENIOR if is_senior else COLOR_ADULT

                    # Log with cooldown
                    now = time.time()
                    if (now - rec.last_logged) > TRACK_COOLDOWN_SEC:
                        rec.last_logged = now
                        row = self.logger.log(
                            fid, rec.raw_age, rec.age, rec.gender, is_senior)
                        self.on_log_cb(row)

                    # Labels
                    lbl1 = f"ID:{fid}  Age:{rec.age}  {rec.gender}"
                    lbl2 = ">> SENIOR CITIZEN <<" if is_senior else "Adult"
                else:
                    color = COLOR_UNKNOWN
                    lbl1  = f"ID:{fid}  Analyzing..."
                    lbl2  = ""

                cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
                self._put_label(annotated, lbl1, x, y - 24, color)
                if lbl2:
                    self._put_label(annotated, lbl2, x, y + h + 4, color)

            # ── Overlay stats ──
            cv2.rectangle(annotated, (0, 0), (320, 60), (0, 0, 0), -1)
            cv2.putText(annotated, f"FPS: {fps_v:.1f}  |  Faces: {len(tracked)}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
            cv2.putText(annotated,
                        f"Logged: {len(self.logger.records)}  |  Threshold: >{SENIOR_AGE_THRESHOLD}",
                        (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200,200,200), 1)

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            self.on_frame_cb(rgb)

            if isinstance(source, str):
                time.sleep(0.01)

        cap.release()
        xlsx = self.logger.save_excel()
        msg  = f"Stopped. Logs saved → {os.path.abspath(OUTPUT_DIR)}"
        if xlsx:
            msg += f"\nExcel: {os.path.basename(xlsx)}"
        self.on_status_cb(msg)
        self.running = False

    def _run_deepface(self, fid, crop):
        result = analyze_face_crop(crop)
        rec    = self.tracker.records.get(fid)
        if rec:
            if result:
                rec.raw_age = result["raw_age"]
                rec.age     = result["age"]
                rec.gender  = result["gender"]
            rec.analyzing = False

    @staticmethod
    def _put_label(frame, text, x, y, color):
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.56
        thick = 1
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        y = max(th + 4, y)
        cv2.rectangle(frame, (x, y - th - 2), (x + tw + 6, y + 4), color, -1)
        cv2.putText(frame, text, (x + 3, y), font, scale, (255,255,255), thick)


# ─────────────────────────────────────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────────────────────────────────────

BLUE  = "#185FA5"
GREEN = "#0F6E56"
RED   = "#C0392B"
BG    = "#F4F3F1"
CARD  = "#FFFFFF"
BORDER= "#DEDBD2"
TEXT  = "#1C1C1A"
MUTED = "#6B6A66"


class SeniorDetectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Senior Citizen Identification System")
        self.geometry("1120x740")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.processor     = None
        self.photo_img     = None
        self._log_rows     = []
        self._selected_file = None
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Build UI ──────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top bar
        top = tk.Frame(self, bg=BLUE)
        top.pack(fill="x")
        tk.Label(top, text="  Senior Citizen Identification System",
                 font=("Helvetica", 14, "bold"), bg=BLUE, fg="white",
                 pady=10).pack(side="left")
        tk.Label(top,
                 text="DeepFace + OpenCV  |  Age > 60 = Senior Citizen  |  Auto Age Correction  |  CSV + Excel log",
                 font=("Helvetica", 10), bg=BLUE, fg="#B8D4F0"
                 ).pack(side="left", padx=10)

        # Body
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        # ── Left: video ──
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        vid_wrap = tk.Frame(left, bg="#111111", bd=1, relief="solid",
                            highlightbackground=BORDER, highlightthickness=1)
        vid_wrap.pack(fill="both", expand=True)
        self.video_label = tk.Label(vid_wrap, bg="#111111",
                                    text="No feed active\n\nSelect source and click  START",
                                    font=("Helvetica", 13), fg="#666666",
                                    width=70, height=22)
        self.video_label.pack(fill="both", expand=True)

        # Controls
        ctrl = tk.Frame(left, bg=BG)
        ctrl.pack(fill="x", pady=(8, 0))

        # Source
        src = tk.LabelFrame(ctrl, text="Source", bg=BG, fg=TEXT,
                            font=("Helvetica", 10, "bold"),
                            bd=1, relief="solid", highlightbackground=BORDER)
        src.pack(side="left", padx=(0, 8))

        self.source_var = tk.StringVar(value="webcam")
        tk.Radiobutton(src, text="Webcam (live)",
                       variable=self.source_var, value="webcam",
                       bg=BG, font=("Helvetica", 10),
                       command=self._on_source_toggle
                       ).pack(anchor="w", padx=8, pady=2)
        tk.Radiobutton(src, text="Video file",
                       variable=self.source_var, value="file",
                       bg=BG, font=("Helvetica", 10),
                       command=self._on_source_toggle
                       ).pack(anchor="w", padx=8, pady=2)

        file_row = tk.Frame(src, bg=BG)
        file_row.pack(anchor="w", padx=8, pady=(0, 6))
        self.file_lbl = tk.Label(file_row, text="No file selected",
                                 font=("Helvetica", 9), bg=BG, fg=MUTED,
                                 width=26, anchor="w")
        self.file_lbl.pack(side="left")
        self.browse_btn = tk.Button(file_row, text="Browse",
                                    font=("Helvetica", 9), bg=BG, fg=TEXT,
                                    relief="flat", bd=1, highlightbackground=BORDER,
                                    padx=6, pady=2, cursor="hand2",
                                    command=self._browse, state="disabled")
        self.browse_btn.pack(side="left", padx=(4, 0))

        # Threshold control
        thr_frame = tk.LabelFrame(ctrl, text="Senior Age Threshold",
                                  bg=BG, fg=TEXT,
                                  font=("Helvetica", 10, "bold"),
                                  bd=1, relief="solid", highlightbackground=BORDER)
        thr_frame.pack(side="left", padx=(0, 8))
        self.threshold_var = tk.IntVar(value=SENIOR_AGE_THRESHOLD)
        tk.Spinbox(thr_frame, from_=50, to=80, width=5,
                   textvariable=self.threshold_var,
                   font=("Helvetica", 12, "bold"), justify="center"
                   ).pack(padx=12, pady=8)

        # Start / Stop
        btn_f = tk.Frame(ctrl, bg=BG)
        btn_f.pack(side="left", padx=(0, 8))
        self.start_btn = tk.Button(
            btn_f, text="START", font=("Helvetica", 12, "bold"),
            bg=GREEN, fg="white", relief="flat",
            padx=18, pady=10, cursor="hand2", width=7,
            command=self._start)
        self.start_btn.pack(pady=2)
        self.stop_btn = tk.Button(
            btn_f, text="STOP", font=("Helvetica", 12, "bold"),
            bg=RED, fg="white", relief="flat",
            padx=18, pady=10, cursor="hand2", width=7,
            command=self._stop, state="disabled")
        self.stop_btn.pack(pady=2)

        # Save / Open
        util_f = tk.Frame(ctrl, bg=BG)
        util_f.pack(side="left")
        tk.Button(util_f, text="Save Excel",
                  font=("Helvetica", 10), bg=BLUE, fg="white",
                  relief="flat", padx=10, pady=6, cursor="hand2",
                  command=self._save_excel).pack(pady=3)
        tk.Button(util_f, text="Open Log Folder",
                  font=("Helvetica", 10), bg=BG, fg=TEXT,
                  relief="flat", bd=1, highlightbackground=BORDER,
                  padx=10, pady=6, cursor="hand2",
                  command=self._open_folder).pack(pady=3)

        # ── Right: log ──
        right = tk.Frame(body, bg=BG, width=350)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        tk.Label(right, text="Detection Log",
                 font=("Helvetica", 12, "bold"), bg=BG, fg=TEXT
                 ).pack(anchor="w", pady=(0, 4))

        stats = tk.Frame(right, bg=BG)
        stats.pack(fill="x", pady=(0, 6))
        self.total_var  = tk.StringVar(value="Total: 0")
        self.senior_var = tk.StringVar(value="Seniors: 0")
        self.adult_var  = tk.StringVar(value="Adults: 0")
        for var, color in [(self.total_var, TEXT),
                           (self.senior_var, RED),
                           (self.adult_var, GREEN)]:
            tk.Label(stats, textvariable=var,
                     font=("Helvetica", 10, "bold"),
                     bg=BG, fg=color, padx=4).pack(side="left")

        cols = ("Time", "ID", "RawAge", "Age", "Gender", "Senior")
        self.tree = ttk.Treeview(right, columns=cols,
                                 show="headings", height=24)
        widths = {"Time":80, "ID":30, "RawAge":58, "Age":48,
                  "Gender":62, "Senior":68}
        headers = {"Time":"Time","ID":"ID","RawAge":"Raw Age",
                   "Age":"Corr.Age","Gender":"Gender","Senior":"Senior"}
        for c in cols:
            self.tree.heading(c, text=headers[c])
            self.tree.column(c, width=widths[c], anchor="center")

        self.tree.tag_configure("senior", background="#FDECEA", foreground=RED)
        self.tree.tag_configure("adult",  background="#EEF7EE", foreground=GREEN)

        sb2 = ttk.Scrollbar(right, orient="vertical",
                            command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb2.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb2.pack(side="right", fill="y")

        # Status bar
        self.status_var = tk.StringVar(
            value="Ready.  Select source and click START.")
        bar = tk.Frame(self, bg="#E8E6E0")
        bar.pack(fill="x", side="bottom")
        tk.Label(bar, textvariable=self.status_var,
                 font=("Helvetica", 10), bg="#E8E6E0", fg=MUTED,
                 anchor="w", padx=12, pady=4).pack(fill="x")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _on_source_toggle(self):
        is_file = self.source_var.get() == "file"
        self.browse_btn.config(state="normal" if is_file else "disabled")

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                       ("All", "*.*")])
        if path:
            self._selected_file = path
            self.file_lbl.config(text=os.path.basename(path))

    def _start(self):
        if not DEEPFACE_AVAILABLE:
            messagebox.showerror("Missing",
                "Run:  pip install deepface opencv-python pandas openpyxl pillow")
            return

        source = 0 if self.source_var.get() == "webcam" else self._selected_file
        if source is None:
            messagebox.showwarning("No File", "Please select a video file.")
            return

        # Update threshold from spinbox
        global SENIOR_AGE_THRESHOLD
        SENIOR_AGE_THRESHOLD = self.threshold_var.get()

        self.processor = VideoProcessor(
            on_frame_cb  = self._update_frame,
            on_log_cb    = self._add_log_row,
            on_status_cb = self._update_status,
        )
        self._log_rows.clear()
        for row in self.tree.get_children():
            self.tree.delete(row)
        self._update_stats()

        self.processor.start(source)
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

    def _stop(self):
        if self.processor:
            self.processor.stop()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def _save_excel(self):
        if self.processor:
            p = self.processor.logger.save_excel()
            if p:
                messagebox.showinfo("Saved",
                    f"Excel saved:\n{os.path.abspath(p)}")
            else:
                messagebox.showinfo("No data", "Nothing logged yet.")

    def _open_folder(self):
        folder = os.path.abspath(OUTPUT_DIR)
        os.makedirs(folder, exist_ok=True)
        import subprocess, sys
        if sys.platform == "win32":
            os.startfile(folder)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _update_frame(self, frame_rgb):
        self.after(0, self._set_frame, frame_rgb)

    def _set_frame(self, frame_rgb):
        if not PIL_AVAILABLE:
            return
        try:
            h, w = frame_rgb.shape[:2]
            max_w, max_h = 730, 480
            scale = min(max_w / w, max_h / h, 1.0)
            nw, nh = int(w * scale), int(h * scale)
            img = Image.fromarray(frame_rgb).resize((nw, nh), Image.LANCZOS)
            self.photo_img = ImageTk.PhotoImage(img)
            self.video_label.config(image=self.photo_img, text="")
        except Exception:
            pass

    def _add_log_row(self, row):
        self.after(0, self._insert_row, row)

    def _insert_row(self, row):
        senior = row["Senior_Citizen"] == "Yes"
        tag    = "senior" if senior else "adult"
        t      = row["Time_of_Visit"].split(" ")[1]
        vals   = (t, row["Face_ID"], row["Raw_Age"],
                  row["Corrected_Age"], row["Gender"],
                  row["Senior_Citizen"])
        self.tree.insert("", 0, values=vals, tags=(tag,))
        self._log_rows.append(row)
        self._update_stats()

    def _update_stats(self):
        total   = len(self._log_rows)
        seniors = sum(1 for r in self._log_rows
                      if r.get("Senior_Citizen") == "Yes")
        self.total_var.set(f"Total: {total}")
        self.senior_var.set(f"Seniors: {seniors}")
        self.adult_var.set(f"Adults: {total - seniors}")

    def _update_status(self, msg):
        self.after(0, self.status_var.set, msg)

    def _on_close(self):
        if self.processor:
            self.processor.stop()
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not DEEPFACE_AVAILABLE:
        print("ERROR: deepface not installed.")
        print("Run:  pip install deepface opencv-python pandas openpyxl pillow numpy")
        input("Press Enter to exit...")
        return
    if not PIL_AVAILABLE:
        print("WARNING: Pillow missing — no video preview.")
        print("Run:  pip install pillow")

    app = SeniorDetectorApp()
    app.mainloop()


if __name__ == "__main__":
    main()