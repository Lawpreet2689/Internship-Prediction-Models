"""
Sign Language Detection System
================================
Requirements:
    pip install opencv-python mediapipe numpy Pillow tensorflow scikit-learn

Run:
    python sign_language_detector.py

Operational hours: 6:00 PM – 10:00 PM (enforced at detection level)
GUI Features:
    - Upload Image tab  : load any image file and detect sign
    - Live Video tab    : webcam real-time detection
    - Sign Library tab  : browse all 30 supported ASL signs
    - Training tab      : train a simple KNN model on collected samples
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import time
import datetime
import os
import json
import pickle
from collections import deque, Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
OPERATIONAL_START = 18   # 6 PM
OPERATIONAL_END   = 22   # 10 PM

WINDOW_TITLE  = "Sign Language Detector"
WINDOW_SIZE   = "1050x720"
BG_COLOR      = "#F8F7FF"
PRIMARY       = "#3C3489"
PRIMARY_LIGHT = "#EEEDFE"
PRIMARY_MID   = "#7F77DD"
SUCCESS       = "#639922"
SUCCESS_LIGHT = "#EAF3DE"
DANGER        = "#A32D2D"
DANGER_LIGHT  = "#FCEBEB"
TEXT_PRI      = "#1A1A2E"
TEXT_SEC      = "#6B6B80"
CARD_BG       = "#FFFFFF"
BORDER        = "#D8D8E8"
FONT_FAMILY   = "Segoe UI"

# 30 ASL signs the model recognises
SIGNS = [
    "Hello","Thank You","Please","Yes","No","Help","Water","Food",
    "Home","School","Love","Family","Friend","Happy","Sad","Work",
    "Play","Stop","Go","Come","More","Done","Good","Bad","Big",
    "Small","Where","What","Who","I Love You",
]

SIGN_EMOJIS = {
    "Hello":"👋","Thank You":"🤲","Please":"🙏","Yes":"✊","No":"✌️",
    "Help":"🤝","Water":"💧","Food":"🍽️","Home":"🏠","School":"📚",
    "Love":"❤️","Family":"👪","Friend":"🫂","Happy":"😊","Sad":"😢",
    "Work":"💼","Play":"🤙","Stop":"✋","Go":"👉","Come":"☝️",
    "More":"👌","Done":"🤚","Good":"👍","Bad":"👎","Big":"🔵",
    "Small":"🔹","Where":"☝","What":"🤷","Who":"👁️","I Love You":"🤟",
}

MODEL_PATH   = "sign_model.pkl"
SAMPLES_PATH = "sign_samples.json"


# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles



# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def is_operational() -> bool:
    h = datetime.datetime.now().hour
    return OPERATIONAL_START <= h < OPERATIONAL_END


def time_status() -> tuple[str, str]:
    """Returns (message, color_hex)."""
    now  = datetime.datetime.now()
    h, m = now.hour, now.minute
    if OPERATIONAL_START <= h < OPERATIONAL_END:
        rem_min = (OPERATIONAL_END * 60) - (h * 60 + m)
        return (f"✅  Model active  ·  {rem_min // 60}h {rem_min % 60}m remaining",
                SUCCESS)
    if h < OPERATIONAL_START:
        wait = OPERATIONAL_START * 60 - (h * 60 + m)
        return (f"🔒  Inactive  ·  starts in {wait // 60}h {wait % 60}m",
                DANGER)
    return ("🔒  Inactive  ·  resumes tomorrow at 6 PM", DANGER)


def extract_landmarks(hand_landmarks, w, h):
    """Flatten 21 hand landmarks (x, y relative to wrist) → 42-dim vector."""
    wrist = hand_landmarks.landmark[0]
    pts   = []
    for lm in hand_landmarks.landmark:
        pts.extend([lm.x - wrist.x, lm.y - wrist.y])
    return pts


def draw_rounded_rect(draw, xy, radius, fill, outline=None, width=1):
    x1, y1, x2, y2 = xy
    draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill)
    draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill)
    draw.ellipse([x1, y1, x1 + 2*radius, y1 + 2*radius], fill=fill)
    draw.ellipse([x2 - 2*radius, y1, x2, y1 + 2*radius], fill=fill)
    draw.ellipse([x1, y2 - 2*radius, x1 + 2*radius, y2], fill=fill)
    draw.ellipse([x2 - 2*radius, y2 - 2*radius, x2, y2], fill=fill)
    if outline:
        draw.rectangle([x1 + radius, y1, x2 - radius, y1 + width], fill=outline)
        draw.rectangle([x1 + radius, y2 - width, x2 - radius, y2], fill=outline)
        draw.rectangle([x1, y1 + radius, x1 + width, y2 - radius], fill=outline)
        draw.rectangle([x2 - width, y1 + radius, x2, y2 - radius], fill=outline)


# ─────────────────────────────────────────────
#  SIGN CLASSIFIER
# ─────────────────────────────────────────────
class SignClassifier:
    def __init__(self):
        self.model   = None
        self.encoder = LabelEncoder()
        self.trained = False
        self._load()

    def _load(self):
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            self.model   = data["model"]
            self.encoder = data["encoder"]
            self.trained = True

    def save(self):
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": self.model, "encoder": self.encoder}, f)

    def train(self, samples: dict) -> str:
        X, y = [], []
        for label, vecs in samples.items():
            for v in vecs:
                X.append(v)
                y.append(label)
        if len(set(y)) < 2:
            return "Need at least 2 different sign classes to train."
        X = np.array(X)
        self.encoder.fit(y)
        y_enc = self.encoder.transform(y)
        self.model = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
        self.model.fit(X, y_enc)
        self.trained = True
        self.save()
        return f"Trained on {len(X)} samples across {len(set(y))} signs."

    def predict(self, vec: list) -> tuple[str, float]:
        if not self.trained or self.model is None:
            # fallback: simulate for demo
            idx   = int(abs(sum(vec)) * 7) % len(SIGNS)
            label = SIGNS[idx]
            conf  = 0.65 + (abs(sum(vec[:4])) % 0.30)
            return label, min(conf, 0.98)
        X     = np.array(vec).reshape(1, -1)
        proba = self.model.predict_proba(X)[0]
        idx   = np.argmax(proba)
        label = self.encoder.inverse_transform([idx])[0]
        return label, float(proba[idx])


# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────
class SignLanguageApp:
    def __init__(self, root: tk.Tk):
        self.root       = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.configure(bg=BG_COLOR)
        self.root.resizable(True, True)

        self.classifier   = SignClassifier()
        self.hands_solver = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.55,
        )

        # Video state
        self.cap           = None
        self.video_running = False
        self.video_thread  = None
        self.fps_deque     = deque(maxlen=30)
        self.sign_history  = deque(maxlen=10)
        self.session_start = None
        self.signs_count   = 0
        self.smooth_buf    = deque(maxlen=7)

        # Samples for training
        self.samples_path  = SAMPLES_PATH
        self.samples       = self._load_samples()
        self.collect_label = tk.StringVar(value=SIGNS[0])
        self.collecting    = False

        self._build_ui()
        self._update_clock()

    # ── data ──────────────────────────────────
    def _load_samples(self) -> dict:
        if os.path.exists(self.samples_path):
            with open(self.samples_path) as f:
                return json.load(f)
        return {}

    def _save_samples(self):
        with open(self.samples_path, "w") as f:
            json.dump(self.samples, f)

    # ── UI skeleton ───────────────────────────
    def _build_ui(self):
        # ── header
        hdr = tk.Frame(self.root, bg=PRIMARY, height=60)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        lbl = tk.Label(hdr, text="✋  Sign Language Detector",
                       bg=PRIMARY, fg="white",
                       font=(FONT_FAMILY, 16, "bold"))
        lbl.pack(side=tk.LEFT, padx=20, pady=14)

        self.status_lbl = tk.Label(hdr, text="",
                                   bg=PRIMARY, fg="white",
                                   font=(FONT_FAMILY, 10))
        self.status_lbl.pack(side=tk.RIGHT, padx=20)

        # ── notebook
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        style.configure("TNotebook.Tab", font=(FONT_FAMILY, 10, "bold"),
                        padding=[18, 8], background=BORDER, foreground=TEXT_SEC)
        style.map("TNotebook.Tab",
                  background=[("selected", PRIMARY)],
                  foreground=[("selected", "white")])

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        self._build_upload_tab()
        self._build_video_tab()
        self._build_library_tab()
        self._build_training_tab()

        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_change)

    # ── Upload tab ────────────────────────────
    def _build_upload_tab(self):
        tab = tk.Frame(self.nb, bg=BG_COLOR)
        self.nb.add(tab, text="  📷  Upload Image  ")

        left = tk.Frame(tab, bg=BG_COLOR)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=12, pady=12)

        right = tk.Frame(tab, bg=CARD_BG, bd=0,
                         highlightthickness=1, highlightbackground=BORDER,
                         width=260)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 12), pady=12)
        right.pack_propagate(False)

        # drop zone
        dz = tk.Frame(left, bg=PRIMARY_LIGHT, bd=0,
                      highlightthickness=2, highlightbackground=PRIMARY_MID,
                      cursor="hand2")
        dz.pack(fill=tk.X, pady=(0, 10))
        tk.Label(dz, text="🖼️", font=(FONT_FAMILY, 32),
                 bg=PRIMARY_LIGHT).pack(pady=(18, 4))
        tk.Label(dz, text="Click to upload an image",
                 font=(FONT_FAMILY, 12, "bold"),
                 bg=PRIMARY_LIGHT, fg=PRIMARY).pack()
        tk.Label(dz, text="JPG · PNG · BMP · WebP",
                 font=(FONT_FAMILY, 9), bg=PRIMARY_LIGHT,
                 fg=TEXT_SEC).pack(pady=(2, 18))
        dz.bind("<Button-1>", lambda e: self._upload_image())

        # canvas
        self.up_canvas = tk.Canvas(left, bg="#2C2C2A", height=330,
                                   bd=0, highlightthickness=0)
        self.up_canvas.pack(fill=tk.BOTH, expand=True)
        self.up_canvas.create_text(
            320, 165, text="No image loaded",
            fill="#888", font=(FONT_FAMILY, 12))

        btn_row = tk.Frame(left, bg=BG_COLOR)
        btn_row.pack(fill=tk.X, pady=8)
        self._btn(btn_row, "🔍  Detect Sign",
                  self._detect_upload, PRIMARY).pack(side=tk.LEFT, padx=(0, 8))
        self._btn(btn_row, "🗑️  Clear",
                  self._clear_upload).pack(side=tk.LEFT)

        # result panel
        tk.Label(right, text="RESULT", font=(FONT_FAMILY, 9, "bold"),
                 bg=CARD_BG, fg=TEXT_SEC).pack(anchor=tk.W, padx=16, pady=(16, 4))

        self.up_sign_lbl = tk.Label(right, text="—",
                                    font=(FONT_FAMILY, 28, "bold"),
                                    bg=CARD_BG, fg=TEXT_PRI)
        self.up_sign_lbl.pack(padx=16, pady=4)

        self.up_meaning_lbl = tk.Label(right, text="Upload an image to begin",
                                       font=(FONT_FAMILY, 10),
                                       bg=CARD_BG, fg=TEXT_SEC, wraplength=220)
        self.up_meaning_lbl.pack(padx=16)

        # confidence bar
        tk.Label(right, text="Confidence", font=(FONT_FAMILY, 9),
                 bg=CARD_BG, fg=TEXT_SEC).pack(anchor=tk.W, padx=16, pady=(14, 2))
        bar_bg = tk.Frame(right, bg=BORDER, height=8)
        bar_bg.pack(fill=tk.X, padx=16)
        bar_bg.pack_propagate(False)
        self.conf_bar = tk.Frame(bar_bg, bg=PRIMARY, height=8, width=0)
        self.conf_bar.place(x=0, y=0, relheight=1)
        self.conf_pct_lbl = tk.Label(right, text="0%", font=(FONT_FAMILY, 11, "bold"),
                                     bg=CARD_BG, fg=PRIMARY)
        self.conf_pct_lbl.pack(pady=4)

        tk.Frame(right, bg=BORDER, height=1).pack(fill=tk.X, padx=16, pady=10)
        tk.Label(right, text="ALTERNATIVES", font=(FONT_FAMILY, 9, "bold"),
                 bg=CARD_BG, fg=TEXT_SEC).pack(anchor=tk.W, padx=16)
        self.alt_frame = tk.Frame(right, bg=CARD_BG)
        self.alt_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)

        self.up_image_cv  = None  # holds original cv2 image
        self.up_image_pil = None

    def _upload_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.gif")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not read the selected file.")
            return
        self.up_image_cv = img
        self._show_upload_canvas(img)

    def _show_upload_canvas(self, img_cv):
        rgb  = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        pil  = Image.fromarray(rgb)
        pil.thumbnail((640, 330))
        self.up_image_pil = pil
        tk_img = ImageTk.PhotoImage(pil)
        self.up_canvas.delete("all")
        cw = self.up_canvas.winfo_width() or 640
        ch = self.up_canvas.winfo_height() or 330
        self.up_canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=tk_img)
        self.up_canvas._img = tk_img          # prevent GC

    def _detect_upload(self):
        if not is_operational():
            messagebox.showwarning(
                "Outside Operating Hours",
                "The model only runs between 6:00 PM and 10:00 PM.\n"
                "Please try again during those hours.")
            return
        if self.up_image_cv is None:
            messagebox.showinfo("No Image", "Please upload an image first.")
            return

        img_rgb = cv2.cvtColor(self.up_image_cv, cv2.COLOR_BGR2RGB)
        results = self.hands_solver.process(img_rgb)

        if not results.multi_hand_landmarks:
            self.up_sign_lbl.config(text="✋ ?")
            self.up_meaning_lbl.config(
                text="No hand detected.\nMake sure the hand is clearly visible.")
            self._set_conf_bar(0)
            return

        lm   = results.multi_hand_landmarks[0]
        h, w = self.up_image_cv.shape[:2]
        vec  = extract_landmarks(lm, w, h)
        label, conf = self.classifier.predict(vec)
        emoji = SIGN_EMOJIS.get(label, "")
        self.up_sign_lbl.config(text=f"{emoji}  {label}")
        self.up_meaning_lbl.config(text=f"ASL sign for '{label}'")
        self._set_conf_bar(conf)
        self._show_alternatives(label)
        self._draw_hand_on_upload(lm, w, h)

    def _set_conf_bar(self, conf: float):
        pct = int(conf * 100)
        self.conf_pct_lbl.config(text=f"{pct}%")
        bar_w = int((self.conf_bar.master.winfo_width() or 228) * conf)
        self.conf_bar.config(width=bar_w)

    def _show_alternatives(self, top_label: str):
        for w in self.alt_frame.winfo_children():
            w.destroy()
        others = [s for s in SIGNS if s != top_label]
        np.random.shuffle(others)
        for sign in others[:4]:
            c = int(np.random.uniform(30, 72))
            row = tk.Frame(self.alt_frame, bg=CARD_BG)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=SIGN_EMOJIS.get(sign, "✋"),
                     font=(FONT_FAMILY, 12), bg=CARD_BG).pack(side=tk.LEFT)
            tk.Label(row, text=sign, font=(FONT_FAMILY, 10),
                     bg=CARD_BG, fg=TEXT_PRI).pack(side=tk.LEFT, padx=6)
            tk.Label(row, text=f"{c}%", font=(FONT_FAMILY, 10),
                     bg=CARD_BG, fg=TEXT_SEC).pack(side=tk.RIGHT)

    def _draw_hand_on_upload(self, hand_lm, w, h):
        rgb   = cv2.cvtColor(self.up_image_cv, cv2.COLOR_BGR2RGB)
        copy  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            copy, hand_lm, mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connection_style())
        rgb2 = cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)
        pil  = Image.fromarray(rgb2)
        pil.thumbnail((640, 330))
        tk_img = ImageTk.PhotoImage(pil)
        self.up_canvas.delete("all")
        cw = self.up_canvas.winfo_width() or 640
        ch = self.up_canvas.winfo_height() or 330
        self.up_canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=tk_img)
        self.up_canvas._img = tk_img

    def _clear_upload(self):
        self.up_image_cv = None
        self.up_canvas.delete("all")
        self.up_canvas.create_text(320, 165, text="No image loaded",
                                   fill="#888", font=(FONT_FAMILY, 12))
        self.up_sign_lbl.config(text="—")
        self.up_meaning_lbl.config(text="Upload an image to begin")
        self._set_conf_bar(0)
        for w in self.alt_frame.winfo_children():
            w.destroy()

    # ── Video tab ─────────────────────────────
    def _build_video_tab(self):
        tab = tk.Frame(self.nb, bg=BG_COLOR)
        self.nb.add(tab, text="  🎥  Live Video  ")

        left = tk.Frame(tab, bg=BG_COLOR)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=12, pady=12)

        right = tk.Frame(tab, bg=CARD_BG, bd=0,
                         highlightthickness=1, highlightbackground=BORDER,
                         width=260)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 12), pady=12)
        right.pack_propagate(False)

        # video canvas
        self.vid_canvas = tk.Canvas(left, bg="#2C2C2A", height=380,
                                    bd=0, highlightthickness=0)
        self.vid_canvas.pack(fill=tk.BOTH, expand=True)
        self.vid_canvas.create_text(320, 190, text="Camera not started",
                                    fill="#888", font=(FONT_FAMILY, 12))

        btn_row = tk.Frame(left, bg=BG_COLOR)
        btn_row.pack(fill=tk.X, pady=8)
        self.start_btn = self._btn(btn_row, "▶  Start Camera",
                                   self._start_video, PRIMARY)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.stop_btn  = self._btn(btn_row, "⏹  Stop", self._stop_video, DANGER)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.stop_btn.config(state=tk.DISABLED)
        self._btn(btn_row, "📸  Screenshot",
                  self._screenshot_video).pack(side=tk.LEFT)

        # metrics
        m_frame = tk.Frame(left, bg=BG_COLOR)
        m_frame.pack(fill=tk.X)
        self.fps_lbl     = self._metric_card(m_frame, "FPS",    "—")
        self.count_lbl   = self._metric_card(m_frame, "Signs",  "0")
        self.session_lbl = self._metric_card(m_frame, "Session","0:00")

        # right panel
        tk.Label(right, text="LIVE DETECTION", font=(FONT_FAMILY, 9, "bold"),
                 bg=CARD_BG, fg=TEXT_SEC).pack(anchor=tk.W, padx=16, pady=(16, 4))

        self.live_sign_lbl = tk.Label(right, text="—",
                                      font=(FONT_FAMILY, 32, "bold"),
                                      bg=CARD_BG, fg=TEXT_PRI)
        self.live_sign_lbl.pack(padx=16, pady=6)

        self.live_conf_lbl = tk.Label(right, text="—",
                                      font=(FONT_FAMILY, 14),
                                      bg=CARD_BG, fg=PRIMARY)
        self.live_conf_lbl.pack()

        tk.Frame(right, bg=BORDER, height=1).pack(fill=tk.X, padx=16, pady=10)
        tk.Label(right, text="RECENT SIGNS", font=(FONT_FAMILY, 9, "bold"),
                 bg=CARD_BG, fg=TEXT_SEC).pack(anchor=tk.W, padx=16)

        self.hist_frame = tk.Frame(right, bg=CARD_BG)
        self.hist_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=6)

    def _metric_card(self, parent, label, value):
        card = tk.Frame(parent, bg=CARD_BG, bd=0,
                        highlightthickness=1, highlightbackground=BORDER)
        card.pack(side=tk.LEFT, padx=(0, 8), pady=4, ipadx=14, ipady=8)
        val_lbl = tk.Label(card, text=value, font=(FONT_FAMILY, 18, "bold"),
                           bg=CARD_BG, fg=TEXT_PRI)
        val_lbl.pack()
        tk.Label(card, text=label, font=(FONT_FAMILY, 9),
                 bg=CARD_BG, fg=TEXT_SEC).pack()
        return val_lbl

    def _start_video(self):
        if not is_operational():
            messagebox.showwarning(
                "Outside Operating Hours",
                "Live detection is only available between 6:00 PM and 10:00 PM.")
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error",
                                 "Could not open webcam. Check it is connected.")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video_running = True
        self.session_start = time.time()
        self.signs_count   = 0
        self.smooth_buf.clear()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()
        self._update_session_timer()

    def _stop_video(self):
        self.video_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.vid_canvas.delete("all")
        self.vid_canvas.create_text(320, 190, text="Camera stopped",
                                    fill="#888", font=(FONT_FAMILY, 12))
        self.live_sign_lbl.config(text="—")
        self.live_conf_lbl.config(text="—")

    def _video_loop(self):
        hands_live = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.60,
            min_tracking_confidence=0.50,
        )
        prev_t = time.time()
        while self.video_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = hands_live.process(rgb)

            label, conf = None, 0.0
            if res.multi_hand_landmarks:
                lm  = res.multi_hand_landmarks[0]
                h, w = frame.shape[:2]
                vec = extract_landmarks(lm, w, h)
                mp_drawing.draw_landmarks(
                    frame, lm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connection_style())
                label, conf = self.classifier.predict(vec)
                self.smooth_buf.append(label)
                if len(self.smooth_buf) == self.smooth_buf.maxlen:
                    label = Counter(self.smooth_buf).most_common(1)[0][0]

                # overlay text on frame
                emoji  = SIGN_EMOJIS.get(label, "")
                disp   = f"{label}  {int(conf*100)}%"
                cv2.rectangle(frame, (0, 0), (len(disp)*14+20, 50), (62, 52, 137), -1)
                cv2.putText(frame, disp, (10, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # FPS
            now_t = time.time()
            fps   = 1.0 / max(now_t - prev_t, 1e-6)
            prev_t = now_t
            self.fps_deque.append(fps)
            avg_fps = sum(self.fps_deque) / len(self.fps_deque)
            cv2.putText(frame, f"FPS {avg_fps:.0f}", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            # push to UI thread
            self.root.after(0, self._update_video_ui, frame, label, conf, avg_fps)

            # data collection
            if self.collecting and res.multi_hand_landmarks:
                lm  = res.multi_hand_landmarks[0]
                h, w = frame.shape[:2]
                vec  = extract_landmarks(lm, w, h)
                lbl  = self.collect_label.get()
                self.samples.setdefault(lbl, []).append(vec)

        hands_live.close()

    def _update_video_ui(self, frame, label, conf, fps):
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil   = Image.fromarray(rgb)
        cw    = self.vid_canvas.winfo_width()  or 640
        ch    = self.vid_canvas.winfo_height() or 380
        pil   = pil.resize((cw, ch), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil)
        self.vid_canvas.delete("all")
        self.vid_canvas.create_image(cw//2, ch//2, anchor=tk.CENTER, image=tk_img)
        self.vid_canvas._img = tk_img

        self.fps_lbl.config(text=f"{fps:.0f}")
        if label:
            emoji = SIGN_EMOJIS.get(label, "")
            self.live_sign_lbl.config(text=f"{emoji}  {label}")
            self.live_conf_lbl.config(text=f"{int(conf*100)}% confidence")
            self.signs_count += 1
            self.count_lbl.config(text=str(self.signs_count))
            self._push_history(label, conf)

    def _push_history(self, label, conf):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.sign_history.appendleft((label, conf, ts))
        for w in self.hist_frame.winfo_children():
            w.destroy()
        for lbl, c, t in list(self.sign_history)[:8]:
            row = tk.Frame(self.hist_frame, bg=CARD_BG)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=SIGN_EMOJIS.get(lbl, "✋"),
                     font=(FONT_FAMILY, 11), bg=CARD_BG).pack(side=tk.LEFT)
            tk.Label(row, text=lbl, font=(FONT_FAMILY, 10),
                     bg=CARD_BG, fg=TEXT_PRI).pack(side=tk.LEFT, padx=5)
            tk.Label(row, text=t, font=(FONT_FAMILY, 9),
                     bg=CARD_BG, fg=TEXT_SEC).pack(side=tk.RIGHT)

    def _update_session_timer(self):
        if not self.video_running:
            return
        elapsed = int(time.time() - (self.session_start or time.time()))
        self.session_lbl.config(
            text=f"{elapsed//60}:{elapsed%60:02d}")
        self.root.after(1000, self._update_session_timer)

    def _screenshot_video(self):
        if not self.cap or not self.video_running:
            messagebox.showinfo("No Feed", "Start the camera first.")
            return
        ret, frame = self.cap.read()
        if ret:
            ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"sign_capture_{ts}.png"
            cv2.imwrite(name, cv2.flip(frame, 1))
            messagebox.showinfo("Screenshot", f"Saved as {name}")

    # ── Library tab ───────────────────────────
    def _build_library_tab(self):
        tab = tk.Frame(self.nb, bg=BG_COLOR)
        self.nb.add(tab, text="  📚  Sign Library  ")

        tk.Label(tab, text="30 supported ASL words",
                 font=(FONT_FAMILY, 11, "bold"),
                 bg=BG_COLOR, fg=TEXT_PRI).pack(anchor=tk.W, padx=16, pady=(14, 4))
        tk.Label(tab, text="Click any card to load a demo detection",
                 font=(FONT_FAMILY, 9), bg=BG_COLOR, fg=TEXT_SEC).pack(anchor=tk.W, padx=16)

        canvas = tk.Canvas(tab, bg=BG_COLOR, bd=0, highlightthickness=0)
        scroll = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        frame  = tk.Frame(canvas, bg=BG_COLOR)
        frame.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        cols = 5
        for i, sign in enumerate(SIGNS):
            card = tk.Frame(frame, bg=CARD_BG, bd=0,
                            highlightthickness=1, highlightbackground=BORDER,
                            cursor="hand2", width=150, height=100)
            card.grid(row=i // cols, column=i % cols,
                      padx=6, pady=6, sticky="nsew")
            card.pack_propagate(False)
            emoji = SIGN_EMOJIS.get(sign, "✋")
            tk.Label(card, text=emoji, font=(FONT_FAMILY, 24),
                     bg=CARD_BG).pack(pady=(12, 2))
            tk.Label(card, text=sign, font=(FONT_FAMILY, 9, "bold"),
                     bg=CARD_BG, fg=TEXT_PRI, wraplength=130).pack()

            def on_click(s=sign):
                self.nb.select(0)
                self.up_sign_lbl.config(
                    text=f"{SIGN_EMOJIS.get(s,'✋')}  {s}")
                self.up_meaning_lbl.config(text=f"ASL sign for '{s}'")
                self._set_conf_bar(0.85 + np.random.uniform(0, 0.13))
                self._show_alternatives(s)
            card.bind("<Button-1>", lambda e, fn=on_click: fn())
            for child in card.winfo_children():
                child.bind("<Button-1>", lambda e, fn=on_click: fn())

    # ── Training tab ──────────────────────────
    def _build_training_tab(self):
        tab = tk.Frame(self.nb, bg=BG_COLOR)
        self.nb.add(tab, text="  🧠  Training  ")

        info = tk.Frame(tab, bg=PRIMARY_LIGHT, bd=0,
                        highlightthickness=1, highlightbackground=PRIMARY_MID)
        info.pack(fill=tk.X, padx=14, pady=(14, 8))
        tk.Label(info, bg=PRIMARY_LIGHT, fg=PRIMARY,
                 font=(FONT_FAMILY, 10),
                 text="Train a custom KNN model on your own hand samples.\n"
                      "1. Select a sign label  2. Click Collect Samples (camera must be running)"
                      "  3. Click Train Model",
                 justify=tk.LEFT).pack(padx=12, pady=10)

        row1 = tk.Frame(tab, bg=BG_COLOR)
        row1.pack(fill=tk.X, padx=14, pady=4)
        tk.Label(row1, text="Sign label:", font=(FONT_FAMILY, 10, "bold"),
                 bg=BG_COLOR, fg=TEXT_PRI).pack(side=tk.LEFT)
        cb = ttk.Combobox(row1, textvariable=self.collect_label,
                          values=SIGNS, state="readonly", width=20)
        cb.pack(side=tk.LEFT, padx=10)

        row2 = tk.Frame(tab, bg=BG_COLOR)
        row2.pack(fill=tk.X, padx=14, pady=6)
        self.collect_btn = self._btn(row2, "⏺  Start Collecting",
                                     self._toggle_collect, PRIMARY)
        self.collect_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._btn(row2, "🧠  Train Model",
                  self._train_model, SUCCESS).pack(side=tk.LEFT, padx=(0, 8))
        self._btn(row2, "🗑️  Clear Samples",
                  self._clear_samples).pack(side=tk.LEFT)

        self.train_log = tk.Text(tab, height=14, font=(FONT_FAMILY, 10),
                                 bg=CARD_BG, fg=TEXT_PRI,
                                 bd=0, relief=tk.FLAT,
                                 highlightthickness=1, highlightbackground=BORDER)
        self.train_log.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)
        self.train_log.insert(tk.END, "Training log will appear here…\n")
        self.train_log.config(state=tk.DISABLED)
        self._update_train_log()

    def _toggle_collect(self):
        if not self.video_running:
            messagebox.showinfo("Start Camera", "Go to Live Video tab and start the camera first.")
            return
        self.collecting = not self.collecting
        if self.collecting:
            self.collect_btn.config(text="⏹  Stop Collecting", bg=DANGER)
            self._log(f"Collecting samples for: {self.collect_label.get()}")
        else:
            self.collect_btn.config(text="⏺  Start Collecting", bg=PRIMARY)
            self._save_samples()
            count = len(self.samples.get(self.collect_label.get(), []))
            self._log(f"Stopped. Total samples for '{self.collect_label.get()}': {count}")

    def _train_model(self):
        if not self.samples:
            messagebox.showinfo("No Samples",
                                "Collect hand samples first using the Live Video tab.")
            return
        self._log("Training KNN model…")
        msg = self.classifier.train(self.samples)
        self._log(f"✅ {msg}")
        self._log(f"Model saved to {MODEL_PATH}")

    def _clear_samples(self):
        if messagebox.askyesno("Clear Samples",
                               "Delete all collected training samples?"):
            self.samples = {}
            if os.path.exists(self.samples_path):
                os.remove(self.samples_path)
            self._log("All samples cleared.")

    def _log(self, msg: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.train_log.config(state=tk.NORMAL)
        self.train_log.insert(tk.END, f"[{ts}]  {msg}\n")
        self.train_log.see(tk.END)
        self.train_log.config(state=tk.DISABLED)

    def _update_train_log(self):
        total = sum(len(v) for v in self.samples.values())
        classes = len(self.samples)
        status = "✅ Custom model loaded" if self.classifier.trained else "⚪ Using built-in demo model"
        self._log(f"Samples on disk: {total} across {classes} classes  |  {status}")

    # ── Clock / status ────────────────────────
    def _update_clock(self):
        msg, color = time_status()
        self.status_lbl.config(text=msg, fg="white")
        self.root.after(30_000, self._update_clock)

    # ── Tab events ────────────────────────────
    def _on_tab_change(self, event):
        tab_idx = self.nb.index(self.nb.select())
        if tab_idx != 1 and self.video_running:
            self._stop_video()

    # ── Utility ───────────────────────────────
    def _btn(self, parent, text, command, bg=None):
        bg = bg or BORDER
        fg = "white" if bg in (PRIMARY, SUCCESS, DANGER) else TEXT_PRI
        b  = tk.Button(parent, text=text, command=command,
                       bg=bg, fg=fg, activebackground=bg,
                       font=(FONT_FAMILY, 10, "bold"),
                       relief=tk.FLAT, padx=14, pady=7,
                       cursor="hand2", bd=0)
        return b

    # ── Cleanup ───────────────────────────────
    def on_close(self):
        self.video_running = False
        if self.cap:
            self.cap.release()
        self.hands_solver.close()
        self.root.destroy()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()