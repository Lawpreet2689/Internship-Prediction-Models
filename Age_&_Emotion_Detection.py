"""
Voice Age & Emotion Detection — Male-Only GUI
================================================
Logic:
  1. Detect gender from voice.
       - Female detected -> REJECT, show "Upload male voice."
  2. Male detected -> detect age.
       - Age > 60  -> mark "Senior Citizen" + detect emotion.
       - Age <= 60 -> show age only (no emotion check).

Install:
    pip install librosa soundfile numpy scikit-learn joblib pillow
    (optional, for mic recording):  pip install sounddevice

Run:
    python voice_age_emotion_gui.py
"""

import os
import sys
import time
import threading
import tempfile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

from voice_features import (
    load_audio, extract_features, VoiceAnalyzer,
    LIBROSA_AVAILABLE, SAMPLE_RATE,
)

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
#  COLORS
# ──────────────────────────────────────────────────────────────────────────────

BLUE   = "#185FA5"
GREEN  = "#0F6E56"
RED    = "#C0392B"
GOLD   = "#B7791F"
BG     = "#F4F3F1"
CARD   = "#FFFFFF"
BORDER = "#DEDBD2"
TEXT   = "#1C1C1A"
MUTED  = "#6B6A66"


SENIOR_AGE_THRESHOLD = 60


class VoiceAgeEmotionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Voice Age & Emotion Detection (Male Voices Only)")
        self.geometry("800x900")
        self.configure(bg=BG)
        self.resizable(True, True)

        self.audio_path = None
        self.analyzer   = VoiceAnalyzer()

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────

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

        tk.Label(f, text="Voice Age & Emotion Detection",
                 font=("Helvetica", 20, "bold"), bg=BG, fg=TEXT
                 ).pack(anchor="w", padx=24, pady=(20, 2))
        tk.Label(f, text="Male voices only. Detects age; if senior (60+), also detects emotion.",
                 font=("Helvetica", 11), bg=BG, fg=MUTED
                 ).pack(anchor="w", padx=24, pady=(0, 14))

        self._logic_card(f)
        self._model_status_card(f)
        self._upload_section(f)

        # Preview
        self.preview_frame = tk.Frame(f, bg=BG)
        self.preview_frame.pack(fill="x", padx=24, pady=(0, 0))

        # Buttons
        btn_row = tk.Frame(f, bg=BG)
        btn_row.pack(anchor="w", padx=24, pady=(12, 0))
        self.analyze_btn = tk.Button(
            btn_row, text="  Analyze Voice",
            font=("Helvetica", 12, "bold"), bg=BLUE, fg="white",
            relief="flat", padx=20, pady=8, cursor="hand2",
            command=self._start_analysis, state="disabled")
        self.analyze_btn.pack(side="left")
        tk.Button(btn_row, text="X  Clear",
                  font=("Helvetica", 11), bg=CARD, fg=MUTED,
                  relief="flat", padx=14, pady=8, cursor="hand2",
                  bd=1, highlightbackground=BORDER,
                  command=self._clear).pack(side="left", padx=(10, 0))

        # Status + progress
        self.status_var = tk.StringVar(value="")
        tk.Label(f, textvariable=self.status_var,
                 font=("Helvetica", 11), bg=BG, fg=MUTED
                 ).pack(anchor="w", padx=24, pady=(8, 0))
        self.progress = ttk.Progressbar(f, mode="indeterminate", length=750)
        self.progress.pack(padx=24, pady=(4, 0))

        # Result
        self.result_frame = tk.Frame(f, bg=BG)
        self.result_frame.pack(fill="x", padx=24, pady=(14, 24))

    def _logic_card(self, parent):
        card = tk.Frame(parent, bg="#E6F1FB")
        card.pack(fill="x", padx=24, pady=(0, 10))
        inner = tk.Frame(card, bg="#E6F1FB")
        inner.pack(fill="x", padx=14, pady=10)
        tk.Label(inner, text="Detection Logic", font=("Helvetica", 11, "bold"),
                 bg="#E6F1FB", fg=BLUE).pack(anchor="w")
        rules = [
            ("Step 1:", "Detect gender from voice. Female -> REJECTED, shows \"Upload male voice.\""),
            ("Step 2:", "Male voice -> detect age."),
            ("Step 3a:", "Age > 60 -> marked SENIOR CITIZEN, emotion is also detected."),
            ("Step 3b:", "Age <= 60 -> only age is shown (no emotion check)."),
        ]
        for label, desc in rules:
            row = tk.Frame(inner, bg="#E6F1FB")
            row.pack(anchor="w", pady=1)
            tk.Label(row, text=label, font=("Helvetica", 10, "bold"),
                     bg="#E6F1FB", fg=TEXT, width=8, anchor="w").pack(side="left")
            tk.Label(row, text=desc, font=("Helvetica", 10),
                     bg="#E6F1FB", fg=MUTED, wraplength=600, justify="left"
                     ).pack(side="left")

    def _model_status_card(self, parent):
        card = tk.Frame(parent, bg="#F0F7F0")
        card.pack(fill="x", padx=24, pady=(0, 12))
        inner = tk.Frame(card, bg="#F0F7F0")
        inner.pack(fill="x", padx=14, pady=10)
        tk.Label(inner, text="Model Status", font=("Helvetica", 10, "bold"),
                 bg="#F0F7F0", fg=GREEN).pack(anchor="w")

        def status_line(name, is_trained):
            txt = (f"{name}: " +
                   ("Trained RandomForest model loaded" if is_trained
                    else "Heuristic baseline active (no trained model found in models/)"))
            color = GREEN if is_trained else MUTED
            tk.Label(inner, text=txt, font=("Helvetica", 10),
                     bg="#F0F7F0", fg=color).pack(anchor="w")

        status_line("Gender model",  self.analyzer.using_trained_gender)
        status_line("Age model",     self.analyzer.using_trained_age)
        status_line("Emotion model", self.analyzer.using_trained_emotion)

        if not LIBROSA_AVAILABLE:
            tk.Label(inner, text="WARNING: librosa not installed — run: pip install librosa soundfile",
                     font=("Helvetica", 10, "bold"), bg="#F0F7F0", fg=RED
                     ).pack(anchor="w", pady=(4, 0))

    def _upload_section(self, parent):
        frame = tk.Frame(parent, bg=CARD, bd=1, relief="solid",
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill="x", padx=24, pady=(0, 12))
        inner = tk.Frame(frame, bg=CARD)
        inner.pack(pady=22)
        tk.Label(inner, text="[ Microphone ]", font=("Helvetica", 18, "bold"),
                 bg=CARD, fg=MUTED).pack()
        tk.Label(inner, text="Upload a voice recording or record live",
                 font=("Helvetica", 13, "bold"), bg=CARD, fg=TEXT
                 ).pack(pady=(6, 2))
        tk.Label(inner, text="WAV recommended (MP3/M4A may need ffmpeg)",
                 font=("Helvetica", 10), bg=CARD, fg=MUTED).pack()

        btn_row = tk.Frame(inner, bg=CARD)
        btn_row.pack(pady=(12, 0))
        tk.Button(btn_row, text="Browse Audio File",
                  font=("Helvetica", 11), bg=BG, fg=TEXT,
                  relief="flat", bd=1, highlightbackground=BORDER,
                  padx=14, pady=6, cursor="hand2",
                  command=self._browse_file).pack(side="left", padx=4)

        rec_state = "normal" if SOUNDDEVICE_AVAILABLE else "disabled"
        rec_text  = "Record 5s from Mic" if SOUNDDEVICE_AVAILABLE else "Mic unavailable (install sounddevice)"
        tk.Button(btn_row, text=rec_text,
                  font=("Helvetica", 11), bg=BG, fg=TEXT,
                  relief="flat", bd=1, highlightbackground=BORDER,
                  padx=14, pady=6, cursor="hand2", state=rec_state,
                  command=self._record_from_mic).pack(side="left", padx=4)

    # ── Actions ───────────────────────────────────────────────────────────

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select an audio file",
            filetypes=[("Audio files", "*.wav *.mp3 *.m4a *.flac *.ogg"),
                       ("All files", "*.*")])
        if path:
            self._set_audio(path, os.path.basename(path))

    def _record_from_mic(self):
        if not SOUNDDEVICE_AVAILABLE:
            messagebox.showerror("Unavailable", "Install sounddevice: pip install sounddevice")
            return
        self.status_var.set("Recording... speak now (5 seconds)")
        self.update_idletasks()
        threading.Thread(target=self._do_recording, daemon=True).start()

    def _do_recording(self):
        duration = 5
        try:
            audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                           channels=1, dtype="float32")
            sd.wait()
            tmp_path = os.path.join(tempfile.gettempdir(), "voice_recording.wav")
            sf.write(tmp_path, audio, SAMPLE_RATE)
            self.after(0, self._set_audio, tmp_path, "Mic recording (5s)")
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Recording Failed", str(e)))
            self.after(0, lambda: self.status_var.set(""))

    def _set_audio(self, path, display_name):
        self.audio_path = path
        for w in self.preview_frame.winfo_children():
            w.destroy()
        for w in self.result_frame.winfo_children():
            w.destroy()
        self.status_var.set("")

        row = tk.Frame(self.preview_frame, bg=CARD, bd=1, relief="solid",
                       highlightbackground=BORDER, highlightthickness=1)
        row.pack(fill="x")
        inner = tk.Frame(row, bg=CARD)
        inner.pack(fill="x", padx=14, pady=10)
        tk.Label(inner, text="[ Audio ]  " + display_name,
                 font=("Helvetica", 12, "bold"), bg=CARD, fg=TEXT
                 ).pack(anchor="w")
        try:
            size_kb = os.path.getsize(path) // 1024
            tk.Label(inner, text=f"{size_kb} KB",
                     font=("Helvetica", 10), bg=CARD, fg=MUTED
                     ).pack(anchor="w", pady=(2, 0))
        except OSError:
            pass

        self.analyze_btn.config(state="normal")

    def _clear(self):
        self.audio_path = None
        for w in self.preview_frame.winfo_children():
            w.destroy()
        for w in self.result_frame.winfo_children():
            w.destroy()
        self.status_var.set("")
        self.analyze_btn.config(state="disabled")

    def _start_analysis(self):
        if not self.audio_path:
            return
        if not LIBROSA_AVAILABLE:
            messagebox.showerror("Missing Library",
                "librosa is required.\nRun: pip install librosa soundfile")
            return

        self.analyze_btn.config(state="disabled", text="Analyzing...")
        self.status_var.set("Extracting acoustic features...")
        for w in self.result_frame.winfo_children():
            w.destroy()
        self.progress.start(10)

        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        try:
            y, sr = load_audio(self.audio_path)
            feats = extract_features(y, sr)
        except Exception as e:
            self.after(0, self._on_error, str(e))
            return
        self.after(0, self._on_features_ready, feats)

    def _on_features_ready(self, feats):
        self.status_var.set("Running gender detection...")

        gender, gender_conf, gender_src = self.analyzer.predict_gender(feats)

        if gender == "Female":
            self.progress.stop()
            self.analyze_btn.config(state="normal", text="  Analyze Voice")
            self.status_var.set("Rejected: female voice detected.")
            self._render_rejected(feats, gender_conf, gender_src)
            return

        # Male -> detect age
        self.status_var.set("Male voice confirmed. Detecting age...")
        age, age_src = self.analyzer.predict_age(feats)

        is_senior = age > SENIOR_AGE_THRESHOLD
        emotion_result = None
        if is_senior:
            self.status_var.set("Senior citizen detected. Detecting emotion...")
            emotion_result = self.analyzer.predict_emotion(feats)

        self.progress.stop()
        self.analyze_btn.config(state="normal", text="  Analyze Voice")
        self.status_var.set("Analysis complete.")
        self._render_accepted(feats, gender_conf, gender_src, age, age_src,
                              is_senior, emotion_result)

    def _on_error(self, msg):
        self.progress.stop()
        self.analyze_btn.config(state="normal", text="  Analyze Voice")
        self.status_var.set("")
        messagebox.showerror("Analysis Failed", msg)

    # ── Result rendering ─────────────────────────────────────────────────

    def _feature_summary_card(self, parent, feats):
        card = tk.Frame(parent, bg="#F7F6F3", bd=1, relief="solid",
                        highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill="x", pady=(0, 12))
        inner = tk.Frame(card, bg="#F7F6F3")
        inner.pack(fill="x", padx=14, pady=10)
        tk.Label(inner, text="Extracted Acoustic Features (transparency)",
                 font=("Helvetica", 10, "bold"), bg="#F7F6F3", fg=TEXT
                 ).pack(anchor="w")
        line = (f"Pitch (median): {feats['pitch_median']:.0f} Hz   |   "
                f"Jitter: {feats['jitter']*100:.2f}%   |   "
                f"Shimmer: {feats['shimmer']*100:.2f}%   |   "
                f"Tempo: {feats['tempo']:.0f}   |   "
                f"Duration: {feats['duration_sec']:.1f}s")
        tk.Label(inner, text=line, font=("Helvetica", 9), bg="#F7F6F3", fg=MUTED,
                 wraplength=730, justify="left").pack(anchor="w", pady=(4, 0))

    def _render_rejected(self, feats, conf, source):
        f = self.result_frame
        banner = tk.Frame(f, bg=RED)
        banner.pack(fill="x", pady=(0, 12))
        inner = tk.Frame(banner, bg=RED)
        inner.pack(pady=16, padx=16, anchor="w")
        tk.Label(inner, text="REJECTED",
                 font=("Helvetica", 18, "bold"), bg=RED, fg="white").pack(anchor="w")
        tk.Label(inner, text="Upload male voice.",
                 font=("Helvetica", 14), bg=RED, fg="white").pack(anchor="w", pady=(2, 0))
        tk.Label(inner, text=f"Detected gender: Female  (confidence {conf*100:.0f}%)  ·  {source}",
                 font=("Helvetica", 10), bg=RED, fg="#FFD9D5").pack(anchor="w", pady=(6, 0))

        self._feature_summary_card(f, feats)

    def _render_accepted(self, feats, gender_conf, gender_src,
                         age, age_src, is_senior, emotion_result):
        f = self.result_frame
        banner_color = GOLD if is_senior else BLUE

        banner = tk.Frame(f, bg=banner_color)
        banner.pack(fill="x", pady=(0, 12))
        inner = tk.Frame(banner, bg=banner_color)
        inner.pack(pady=16, padx=16, anchor="w")

        if is_senior:
            tk.Label(inner, text=">> SENIOR CITIZEN <<",
                     font=("Helvetica", 18, "bold"), bg=banner_color, fg="white"
                     ).pack(anchor="w")
        else:
            tk.Label(inner, text="Male Voice — Age Detected",
                     font=("Helvetica", 18, "bold"), bg=banner_color, fg="white"
                     ).pack(anchor="w")

        tk.Label(inner, text=f"Estimated Age: {age} years",
                 font=("Helvetica", 14), bg=banner_color, fg="white"
                 ).pack(anchor="w", pady=(2, 0))

        if emotion_result:
            emo_label, emo_conf, emo_src = emotion_result
            tk.Label(inner, text=f"Detected Emotion: {emo_label}  ({emo_conf*100:.0f}% confidence)",
                     font=("Helvetica", 13, "bold"), bg=banner_color, fg="#FFF6E0"
                     ).pack(anchor="w", pady=(6, 0))

        # Metric grid
        grid = tk.Frame(f, bg=BG)
        grid.pack(fill="x", pady=(0, 12))
        metrics = [
            ("Gender",  f"Male ({gender_conf*100:.0f}%)", TEXT),
            ("Age",     f"{age} yrs", TEXT),
            ("Status",  "Senior Citizen" if is_senior else "Adult",
             GOLD if is_senior else GREEN),
            ("Age Model Source", age_src.split(" ")[0], BLUE),
        ]
        for i, (lbl, val, vc) in enumerate(metrics):
            cell = tk.Frame(grid, bg=CARD, bd=1, relief="solid",
                            highlightbackground=BORDER, highlightthickness=1)
            cell.grid(row=0, column=i, padx=4, sticky="nsew")
            grid.columnconfigure(i, weight=1)
            tk.Label(cell, text=lbl, font=("Helvetica", 9), bg=CARD, fg=MUTED,
                     wraplength=160, justify="center").pack(pady=(10, 2), padx=6)
            tk.Label(cell, text=val, font=("Helvetica", 11, "bold"), bg=CARD, fg=vc,
                     wraplength=170, justify="center").pack(pady=(0, 10), padx=6)

        if emotion_result:
            emo_label, emo_conf, emo_src = emotion_result
            emo_card = tk.Frame(f, bg="#FFF8E8", bd=1, relief="solid",
                                highlightbackground=BORDER, highlightthickness=1)
            emo_card.pack(fill="x", pady=(0, 12))
            inner2 = tk.Frame(emo_card, bg="#FFF8E8")
            inner2.pack(fill="x", padx=14, pady=10)
            tk.Label(inner2, text="Emotion Detection (Senior Citizens Only)",
                     font=("Helvetica", 10, "bold"), bg="#FFF8E8", fg=GOLD
                     ).pack(anchor="w")
            tk.Label(inner2, text=f"Result: {emo_label}   |   Source: {emo_src}",
                     font=("Helvetica", 10), bg="#FFF8E8", fg=MUTED
                     ).pack(anchor="w", pady=(2, 0))

        self._feature_summary_card(f, feats)

        # Logic explanation
        exp = tk.Frame(f, bg="#F0F4F8", bd=1, relief="solid",
                       highlightbackground=BORDER, highlightthickness=1)
        exp.pack(fill="x", pady=(0, 12))
        tk.Label(exp, text="Decision Path", font=("Helvetica", 11, "bold"),
                 bg="#F0F4F8", fg=TEXT).pack(anchor="w", padx=14, pady=(10, 2))
        if is_senior:
            explanation = (
                f"Gender detected as Male ({gender_src}).\n"
                f"Age estimated at {age} years ({age_src}) — exceeds the senior "
                f"threshold of {SENIOR_AGE_THRESHOLD}.\n"
                f"-> Marked as SENIOR CITIZEN. Emotion detection was triggered."
            )
        else:
            explanation = (
                f"Gender detected as Male ({gender_src}).\n"
                f"Age estimated at {age} years ({age_src}) — at or below the senior "
                f"threshold of {SENIOR_AGE_THRESHOLD}.\n"
                f"-> Only age is reported. Emotion detection was skipped (per spec)."
            )
        tk.Label(exp, text=explanation, font=("Helvetica", 11), bg="#F0F4F8", fg=MUTED,
                 wraplength=730, justify="left").pack(anchor="w", padx=14, pady=(0, 10))


def main():
    if not LIBROSA_AVAILABLE:
        print("ERROR: librosa not installed.")
        print("Run:  pip install librosa soundfile numpy scikit-learn joblib")
        input("Press Enter to exit...")
        return

    app = VoiceAgeEmotionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
