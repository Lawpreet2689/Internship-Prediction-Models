"""
car_color_detection_app.py
==========================

What it does:
- Detects cars and people in a traffic image (YOLOv8 via `ultralytics`)
- Classifies each car's dominant color using HSV analysis
- Draws a RED rectangle around BLUE cars, and a BLUE rectangle around
  every other colored car
- Draws a GREEN rectangle around each detected person and shows the
  people count
- Provides a Tkinter GUI: load an image, preview it, run detection,
  see the annotated result + a results panel, and save the output

-----------------------------------------------------------------------
SETUP
-----------------------------------------------------------------------
Install dependencies (these are real third-party packages -- the rest
of this file is NOT a package, just plain code, so there is nothing
else to "install"):

    pip install opencv-python ultralytics pillow numpy

Run:

    python car_color_detection_app.py

The first time you click "Run Detection", `ultralytics` will
auto-download the YOLOv8-nano weights (~6 MB) -- this needs internet
once; after that it works offline.

-----------------------------------------------------------------------
FILE MAP (this single file replaces all of the following, in order)
-----------------------------------------------------------------------
  1. color_classifier.py  -> HSV color classification of a car crop
  2. detector.py           -> YOLOv8 wrapper for car/truck/person detection
  3. pipeline.py            -> combines detection + color + drawing
  4. app.py                  -> Tkinter GUI
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


# =========================================================================
# 1. COLOR CLASSIFICATION  (formerly color_classifier.py)
# =========================================================================
#
# Classifies the dominant color of a cropped car image, and specifically
# flags whether it is BLUE or NOT BLUE.
#
# Approach:
#   1. Convert the crop to HSV (Hue, Saturation, Value) -- HSV separates
#      color (Hue) from brightness, which makes color classification far
#      more robust to lighting/shadow changes than raw RGB.
#   2. Mask out near-white, near-black, and gray pixels (low saturation),
#      since these are usually glass, shadows, road reflections, or
#      license plates -- not the car's paint.
#   3. Build a histogram over the remaining "colorful" pixels' Hue values
#      and find the dominant hue bucket.
#   4. Map the dominant hue to a human-readable color name using standard
#      HSV color-wheel ranges.
#   5. Return both the color name and a simple boolean: is_blue.

# OpenCV HSV ranges: H -> [0,179], S -> [0,255], V -> [0,255]
COLOR_RANGES = {
    "red":    [(0, 10), (170, 179)],   # red wraps around hue 0
    "orange": [(11, 22)],
    "yellow": [(23, 34)],
    "green":  [(35, 85)],
    "blue":   [(86, 130)],
    "purple": [(131, 155)],
    "pink":   [(156, 169)],
}


def _hue_in_ranges(hue, ranges):
    return any(lo <= hue <= hi for lo, hi in ranges)


def classify_hue(hue):
    """Map a single hue value (0-179) to a color name."""
    for name, ranges in COLOR_RANGES.items():
        if _hue_in_ranges(hue, ranges):
            return name
    return "unknown"


def get_dominant_color(bgr_image, sat_thresh=40, val_low=30):
    """
    Determine the dominant paint color of a cropped car image.

    Parameters
    ----------
    bgr_image : np.ndarray
        Cropped car region in BGR (as read by OpenCV).
    sat_thresh : int
        Minimum saturation to consider a pixel "colorful" (filters out
        white/gray/black/glare pixels -- these are always LOW saturation
        regardless of brightness).
    val_low : int
        Minimum brightness to keep; filters out near-black shadow pixels.

    Returns
    -------
    dict with keys:
        color_name : str   - e.g. "blue", "red", "white/light", "unknown"
        is_blue    : bool
        mean_hue   : float or None
        confidence : float  - fraction of pixels that were "colorful"
    """
    if bgr_image is None or bgr_image.size == 0:
        return {"color_name": "unknown", "is_blue": False, "mean_hue": None, "confidence": 0.0}

    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Keep only saturated, non-shadow pixels (actual paint, not glass/shadow/glare).
    # NOTE: we deliberately do NOT cap on high V -- a pure, fully-lit blue/red
    # car has V=255 and S=255 at the same time, so capping V would wrongly
    # discard real paint. Glare/white pixels are correctly excluded by the
    # saturation test alone (glare is low-saturation even when bright).
    colorful_mask = (s >= sat_thresh) & (v >= val_low)

    total_pixels = h.size
    colorful_pixels = int(np.count_nonzero(colorful_mask))
    confidence = colorful_pixels / total_pixels if total_pixels else 0.0

    if colorful_pixels < max(20, 0.03 * total_pixels):
        # Not enough colorful pixels -> likely a white, black, gray, or silver car
        mean_v = float(np.mean(v))
        if mean_v > 170:
            return {"color_name": "white/light", "is_blue": False, "mean_hue": None, "confidence": confidence}
        else:
            return {"color_name": "black/dark gray", "is_blue": False, "mean_hue": None, "confidence": confidence}

    hue_values = h[colorful_mask].astype(np.float32)

    # Histogram of hue values (36 bins of 5 degrees each, since OpenCV hue is 0-179)
    hist, bin_edges = np.histogram(hue_values, bins=36, range=(0, 180))
    dominant_bin = int(np.argmax(hist))
    dominant_hue = (bin_edges[dominant_bin] + bin_edges[dominant_bin + 1]) / 2.0

    color_name = classify_hue(dominant_hue)
    is_blue = color_name == "blue"

    return {
        "color_name": color_name,
        "is_blue": is_blue,
        "mean_hue": float(dominant_hue),
        "confidence": confidence,
    }


def box_color_for_car(is_blue):
    """
    Per task spec:
      - Blue cars  -> RED rectangle
      - Other cars -> BLUE rectangle
    Returns BGR tuple for OpenCV drawing.
    """
    return (0, 0, 255) if is_blue else (255, 0, 0)


# =========================================================================
# 2. DETECTION  (formerly detector.py)
# =========================================================================
#
# Wraps a YOLOv8 model (via the `ultralytics` package) to detect cars and
# people in an image. Uses YOLOv8's COCO-pretrained weights, which already
# include the classes we need:
#     class id 2 -> "car"
#     class id 7 -> "truck"   (catches SUVs/vans/pickups too)
#     class id 0 -> "person"
#
# NOTE: requires internet access the FIRST time it runs, so ultralytics
# can auto-download "yolov8n.pt" (~6MB). After that, it works offline.

CAR_CLASS_IDS = {2, 7}   # 2 = car, 7 = truck
PERSON_CLASS_ID = 0


class TrafficDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.35):
        """
        model_path: YOLOv8 weights file. "yolov8n.pt" (nano) is fastest and
                    is auto-downloaded by ultralytics if not already
                    present. Use "yolov8s.pt" / "yolov8m.pt" for better
                    accuracy at the cost of speed.
        conf_threshold: minimum detection confidence to keep a box.
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "The 'ultralytics' package is required for detection.\n"
                "Install it with:\n"
                "    pip install ultralytics\n"
                "Then re-run this application."
            )
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, bgr_image):
        """
        Run detection on a BGR image (as loaded by cv2.imread).

        Returns
        -------
        dict with keys:
            cars   : list of (x1, y1, x2, y2, confidence) tuples
            people : list of (x1, y1, x2, y2, confidence) tuples
        """
        results = self.model.predict(source=bgr_image, conf=self.conf_threshold, verbose=False)

        cars, people = [], []
        if not results:
            return {"cars": cars, "people": people}

        result = results[0]
        if result.boxes is None:
            return {"cars": cars, "people": people}

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

            if cls_id in CAR_CLASS_IDS:
                cars.append((x1, y1, x2, y2, conf))
            elif cls_id == PERSON_CLASS_ID:
                people.append((x1, y1, x2, y2, conf))

        return {"cars": cars, "people": people}


# =========================================================================
# 3. PIPELINE  (formerly pipeline.py)
# =========================================================================
#
# Combines detection (cars + people) with per-car color classification,
# and draws the final annotated image:
#     - RED rectangle around BLUE cars
#     - BLUE rectangle around all other car colors
#     - GREEN rectangle around people
#     - Summary text overlay: total car count, blue car count, people count

PERSON_BOX_COLOR = (0, 200, 0)  # green, BGR


def process_image(bgr_image, detector):
    """
    Run the full pipeline on a single BGR image.

    Parameters
    ----------
    bgr_image : np.ndarray
    detector  : object with a .detect(bgr_image) method returning
                {"cars": [(x1,y1,x2,y2,conf), ...], "people": [...]}
                (matches TrafficDetector above, but any object with the
                same interface -- e.g. a mock for testing -- works too)

    Returns
    -------
    annotated_image : np.ndarray (a copy of bgr_image with boxes/text drawn)
    summary : dict with counts and per-car details
    """
    annotated = bgr_image.copy()
    detections = detector.detect(bgr_image)

    car_results = []
    blue_count = 0

    for (x1, y1, x2, y2, conf) in detections["cars"]:
        crop = bgr_image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        color_info = get_dominant_color(crop)
        is_blue = color_info["is_blue"]
        if is_blue:
            blue_count += 1

        box_color = box_color_for_car(is_blue)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)

        label = f"{color_info['color_name']} ({conf:.2f})"
        label_y = max(0, y1 - 8)
        cv2.putText(annotated, label, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)

        car_results.append({
            "box": (x1, y1, x2, y2),
            "color_name": color_info["color_name"],
            "is_blue": is_blue,
            "confidence": conf,
        })

    people_count = len(detections["people"])
    for (x1, y1, x2, y2, conf) in detections["people"]:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), PERSON_BOX_COLOR, 2)
        cv2.putText(annotated, f"person ({conf:.2f})", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, PERSON_BOX_COLOR, 1, cv2.LINE_AA)

    # Summary overlay banner
    total_cars = len(car_results)
    summary_lines = [
        f"Cars detected: {total_cars}  (Blue: {blue_count}, Other: {total_cars - blue_count})",
        f"People detected: {people_count}",
    ]
    overlay_h = 30 + 28 * len(summary_lines)
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (520, overlay_h), (0, 0, 0), -1)
    annotated = cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0)

    for i, line in enumerate(summary_lines):
        cv2.putText(annotated, line, (12, 28 + 28 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    summary = {
        "total_cars": total_cars,
        "blue_cars": blue_count,
        "other_cars": total_cars - blue_count,
        "people_count": people_count,
        "car_details": car_results,
    }

    return annotated, summary


# =========================================================================
# 4. GUI  (formerly app.py)
# =========================================================================

MAX_PREVIEW_W = 480
MAX_PREVIEW_H = 360


def cv2_to_tk(bgr_image, max_w=MAX_PREVIEW_W, max_h=MAX_PREVIEW_H):
    """Convert an OpenCV BGR image to a Tkinter-displayable PhotoImage,
    resized (preserving aspect ratio) to fit within max_w x max_h."""
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    w, h = pil_img.size
    scale = min(max_w / w, max_h / h, 1.0)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    pil_img = pil_img.resize(new_size, Image.LANCZOS)

    return ImageTk.PhotoImage(pil_img)


class CarColorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Color Detection & Traffic Counter")
        self.root.geometry("1040x680")
        self.root.minsize(900, 600)

        self.image_path = None
        self.original_bgr = None
        self.annotated_bgr = None
        self.detector = None  # lazily loaded on first run

        self._build_ui()

    # ---------------------------------------------------------- UI layout
    def _build_ui(self):
        toolbar = ttk.Frame(self.root, padding=10)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.load_btn = ttk.Button(toolbar, text="Load Image", command=self.load_image)
        self.load_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.run_btn = ttk.Button(toolbar, text="Run Detection", command=self.run_detection, state=tk.DISABLED)
        self.run_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.save_btn = ttk.Button(toolbar, text="Save Annotated Image", command=self.save_result, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.path_label = ttk.Label(toolbar, text="No image loaded", foreground="gray")
        self.path_label.pack(side=tk.LEFT, padx=(10, 0))

        content = ttk.Frame(self.root, padding=10)
        content.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        previews = ttk.Frame(content)
        previews.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        orig_frame = ttk.LabelFrame(previews, text="Original Image", padding=5)
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.orig_canvas = tk.Label(orig_frame, background="#222", anchor="center")
        self.orig_canvas.pack(fill=tk.BOTH, expand=True)

        result_frame = ttk.LabelFrame(previews, text="Detection Result", padding=5)
        result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.result_canvas = tk.Label(result_frame, background="#222", anchor="center")
        self.result_canvas.pack(fill=tk.BOTH, expand=True)

        stats_frame = ttk.LabelFrame(content, text="Results", padding=10, width=260)
        stats_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))

        self.stats_text = tk.Text(stats_frame, width=32, height=28, state=tk.DISABLED,
                                   wrap=tk.WORD, background="#f7f7f7")
        self.stats_text.pack(fill=tk.BOTH, expand=True)

        legend = ttk.Label(
            stats_frame,
            text="Legend:\n🔴 Red box = Blue car\n🔵 Blue box = Other color car\n🟢 Green box = Person",
            justify=tk.LEFT,
        )
        legend.pack(pady=(8, 0), anchor="w")

        self.status_var = tk.StringVar(value="Ready. Load an image to begin.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w", padding=4)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ------------------------------------------------------------ actions
    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select traffic image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return

        bgr = cv2.imread(path)
        if bgr is None:
            messagebox.showerror("Error", f"Could not read image:\n{path}")
            return

        self.image_path = path
        self.original_bgr = bgr
        self.annotated_bgr = None

        tk_img = cv2_to_tk(bgr)
        self.orig_canvas.configure(image=tk_img)
        self.orig_canvas.image = tk_img  # keep a reference (avoid garbage collection)

        self.result_canvas.configure(image="")
        self.result_canvas.image = None

        self.path_label.configure(text=os.path.basename(path), foreground="black")
        self.run_btn.configure(state=tk.NORMAL)
        self.save_btn.configure(state=tk.DISABLED)
        self._set_stats_text("Image loaded. Click 'Run Detection' to analyze.")
        self.status_var.set(f"Loaded: {path}")

    def run_detection(self):
        if self.original_bgr is None:
            return
        self.run_btn.configure(state=tk.DISABLED)
        self.load_btn.configure(state=tk.DISABLED)
        self.status_var.set("Loading model / running detection... this may take a moment on first run.")
        self._set_stats_text("Processing...")

        # Run in a background thread so the GUI doesn't freeze, especially
        # during the first-time YOLO weight download / model load.
        thread = threading.Thread(target=self._run_detection_worker, daemon=True)
        thread.start()

    def _run_detection_worker(self):
        try:
            if self.detector is None:
                self.detector = TrafficDetector()

            annotated, summary = process_image(self.original_bgr, self.detector)
            self.annotated_bgr = annotated
            self.root.after(0, self._on_detection_done, summary)
        except ImportError as e:
            self.root.after(0, self._on_detection_error, str(e))
        except Exception as e:
            self.root.after(0, self._on_detection_error, f"Unexpected error: {e}")

    def _on_detection_done(self, summary):
        tk_img = cv2_to_tk(self.annotated_bgr)
        self.result_canvas.configure(image=tk_img)
        self.result_canvas.image = tk_img

        self._render_summary(summary)

        self.run_btn.configure(state=tk.NORMAL)
        self.load_btn.configure(state=tk.NORMAL)
        self.save_btn.configure(state=tk.NORMAL)
        self.status_var.set("Detection complete.")

    def _on_detection_error(self, message):
        self.run_btn.configure(state=tk.NORMAL)
        self.load_btn.configure(state=tk.NORMAL)
        self._set_stats_text(f"Error:\n{message}")
        self.status_var.set("Detection failed. See results panel.")
        messagebox.showerror("Detection Error", message)

    def save_result(self):
        if self.annotated_bgr is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save annotated image",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
        )
        if not path:
            return
        cv2.imwrite(path, self.annotated_bgr)
        self.status_var.set(f"Saved annotated image to: {path}")

    # ------------------------------------------------------------- helpers
    def _set_stats_text(self, text):
        self.stats_text.configure(state=tk.NORMAL)
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert(tk.END, text)
        self.stats_text.configure(state=tk.DISABLED)

    def _render_summary(self, summary):
        lines = []
        lines.append("=== TRAFFIC SUMMARY ===\n")
        lines.append(f"Total cars detected: {summary['total_cars']}")
        lines.append(f"  Blue cars:  {summary['blue_cars']}")
        lines.append(f"  Other cars: {summary['other_cars']}")
        lines.append("")
        lines.append(f"People detected: {summary['people_count']}")
        lines.append("")
        lines.append("--- Per-car details ---")
        for i, car in enumerate(summary["car_details"], start=1):
            tag = "BLUE -> RED BOX" if car["is_blue"] else "OTHER -> BLUE BOX"
            lines.append(f"{i}. {car['color_name']} (conf {car['confidence']:.2f}) [{tag}]")

        self._set_stats_text("\n".join(lines))


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    app = CarColorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
