#  Machine Learning Projects Collection

A collection of six machine learning projects covering computer vision, audio processing, and real-time detection — each featuring a custom-built model and graphical user interface.

---

## 📋 Table of Contents

- [Project 1 — Long Hair Gender Identification]
- [Project 2 — Senior Citizen Identification]
- [Project 3 — Age & Emotion Detection via Voice]
- [Project 4 — Sign Language Detection]
- [Project 5 — Car Colour Detection]
- [Project 6 — Nationality Detection]
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## 1. Long Hair Gender Identification

### 📌 Description
Detects gender based on **hair length** with a special age-based override rule:

| Age Range | Behaviour |
|-----------|-----------|
| 20–30 | Long hair → classified as Female; Short hair → classified as Male (regardless of actual gender) |
| Below 20 or Above 30 | Standard gender prediction (hair length ignored) |

### ✨ Features
- Custom-trained image classification model
- Age estimation module for range detection
- GUI for image upload and result display

```

---

## 2. Senior Citizen Identification

### 📌 Description
Real-time **multi-person detection** from video or webcam feed, suitable for malls or local stores. Detects age and gender for each person, flags those above 60 as senior citizens, and logs all data automatically.

| Person | Detection |
|--------|-----------|
| Age ≤ 60 | Age + Gender |
| Age > 60 | Age + Gender + ✅ Senior Citizen Tag |

### ✨ Features
- Multi-person detection in video/webcam feed
- Age & gender prediction per detected individual
- Automatic logging to **Excel / CSV** (age, gender, time of visit)
- Optional GUI


```

---

## 3. Age & Emotion Detection via Voice

### 📌 Description
Processes **voice notes** to estimate a speaker's age and emotion — but only for male voices. Female voices are automatically rejected.

| Condition | Output |
|-----------|--------|
| Female voice detected | ❌ "Upload male voice." |
| Male voice, Age < 60 | Age only |
| Male voice, Age ≥ 60 | Age + Emotion + ✅ Senior Citizen Tag |

### ✨ Features
- Gender classification from audio
- Age estimation from voice
- Emotion detection (for senior citizens)
- GUI with voice note upload


```

---

## 4. Sign Language Detection

### 📌 Description
Recognizes **sign language gestures** for a predefined set of words, with a built-in **time restriction** — the model only operates between **6:00 PM and 10:00 PM**.

### ✨ Features
- Custom-trained sign language recognition model
- Time-lock mechanism (active only 6 PM – 10 PM)
- GUI with:
  - 📁 Image upload mode
  - 🎥 Real-time video detection mode


```

> ⚠️ **Note:** The model will be inactive outside 6:00 PM – 10:00 PM.

---

## 5. Car Colour Detection

### 📌 Description
Detects **car colours** in traffic images/video and counts vehicles at a signal. Uses colour-coded bounding boxes to highlight detections and also counts people present.

| Detection | Bounding Box Colour |
|-----------|---------------------|
| Blue cars | 🔴 Red rectangle |
| All other cars | 🔵 Blue rectangle |
| People at signal | Displayed as count |

### ✨ Features
- Car colour classification
- Vehicle counting at traffic signals
- People detection and count overlay
- GUI with image preview panel


```

---

## 6. Nationality Detection

### 📌 Description
Predicts a person's **nationality and emotion** from a photo, with additional predictions based on detected nationality.

| Nationality | Additional Predictions |
|-------------|------------------------|
| 🇮🇳 Indian | Age + Dress Colour + Emotion |
| 🇺🇸 American | Age + Emotion |
| 🌍 African | Emotion + Dress Colour |
| Other | Nationality + Emotion only |

### ✨ Features
- Nationality classification from facial features
- Emotion detection
- Conditional prediction pipeline based on nationality
- GUI with image preview and structured output section


```

---

## 🛠 Tech Stack

| Category | Tools / Libraries |
|----------|-------------------|
| Language | Python 3.9+ |
| Deep Learning | TensorFlow / PyTorch |
| Computer Vision | OpenCV, MediaPipe |
| Audio Processing | Librosa, SpeechBrain |
| GUI | Tkinter / PyQt5 |
| Data Logging | Pandas, OpenPyXL |
| Model Training | scikit-learn, Keras |

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ml-projects-collection.git
   cd ml-projects-collection
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Navigate to any project folder and run**
   ```bash
   cd project1_long_hair_gender
   python app.py
   ```

---



## 📊 Evaluation Criteria

Each project is evaluated on:
- ✅ Model performance & accuracy
- ✅ GUI functionality and usability
- ✅ Correct implementation of business logic
- ✅ Code quality and structure

---


---

## 🙋 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

> Built with 💡 logic, 🧠 ML, and ☕ a lot of coffee.
