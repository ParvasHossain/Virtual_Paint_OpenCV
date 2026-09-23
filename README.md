# Geometric AI Virtual Paint

An interactive, real-time Computer Vision application that turns your webcam into a virtual canvas using hand gestures. Powered by **OpenCV** and **MediaPipe Hand Landmarker**, this app utilizes relative palm geometry to detect finger states robustly across various distances and hand sizes.

---

## 📸 Demo & Interface

The application features an intuitive overlay UI with color selection palettes and geometric visualization vectors:

- **Top Navigation Bar**: Select drawing colors, dynamic eraser, or clear the screen completely.
- **Geometric Overlay**: Visualizes hand anchor points, palm centroid, bounding circle reference metrics, and tracking vectors.

---

## ✨ Features

- **Geometric Normalization**: Finger extensions are calculated relative to the user's palm center and scale. This ensures stable gesture detection regardless of how close or far your hand is from the camera.
- **Auto Model Management**: Automatically downloads the required MediaPipe `hand_landmarker.task` model file on the first run if missing.
- **Real-Time Interactive Canvas**: Smooth line rendering using OpenCV array masks and bitwise operations.
- **Gestural Color Selection**: Easily pick colors or switch tools by pointing at the header UI.

---

## ✋ Gestures & Controls

| Gesture Mode | Finger Configuration | Action |
| :--- | :--- | :--- |
| **Selection Mode** | **Index + Middle Extended** (Ring & Pinky folded) | Hover over top menu boxes to choose colors, eraser, or clear canvas. Line painting is paused during this mode. |
| **Drawing Mode** | **Index Extended Only** (Middle folded) | Draw onto the virtual canvas using the selected color or brush thickness. |
| **Idle / Pause** | Fist, Palm Flat, or No Hand Detected | Resets drawing vectors and prevents unintended line drawing across the canvas. |

### Color Palette Options
- **BLUE / GREEN / RED / YELLOW / PURPLE**: Standard brush colors ($10\text{px}$ width).
- **ERASER**: Converts the tip into an eraser ($50\text{px}$ width).
- **CLEAR**: Instantly resets the entire canvas layer.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.8+** installed on your system.

### 2. Clone or Download Project
```bash
git clone https://github.com/your-username/virtual-paint.git
cd virtual-paint
```

### 3. Install Dependencies
Install the required packages using `pip`:

```bash
pip install opencv-python numpy mediapipe
```

---

## 🚀 Running the Application

Execute the Python script:

```bash
python virtual_paint.py
```

1. **First Launch**: The program will automatically download `hand_landmarker.task` from Google's MediaPipe repository if it isn't found in the directory.
2. **Controls**:
   - Raise your hand in front of the camera.
   - Use two fingers (Index + Middle) to select a color from the menu on top.
   - Fold your middle finger to start drawing with your index finger.
   - Press **`q`** on your keyboard to exit the application.

---

## 📐 How It Works (Geometry Engine)

Instead of relying on simple pixel threshold coordinates, the program uses relative Euclidean distance ratios:

1. **Palm Centroid Calculation**: Computes the mean position between the wrist ($L_0$), index MCP ($L_5$), pinky MCP ($L_{17}$), and middle MCP ($L_9$).
2. **Dynamic Scale Factor**: Measures the distance across knuckles (Index MCP to Pinky MCP) to determine scale ($S_{\text{palm}}$).
3. **Normalized Finger Distance Ratio**:
   $$\text{Ratio} = \frac{\text{Distance}(\text{Fingertip}, \text{Palm Centroid})}{S_{\text{palm}}}$$
   If the ratio exceeds $1.25$, the finger is recognized as **extended**, creating a robust tracking model under varying perspective distortions.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
```

### Summary of Generated Documentation:
- **Overview**: Highlights core stack (OpenCV, MediaPipe, NumPy).
- **Gesture Table**: Outlines finger gesture triggers (Selection vs. Drawing mode).
- **Setup Guide**: Step-by-step instructions to run dependencies and launch the app.
- **Mathematical Explanation**: Documents how the distance normalization formula works.