# GestureAR: Real-Time Hand Gesture-Controlled Augmented Reality Interface

A webcam-based Augmented Reality (AR) project that lets you control a virtual 3D cube with hand gestures. No headset, mouse, or keyboard is needed. The system tracks your hand, recognizes the gesture with a machine learning classifier trained on your own data, and uses it to move, resize, rotate, or freeze a 3D object drawn over the live video.

Developed as a project for the **AR & VR** subject (8th semester, Integrated M.Sc. Computer Science, AI/ML specialization).

---

## Features

- Real-time hand tracking (21 landmarks per hand) using MediaPipe
- Custom gesture classifier trained on a self-collected dataset
- Four gestures mapped to AR actions
- Virtual 3D wireframe cube with perspective projection, drawn from scratch with OpenCV and NumPy
- Prediction smoothing (majority vote over 5 frames) and a confidence threshold to avoid flicker
- Runs on a normal laptop CPU with a webcam (no GPU required)

## Gestures and Actions

| Gesture | Hand shape | Action on the cube |
|---|---|---|
| Open palm | All five fingers spread | **Move**: the cube follows your palm |
| Pinch | Thumb and index fingertips together, other fingers extended | **Scale**: the gap between thumb and index sets the size |
| Point | Only the index finger extended | **Rotate**: moving the fingertip turns the cube |
| Fist | All fingers closed | **Freeze**: the cube stops and turns orange |

## How It Works

```
Webcam frame
    -> MediaPipe Hand Landmarker (21 points per hand)
    -> Normalization (wrist-relative, scale-independent, 63 features)
    -> Gesture classifier (SVM / Random Forest / Neural Network)
    -> Smoothing and confidence check
    -> Gesture-to-action mapping (move / scale / rotate / freeze)
    -> 3D cube transform (rotation, perspective, scaling)
    -> Cube drawn over the live video
```

1. **Hand tracking:** MediaPipe returns 21 landmark points (x, y, z) for the hand in each frame.
2. **Normalization:** landmarks are made relative to the wrist and divided by hand size, so the features do not depend on where the hand is on screen or how far it is from the camera.
3. **Classification:** a classifier trained on the collected data predicts one of the four gestures. Three models (Random Forest, SVM, small neural network) are compared and the best is saved.
4. **Action mapping:** the recognized gesture decides which cube property changes (position, size, or rotation angles).
5. **AR rendering:** the 8 corners of the cube are rotated, projected with a simple perspective model, and drawn as lines on the video frame.

## Tech Stack

- Python 3.10 to 3.14 (tested on Windows)
- OpenCV: webcam capture and drawing
- MediaPipe: hand landmark detection
- NumPy and pandas: data handling and 3D math
- scikit-learn: classifiers, evaluation
- Matplotlib: confusion matrix plot
- joblib: saving the trained model

## Project Structure

```
GestureAR/
├── data/
│   └── gestures.csv            # collected landmark dataset
├── models/
│   ├── gesture_model.pkl       # trained classifier
│   └── confusion_matrix.png    # evaluation plot
├── hand_landmarker.task        # MediaPipe hand model (downloaded)
├── hand_test.py                # check hand tracking works
├── collect_data.py             # record gesture samples into the CSV
├── train_model.py              # train, compare, evaluate, and save models
├── live_recognition.py         # live gesture recognition with confidence and FPS
├── cube_test.py                # test the 3D cube with keyboard controls
├── gesture_ar.py               # main app: gesture-controlled AR cube
├── utils.py                    # shared helpers (normalization, drawing, cube)
└── README.md
```

## Setup (Windows)

Open Command Prompt in the project folder.

```
python -m venv gesturear_env
gesturear_env\Scripts\activate
python -m pip install --upgrade pip
pip install mediapipe opencv-python numpy pandas scikit-learn matplotlib joblib
```

Download the MediaPipe hand model into the project folder:

```
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Optional: save your package versions for reproducibility.

```
pip freeze > requirements.txt
```

> If MediaPipe fails to install on a very new Python version, install Python 3.11 and create the environment with `py -3.11 -m venv gesturear_env`.

## Usage

### 1. Test hand tracking

```
python hand_test.py
```

A green skeleton should follow your hand. Press `q` to quit.

### 2. Collect gesture data

```
python collect_data.py
```

| Key | Action |
|---|---|
| `1` / `2` / `3` / `4` | Select open_palm / pinch / point / fist |
| `SPACE` | Start or stop recording |
| `q` | Quit |

Record roughly 400 to 1,000+ samples per gesture. Vary the distance, tilt, position in the frame, and use both hands. The script appends to `data/gestures.csv`, so you can run it again to add more data.

### 3. Train the classifier

```
python train_model.py
```

This compares Random Forest, SVM, and a neural network using a block-wise train/test split (consecutive frames stay together so near-duplicate frames do not inflate the accuracy). The best model is retrained on all data and saved to `models/gesture_model.pkl`, and a confusion matrix is saved to `models/confusion_matrix.png`.

### 4. Check live recognition (optional)

```
python live_recognition.py
```

Shows the predicted gesture, confidence, and FPS.

### 5. Run the AR application

```
python gesture_ar.py
```

| Key | Action |
|---|---|
| `r` | Reset the cube |
| `q` | Quit |

## Tuning

The constants at the top of `gesture_ar.py` control the feel of the interaction.

| Constant | Effect |
|---|---|
| `MIN_SIZE`, `MAX_SIZE` | Smallest and largest cube size (pixels) |
| `PINCH_RANGE` | Thumb-index gap that maps to the largest size (lower = more sensitive) |
| `ROT_SPEED` | Rotation speed when pointing |
| `SMOOTH` | Motion smoothing (lower = smoother but slower) |
| `CONF_MIN` | Minimum model confidence required to act on a gesture |

If the rotation direction feels reversed, flip the signs of the `ry` and `rx` updates in the point branch.

## Results

Fill in these values from your own runs.

| Item | Value |
|---|---|
| Dataset size (samples per gesture) | ... |
| Best model | ... |
| Test accuracy (block-wise split) | ... % |
| Real-time speed | ... FPS |

Include `models/confusion_matrix.png` in the report, and note performance under different lighting, distances, and hands.

## Limitations

- Works with one hand at a time.
- Accuracy depends on lighting and on the hand being fully inside the frame.
- The classifier was trained on data from a single user, so other users may see lower accuracy.
- The cube is a wireframe drawn with a simple perspective model, not a full 3D engine.
- Pinch and point can be confused when the fingers are not clearly formed.

## Future Work

- Add more gestures (for example thumbs-up to change color)
- Support two hands
- Replace the cube with real 3D models (OpenGL or a game engine)
- Collect data from multiple users for better generalization
- Deploy as a web app with MediaPipe in the browser

## Applications

Touchless presentation control, smart classrooms, gaming, virtual try-on, and settings where touching a screen is impractical, such as operating rooms or workshops.

## Author

Mathew, Integrated M.Sc. Computer Science (AI/ML), Bharata Mata College, Thrikkakara, Kerala, India.

## Acknowledgements

- [MediaPipe](https://ai.google.dev/edge/mediapipe) by Google for hand landmark detection
- OpenCV and scikit-learn open-source communities
