# Udacity Self Driving Car🚗

An end-to-end self-driving car project using the **Udacity Self-Driving Car Simulator**, **TensorFlow/Keras**, and the **NVIDIA Behavioral Cloning Network**.

This project trains a Convolutional Neural Network (CNN) to predict the steering angle from **center camera images** collected while driving manually in the simulator. The trained model is then used to drive the vehicle autonomously.

---

# Project Overview

The project consists of only **two Python files**:

* **train.py** – Trains the NVIDIA CNN and saves the model.
* **drive.py** – Loads the trained model and controls the vehicle inside the simulator.

This simple structure makes the project easy to understand while demonstrating the complete behavioral cloning workflow.

---

# Project Structure

```text
SelfDrivingCNN/
│
├── train.py
├── drive.py
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── driving_log.csv
│   └── IMG/
│       ├── center_000001.jpg
│       ├── center_000002.jpg
│       └── ...
│
└── model.h5
```

---

# Requirements

* Python 3.10 or later
* TensorFlow
* NumPy
* Pandas
* OpenCV
* Pillow
* Flask
* Eventlet
* python-socketio
* scikit-learn

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

# Dataset

The dataset is generated using the **Udacity Self-Driving Car Simulator**.

Only the **center camera images** are used for training.

Expected directory structure:

```text
dataset/
│
├── driving_log.csv
└── IMG/
    ├── center_000001.jpg
    ├── center_000002.jpg
    └── ...
```

The `driving_log.csv` file contains:

```text
center,left,right,steering,throttle,brake,speed
```

Although the simulator records images from three cameras, this project only uses:

* Center camera image
* Steering angle

---

# Image Preprocessing

Every image is preprocessed before being passed to the neural network.

Processing steps:

1. Crop the sky.
2. Crop the car hood.
3. Resize to **200 × 66** pixels.
4. Normalize pixel values inside the model.

The same preprocessing is used during both training and autonomous driving.

---

# Model Architecture

This project uses the **NVIDIA End-to-End Learning for Self-Driving Cars** architecture.

Network:

```text
Input (66 × 200 × 3)

↓

Normalization

↓

Conv 24

↓

Conv 36

↓

Conv 48

↓

Conv 64

↓

Conv 64

↓

Flatten

↓

Dense 100

↓

Dense 50

↓

Dense 10

↓

Output (Steering Angle)
```

---

# Training the Model

Make sure your dataset is located inside the `dataset/` folder.

Run:

```bash
python train.py
```

The training script will:

* Load the dataset
* Read center camera images
* Preprocess images
* Split training and validation data
* Train the NVIDIA CNN
* Save the trained model

Output:

```text
model.h5
```

---

# Running Autonomous Mode

After training:

1. Open the Udacity Self-Driving Car Simulator.
2. Select your preferred track.
3. Switch the simulator to **Autonomous Mode**.
4. Run:

```bash
python drive.py
```

The script will:

* Load `model.h5`
* Receive camera frames from the simulator
* Preprocess each frame
* Predict the steering angle
* Send steering and throttle commands to the simulator

The vehicle will begin driving automatically using the trained model.

---

# Complete Workflow

```text
Open Simulator
        │
        ▼
Training Mode
        │
        ▼
Collect Dataset
        │
        ▼
dataset/
        │
        ▼
python train.py
        │
        ▼
model.h5
        │
        ▼
Open Simulator
        │
        ▼
Autonomous Mode
        │
        ▼
python drive.py
        │
        ▼
Self-Driving Car
```

---

# Notes

* This project uses **only the center camera** for simplicity.
* Image preprocessing during training and inference must be identical.
* Better driving performance generally requires a larger and more diverse dataset.
* Collect recovery driving data to improve the model's ability to return to the lane after drifting.

---

# References

* NVIDIA – *End to End Learning for Self-Driving Cars*
* Udacity – Self-Driving Car Simulator
* TensorFlow & Keras Documentation

---

# License

This project is released for educational purposes. Feel free to use, modify, and improve it for learning and research.
