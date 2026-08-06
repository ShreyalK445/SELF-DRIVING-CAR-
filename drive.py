import argparse
import base64
from io import BytesIO

import cv2
import eventlet
import eventlet.wsgi
import numpy as np
import socketio
from flask import Flask
from PIL import Image
from tensorflow.keras.models import load_model

# SocketIO server
sio = socketio.Server(cors_allowed_origins="*")
app = Flask(__name__)

model = None


def preprocess(img):
    # Remove alpha channel if present
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    # Crop
    img = img[60:135, :, :]

    # Resize
    img = cv2.resize(img, (200, 66))

    # RGB -> YUV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    # Normalize
    img = img.astype(np.float32) / 255.0

    return img


@sio.on("telemetry")
def telemetry(sid, data):

    if data is None:
        return

    # Decode image
    image = Image.open(BytesIO(base64.b64decode(data["image"])))
    image = np.asarray(image)

    # Preprocess
    image = preprocess(image)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Predict steering
    prediction = model.predict(image, verbose=0)

    print("Prediction:", prediction)
    print("Shape:", prediction.shape)

    steering_angle = float(np.squeeze(prediction))

    # Dynamic throttle
    if abs(steering_angle) > 0.45:
        throttle = 0.12
    elif abs(steering_angle) > 0.25:
        throttle = 0.18
    else:
        throttle = 0.30

    print(
        f"Steering: {steering_angle:.4f}  Throttle: {throttle:.2f}"
    )

    send_control(steering_angle, throttle)


@sio.on("connect")
def connect(sid, environ):
    print("Connected to simulator")
    send_control(0.0, 0.0)


def send_control(steering_angle, throttle):

    sio.emit(
        "steer",
        data={
            "steering_angle": str(steering_angle),
            "throttle": str(throttle)
        },
        skip_sid=True
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)

    args = parser.parse_args()

    print("Loading model...")

    model = load_model(args.model, compile=False)

    print("Model loaded successfully!")

    app = socketio.WSGIApp(sio, app)

    eventlet.wsgi.server(
        eventlet.listen(("", 4567)),
        app
    )
