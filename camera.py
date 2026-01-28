import cv2
import numpy as np
import os
import json
import pandas as pd
from datetime import datetime
import time

DATABASE_DIR = "database"
METADATA_FILE = os.path.join(DATABASE_DIR, "metadata.json")
ATTENDANCE_FILE = "attendance.csv"
CONFIDENCE_THRESHOLD = 55

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

recognizer = None
names = {}
ages = {}
last_logged = {}

def load_metadata():
    if not os.path.exists(METADATA_FILE):
        return {}
    with open(METADATA_FILE, 'r') as f:
        return json.load(f)

def train_recognizer():
    global recognizer, names, ages
    metadata = load_metadata()

    faces, ids = [], []
    names, ages = {}, {}
    current_id = 0

    for file in os.listdir(DATABASE_DIR):
        if file.lower().endswith(('.jpg', '.png')):
            img = cv2.imread(os.path.join(DATABASE_DIR, file))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(gray, 1.1, 5)

            if len(detected):
                x, y, w, h = detected[0]
                faces.append(gray[y:y+h, x:x+w])
                ids.append(current_id)
                names[current_id] = metadata[file]["name"]
                ages[current_id] = metadata[file]["age"]
                current_id += 1

    if faces:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(ids))

def log_attendance(name, age):
    now = datetime.now()
    if name in last_logged and time.time() - last_logged[name] < 60:
        return

    last_logged[name] = time.time()
    pd.DataFrame([{
        "Name": name,
        "Age": age,
        "Date": now.strftime("%Y-%m-%d"),
        "Time": now.strftime("%H:%M:%S")
    }]).to_csv(ATTENDANCE_FILE, mode='a', header=not os.path.exists(ATTENDANCE_FILE), index=False)

def process_frame(frame_bytes):
    train_recognizer()

    npimg = np.frombuffer(frame_bytes, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    results = []

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        label = "Unknown"

        if recognizer:
            id_pred, conf = recognizer.predict(roi)
            if conf < CONFIDENCE_THRESHOLD:
                label = names[id_pred]
                log_attendance(label, ages[id_pred])

        results.append(label)

    return {"faces": results}
