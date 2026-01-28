import cv2
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import time

# Constants
DATABASE_DIR = "database"
METADATA_FILE = os.path.join(DATABASE_DIR, "metadata.json")
ATTENDANCE_FILE = "attendance.csv"
CONFIDENCE_THRESHOLD = 55

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        
        # Load Haar Cascade
        self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        
        # Load Metadata
        self.metadata = self.load_metadata()
        
        # Train Recognizer
        self.recognizer, self.names, self.ages = self.train_recognizer()
        
        # Attendance Cooldown (to prevent spamming CSV)
        # Dictionary of name -> last_logged_timestamp
        self.last_logged = {}
        
        # Ensure CSV exists
        if not os.path.exists(ATTENDANCE_FILE):
            pd.DataFrame(columns=["Name", "Age", "Date", "Time"]).to_csv(ATTENDANCE_FILE, index=False)

    def __del__(self):
        self.video.release()
    
    def load_metadata(self):
        if not os.path.exists(METADATA_FILE):
            return {}
        with open(METADATA_FILE, 'r') as f:
            try:
                return json.load(f)
            except:
                return {}

    def train_recognizer(self):
        print("Training Recognizer...")
        if not os.path.exists(DATABASE_DIR):
            os.makedirs(DATABASE_DIR)
            return None, {}, {}

        faces = []
        ids = []
        id_to_name = {}
        id_to_age = {}
        current_id = 0
        trained_count = 0

        for filename in os.listdir(DATABASE_DIR):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(DATABASE_DIR, filename)
                img = cv2.imread(path)
                if img is None: continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                detected = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                
                if len(detected) > 0:
                    (x, y, w, h) = detected[0]
                    faces.append(gray[y:y+h, x:x+w])
                    ids.append(current_id)
                    
                    info = self.metadata.get(filename, {})
                    id_to_name[current_id] = info.get("name", "Unknown")
                    id_to_age[current_id] = info.get("age", "?")
                    
                    current_id += 1
                    trained_count += 1
        
        if trained_count == 0:
            return None, {}, {}
            
        try:
            rec = cv2.face.LBPHFaceRecognizer_create()
            rec.train(faces, np.array(ids))
            print(f"Training Complete. {trained_count} ids.")
            return rec, id_to_name, id_to_age
        except Exception as e:
            print(f"Recognizer Error: {e}")
            return None, {}, {}

    def log_attendance(self, name, age):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        current_date = now.strftime("%Y-%m-%d")
        
        # Check cooldown (60 seconds)
        last = self.last_logged.get(name)
        if last and (time.time() - last < 60):
            return # Too soon
            
        print(f"Logging attendance for {name}")
        self.last_logged[name] = time.time()
        
        new_entry = pd.DataFrame([{
            "Name": name, 
            "Age": age, 
            "Date": current_date, 
            "Time": current_time
        }])
        
        # Append to CSV
        new_entry.to_csv(ATTENDANCE_FILE, mode='a', header=False, index=False)

    def get_frame(self):
        success, image = self.video.read()
        if not success:
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            label = "Unknown"
            color = (0, 0, 255)
            
            if self.recognizer:
                try:
                    id_pred, conf = self.recognizer.predict(roi_gray)
                    if conf < CONFIDENCE_THRESHOLD:
                        name = self.names.get(id_pred, "Unknown")
                        age = self.ages.get(id_pred, "?")
                        # Display Name only (remove confidence number)
                        label = name
                        color = (0, 255, 0)
                        
                        # Log Attendance
                        if name != "Unknown":
                            self.log_attendance(name, age)
                    else:
                        label = "Unknown"
                except:
                    pass
            
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Draw Age below the box if available
            if label != "Unknown":
                age_text = f"Age: {age}"
                cv2.putText(image, age_text, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
