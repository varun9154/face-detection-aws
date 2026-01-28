import cv2
import sys
import os
import json
import numpy as np

# Constants
DATABASE_DIR = "database"
METADATA_FILE = os.path.join(DATABASE_DIR, "metadata.json")
# Lower is stricter. 
# 85 was too loose (matching everyone). 
# Try 55. If still too loose, lower to 45.
CONFIDENCE_THRESHOLD = 55 

def load_metadata():
    if not os.path.exists(METADATA_FILE):
        return {}
    with open(METADATA_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print("Error: metadata.json is malformed.")
            return {}

def train_recognizer(face_cascade):
    print("Training Face Recognizer from database...")
    
    # Check if database directory exists
    if not os.path.exists(DATABASE_DIR):
        os.makedirs(DATABASE_DIR)
        print(f"Created {DATABASE_DIR}. Please add images.")
        return None, {}, {}

    metadata = load_metadata()
    
    # Prepare training data
    faces = []
    ids = []
    
    # Mappings
    id_to_name = {}
    id_to_age = {}
    
    current_id = 0
    trained_count = 0
    
    # Iterate through files in database
    for filename in os.listdir(DATABASE_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(DATABASE_DIR, filename)
            
            # Read image
            img = cv2.imread(path)
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect face in this training image
            # We assume one face per image for the database
            detected_faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(detected_faces) > 0:
                # Taking the first/largest face
                (x, y, w, h) = detected_faces[0]
                roi_gray = gray[y:y+h, x:x+w]
                
                faces.append(roi_gray)
                ids.append(current_id)
                
                # Get info from metadata associated with filename
                info = metadata.get(filename, {})
                name = info.get("name", "Unknown")
                age = info.get("age", "?")
                
                id_to_name[current_id] = name
                id_to_age[current_id] = age
                
                current_id += 1
                trained_count += 1
            else:
                print(f"Warning: No face detected in {filename}. Skipping.")
    
    if trained_count == 0:
        print("No faces found in database for training.")
        return None, {}, {}
        
    # Create LBPH Recognizer
    # Requires opencv-contrib-python
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(ids))
        print(f"Training complete. {trained_count} faces learned.")
        return recognizer, id_to_name, id_to_age
    except AttributeError:
        print("Error: cv2.face not available. Ensure 'opencv-contrib-python' is installed.")
        return None, {}, {}

def main():
    print("Initializing System...")
    
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print("Error: Failed to load Haar Cascade.")
        sys.exit(1)

    # Train Recognizer
    recognizer, names, ages = train_recognizer(face_cascade)
    
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
        
    print("Starting Video Stream. Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        detected_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in detected_faces:
            roi_gray = gray[y:y+h, x:x+w]
            
            # Default values (Unknown)
            label_text = "Unknown"
            age_text = ""
            color = (0, 0, 255) # Red for unknown
            conf_text = ""
            
            if recognizer:
                try:
                    # predict returns (id, confidence)
                    # confidence: 0 is perfect match.
                    id_pred, conf = recognizer.predict(roi_gray)
                    conf_text = f" {round(conf)}"
                    
                    if conf < CONFIDENCE_THRESHOLD:
                        name = names.get(id_pred, "Unknown")
                        age = ages.get(id_pred, "?")
                        label_text = name
                        age_text = f"Age: {age}"
                        color = (0, 255, 0) # Green for known
                    else:
                        # Confidence too high (bad match)
                        label_text = "Unknown"
                        color = (0, 0, 255)
                        
                except Exception as e:
                    print(f"Prediction error: {e}")
            
            # Draw Box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw Text
            # Name
            cv2.putText(frame, label_text + conf_text, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            # Age (if known)
            if age_text:
                cv2.putText(frame, age_text, (x, y+h+25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Face Recognition System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
