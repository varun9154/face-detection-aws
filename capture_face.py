import cv2
import os
import time

DATABASE_DIR = "database"

def main():
    if not os.path.exists(DATABASE_DIR):
        os.makedirs(DATABASE_DIR)

    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    print("========================================")
    print("      FACE CAPTURE TOOL")
    print("========================================")
    print("Instructions:")
    print("1. Look into the camera.")
    print("2. Press 's' to SAVE your photo.")
    print("3. Press 'q' to QUIT without saving.")
    print("========================================")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Display instructions on screen
        cv2.putText(frame, "Press 's' to Save Face", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Capture Face', frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            # Save the frame
            filename = input("Enter filename (e.g., 'me.jpg'): ").strip()
            if not filename:
                filename = f"user_{int(time.time())}.jpg"
            
            if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                filename += ".jpg"
                
            path = os.path.join(DATABASE_DIR, filename)
            cv2.imwrite(path, frame)
            print(f"Success! Photo saved to {path}")
            print("Now please update metadata.json with your details.")
            break
            
        elif key == ord('q'):
            print("Cancelled.")
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
