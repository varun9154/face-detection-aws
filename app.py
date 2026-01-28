from flask import Flask, render_template, Response, request, redirect, url_for
from camera import VideoCamera
import os
import json
import time

app = Flask(__name__)

# Global camera instance
# We create it once so we don't open/close webcam constantly
camera = None

def get_camera():
    global camera
    if camera is None:
        camera = VideoCamera()
    return camera

@app.route('/')
def index():
    return render_template('dashboard.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(get_camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        file = request.files['file']
        
        if file and name:
            filename = file.filename
            # Save file
            path = os.path.join("database", filename)
            file.save(path)
            
            # Update metadata
            metadata_path = os.path.join("database", "metadata.json")
            data = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    try:
                        data = json.load(f)
                    except:
                        pass
            
            data[filename] = {"name": name, "age": age}
            
            with open(metadata_path, 'w') as f:
                json.dump(data, f, indent=4)
                
            # Re-initialize camera to retrain
            global camera
            del camera
            camera = None
            
            return redirect(url_for('index'))
            
    return render_template('add_user.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
