from flask import Flask, render_template, request, redirect, url_for, jsonify
from camera import process_frame
import os
import json

app = Flask(__name__)

DATABASE_DIR = "database"
METADATA_FILE = os.path.join(DATABASE_DIR, "metadata.json")

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        file = request.files['file']

        if not os.path.exists(DATABASE_DIR):
            os.makedirs(DATABASE_DIR)

        filename = file.filename
        path = os.path.join(DATABASE_DIR, filename)
        file.save(path)

        metadata = {}
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)

        metadata[filename] = {"name": name, "age": age}

        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=4)

        return redirect(url_for('index'))

    return render_template('add_user.html')


@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    file = request.files['image']
    frame_bytes = file.read()

    result = process_frame(frame_bytes)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
