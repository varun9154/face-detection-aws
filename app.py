from flask import Flask, render_template, request, jsonify
import os
import json
import uuid
import boto3
import traceback

app = Flask(__name__)

# Local storage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, "database")
METADATA_FILE = os.path.join(DATABASE_DIR, "metadata.json")

# AWS S3 config
S3_BUCKET = "face-detection-data-varun"
S3_FOLDER = "users/"

# Explicit region (IMPORTANT)
s3 = boto3.client(
    "s3",
    region_name="ap-south-1"
)


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/add_user", methods=["GET", "POST"])
def add_user():
    if request.method == "GET":
        return render_template("add_user.html")

    try:
        print("INFO: /add_user request received")

        name = request.form.get("name")
        age = request.form.get("age")
        file = request.files.get("file")

        if not name or not age or not file:
            return jsonify({
                "success": False,
                "message": "All fields are required"
            }), 400

        # Ensure database folder exists
        os.makedirs(DATABASE_DIR, exist_ok=True)

        # Safe unique filename
        ext = os.path.splitext(file.filename)[1].lower()
        filename = f"{uuid.uuid4()}{ext}"

        local_path = os.path.join(DATABASE_DIR, filename)
        file.save(local_path)

        print(f"INFO: File saved locally -> {local_path}")

        # Upload to S3
        s3.upload_file(
            local_path,
            S3_BUCKET,
            S3_FOLDER + filename
        )

        print("INFO: File uploaded to S3 successfully")

        # Load existing metadata
        metadata = {}
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "r") as f:
                metadata = json.load(f)

        metadata[filename] = {
            "name": name,
            "age": age,
            "s3_path": f"s3://{S3_BUCKET}/{S3_FOLDER}{filename}"
        }

        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=4)

        print("INFO: Metadata updated")

        return jsonify({
            "success": True,
            "message": "User registered successfully and image stored in S3"
        })

    except Exception as e:
        print("ERROR in /add_user")
        traceback.print_exc()

        return jsonify({
            "success": False,
            "message": "Internal server error"
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
