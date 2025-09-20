import os
import uuid
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from deepface import DeepFace
from PIL import Image
from mtcnn import MTCNN
import numpy as np

# ---- Config ----
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB per request
CUSTOM_THRESHOLD = 0.55  # can be adjusted after testing

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

detector = MTCNN()  # MTCNN for robust face detection

# ---- Helpers ----
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def save_file(file_obj):
    filename = secure_filename(file_obj.filename)
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else "jpg"
    uid = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], uid)
    
    img = Image.open(file_obj).convert("RGB")
    max_dim = 1200
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim))
    img.save(path, format="JPEG", quality=85)
    return path

def crop_face(img_path):
    """Detect and crop the largest face using MTCNN"""
    img = np.array(Image.open(img_path).convert("RGB"))
    detections = detector.detect_faces(img)
    if not detections:
        return None  # no face detected
    # select the largest face
    largest = max(detections, key=lambda d: d['box'][2] * d['box'][3])
    x, y, w, h = largest['box']
    x, y = max(0, x), max(0, y)
    cropped = img[y:y+h, x:x+w]
    return Image.fromarray(cropped).resize((112,112))

# ---- Routes ----
@app.route("/")
def home():
    return "Face API running. POST /verify with 5 images (selfie + 4)."

@app.route("/verify", methods=["POST"])
def verify():
    expected = ["selfie", "img1", "img2", "img3", "img4"]
    files = {}
    for key in expected:
        f = request.files.get(key)
        if not f:
            return jsonify({"error": f"Missing file field: {key}"}), 400
        if not allowed_file(f.filename):
            return jsonify({"error": f"Invalid file type for field: {key}"}), 400
        files[key] = f

    saved_paths = []
    try:
        for k, f in files.items():
            path = save_file(f)
            saved_paths.append(path)

        selfie_path = saved_paths[0]
        candidate_paths = saved_paths[1:]

        selfie_face = crop_face(selfie_path)
        if selfie_face is None:
            return jsonify({"error": "No face detected in selfie"}), 400

        results = []
        for i, candidate_path in enumerate(candidate_paths, start=1):
            candidate_face = crop_face(candidate_path)
            if candidate_face is None:
                results.append({
                    "image_field": f"img{i}",
                    "match": False,
                    "error": "No face detected in candidate image"
                })
                continue

            # save temporary cropped faces
            tmp_selfie_path = os.path.join(UPLOAD_FOLDER, f"tmp_selfie_{uuid.uuid4().hex}.jpg")
            tmp_candidate_path = os.path.join(UPLOAD_FOLDER, f"tmp_candidate_{uuid.uuid4().hex}.jpg")
            selfie_face.save(tmp_selfie_path)
            candidate_face.save(tmp_candidate_path)

            try:
                res = DeepFace.verify(
                    img1_path=tmp_selfie_path,
                    img2_path=tmp_candidate_path,
                    model_name="ArcFace",
                    enforce_detection=False
                )
                distance = res.get("distance")
                match = False if distance is None else distance <= CUSTOM_THRESHOLD

                results.append({
                    "image_field": f"img{i}",
                    "match": match,
                    "distance": distance,
                    "threshold": CUSTOM_THRESHOLD,
                    "model": res.get("model", None)
                })

            except Exception as e:
                results.append({
                    "image_field": f"img{i}",
                    "match": False,
                    "error": str(e)
                })
            finally:
                # cleanup temporary cropped images
                os.remove(tmp_selfie_path)
                os.remove(tmp_candidate_path)

        final_decision = any(r.get("match") for r in results)

        return jsonify({
            "results": results,
            "final_decision": final_decision
        })

    finally:
        for p in saved_paths:
            try:
                os.remove(p)
            except:
                pass

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
