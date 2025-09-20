Project: Face Verification API (Flask + DeepFace + MTCNN + ArcFace)
==================================================================

Description:
------------
REST API to verify if a selfie matches 4 other candidate images.
Uses:
- Flask for API
- MTCNN for face detection
- ArcFace (DeepFace) for face recognition
- PIL and NumPy for image processing

Main Features:
--------------
- Accepts a selfie + 4 candidate images
- Detects and crops faces automatically
- Compares faces using ArcFace embeddings
- Custom threshold (default 0.55) for match
- Deletes uploaded images after processing
- Returns JSON response with results and final decision

Files and Purpose:
------------------
1. app.py
   - Main application, defines routes "/", "/verify"
   - Handles uploads, detection, verification, and cleanup

2. requirements.txt
   - Python dependencies
   - Install with: pip install -r requirements.txt

3. README.md
   - Documentation, setup, configuration, usage, endpoints

4. uploads/ (folder)
   - Temporary storage for images, auto-deleted

Configuration:
--------------
- Allowed file types: png, jpg, jpeg, webp
- Max upload size: 10 MB
- Custom threshold: 0.55 (adjustable)

Virtual Environment Setup:
--------------------------
1. Create a virtual environment:
   python -m venv venv

2. Activate the virtual environment:
   - On Windows:
     venv\Scripts\activate
   - On macOS/Linux:
     source venv/bin/activate

3. Install dependencies inside the virtual environment:
   pip install -r requirements.txt

How to Run:
-----------
1. Activate virtual environment (see above)
2. Start the API:
   python app.py

API Endpoints:
--------------
- GET  /        : Health check
- POST /verify  : Verify selfie vs 4 candidate images

Expected POST fields:
---------------------
- selfie
- img1
- img2
- img3
- img4

Notes:
------
- All uploaded images are deleted after each request
- Use test_results.txt to record experiments
- Threshold can be tuned for accuracy
