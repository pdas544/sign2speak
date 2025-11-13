import os
import cv2
import torch
import numpy as np
from fastapi import FastAPI, BackgroundTasks, Response
from fastapi.responses import StreamingResponse, JSONResponse
from model_transformer import SignLanguageTransformer, Config, text_to_speech
import mediapipe as mp
import tempfile
from threading import Thread, Lock

app = FastAPI(title="Sign2Speak Real-time API")

# Load Config and Model
config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLanguageTransformer(config).to(device)
model_path = "outputs/models/model_transformer.pth"
if not os.path.exists(model_path):
    raise RuntimeError(f"Model not found at {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# MediaPipe Holistic for keypoint extraction
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Buffer for keypoint sequences
frame_buffer = []
keypoint_buffer = []
buffer_lock = Lock()
max_buffer_size = config.max_seq_length  


def extract_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    def extract_landmarks(landmarks, count, dims):
        if landmarks:
            return np.array([[lm.x, lm.y, lm.z] if dims == 3 else [lm.x, lm.y] for lm in landmarks.landmark]).flatten()
        else:
            return np.zeros(count * dims)

    pose = extract_landmarks(results.pose_landmarks, 33, 3)  # Using 3 dims: x,y,z,visibility
    face = extract_landmarks(results.face_landmarks, 468, 3)
    lh = extract_landmarks(results.left_hand_landmarks, 21, 3)
    rh = extract_landmarks(results.right_hand_landmarks, 21, 3)

    keypoints = np.concatenate([pose, face, lh, rh])
    height, width, _ = image.shape
    boxes = []

    # Get bounding boxes (in pixel coordinates) of hands/pose, scaled to frame size
    def get_bbox(landmarks):
        if not landmarks:
            return None
        xs = [lm.x for lm in landmarks.landmark]
        ys = [lm.y for lm in landmarks.landmark]
        xmin = int(min(xs) * width)
        xmax = int(max(xs) * width)
        ymin = int(min(ys) * height)
        ymax = int(max(ys) * height)
        # Add padding
        pad = 20
        return max(xmin - pad, 0), max(ymin - pad, 0), min(xmax + pad, width), min(ymax + pad, height)

    if results.left_hand_landmarks:
        box = get_bbox(results.left_hand_landmarks)
        if box:
            boxes.append(("Left Hand", box))
    if results.right_hand_landmarks:
        box = get_bbox(results.right_hand_landmarks)
        if box:
            boxes.append(("Right Hand", box))
    if results.pose_landmarks:
        box = get_bbox(results.pose_landmarks)
        if box:
            boxes.append(("Pose", box))

    return keypoints, boxes

def webcam_capture():
    global frame_buffer, keypoint_buffer

    cap = cv2.VideoCapture(0)  # Webcam capture
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        keypoints, boxes = extract_keypoints(frame)

        # Draw bounding boxes with labels
        for label, (xmin, ymin, xmax, ymax) in boxes:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        with buffer_lock:
            frame_buffer.append(frame.copy())
            keypoint_buffer.append(keypoints)
            # Keep buffer length fixed
            if len(keypoint_buffer) > max_buffer_size:
                frame_buffer = frame_buffer[-max_buffer_size:]
                keypoint_buffer = keypoint_buffer[-max_buffer_size:]

        # Small sleep to reduce CPU usage
        cv2.waitKey(1)

@app.on_event("startup")
def startup_event():
    # Start webcam thread for continuous capture
    thread = Thread(target=webcam_capture, daemon=True)
    thread.start()

def generate_video_stream():
    while True:
        with buffer_lock:
            if not frame_buffer:
                continue
            frame = frame_buffer[-1]
        # Encode frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
def video_feed():
    """
    Streaming endpoint showing live webcam video with bounding boxes around detected signs.
    """
    return StreamingResponse(generate_video_stream(),
                             media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/")
def root():
    return {"message": "Sign2Speak Real-time API running"}

@app.get("/predict")
def predict():
    """
    Use the current buffered keypoints sequence to perform inference.
    Returns predicted gloss and generates speech audio file.
    """
    with buffer_lock:
        if len(keypoint_buffer) < max_buffer_size:
            return JSONResponse({"error": f"Need {max_buffer_size} frames, got {len(keypoint_buffer)}"})
        input_seq = np.array(keypoint_buffer[-max_buffer_size:])

    try:
        tensor_input = torch.from_numpy(input_seq).unsqueeze(0).float().to(device)  # (1, seq_len, features)
        with torch.no_grad():
            outputs = model(tensor_input)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, dim=1)
            predicted_gloss = config.idx_to_gloss[pred.item()]
            confidence = conf.item()

        # Generate speech audio
        tts_output = text_to_speech(predicted_gloss)


        return {
            "predicted_gloss": predicted_gloss,
            "confidence": round(confidence*100,2),
            "audio_file": tts_output if tts_output and os.path.exists(tts_output) else None
        }
    except Exception as e:
        return JSONResponse({"error": f"Inference failed: {str(e)}"})

# Optional: Endpoint to stream audio or video could be added here

if __name__ == "__main__":
    import uvicorn
    print("Starting Sign2Speak API on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
