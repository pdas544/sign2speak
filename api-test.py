import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import torch
from model_transformer import SignLanguageTransformer, Config, text_to_speech
import cv2
import base64
from io import BytesIO

app = FastAPI(title="Sign2Speak Real-time API")

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLanguageTransformer(config).to(device)
model_path = "outputs/models/model_transformer.pth"
if not os.path.exists(model_path):
    raise RuntimeError(f"Model not found at {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# audio_dir = os.getcwd()
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

@app.get("/", response_class=HTMLResponse)
def root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sign2Speak Real-time Recognition</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; padding: 20px; }
            #video-container { display: inline-block; border: 2px solid #444; }
            button { padding: 10px 20px; font-size: 16px; margin: 5px; }
            #prediction { margin-top: 20px; font-weight: bold; font-size: 18px; }
            #audio-container { margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>Sign2Speak Real-time Recognition</h1>
        <div id="video-container">
            <video id="webcam" width="640" height="480" autoplay playsinline muted style="background:#000;display:block"></video>
            <canvas id="visualization" width="640" height="480" style="display:block;position:absolute;top:0;left:0;"></canvas>
        </div>
        <div id="buttons">
            <button id="start-btn">Start Recognition</button>
            <button id="stop-btn" disabled>Stop Recognition</button>
        </div>
        <div id="prediction">Prediction will appear here</div>
        <div id="audio-container"></div>
        <script>
            let recognizing = false;
            let webcamStream = null;
            let frameCaptureInterval = null;
            let keypointsBuffer = [];
            const maxFrames = 150;
            const video = document.getElementById('webcam');
            const canvas = document.getElementById('visualization');
            const ctx = canvas.getContext('2d');
            const startBtn = document.getElementById('start-btn');
            const stopBtn = document.getElementById('stop-btn');
            const predictionDiv = document.getElementById('prediction');
            const audioContainer = document.getElementById('audio-container');

            // Resizing/canvas handling
            function drawImageToCanvas(base64img) {
                let img = new window.Image();
                img.onload = function () {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                };
                img.src = base64img;
            }

            async function startWebcam() {
                try {
                    webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
                    video.srcObject = webcamStream;
                } catch (err) {
                    alert("Webcam error: " + err.message);
                }
            }

            function stopWebcam() {
                if (webcamStream && webcamStream.getTracks) {
                    webcamStream.getTracks().forEach(t => t.stop());
                }
                video.srcObject = null;
            }

            function captureFrameAndSend() {
                // Draw current video frame to temp canvas
                let tempCanvas = document.createElement('canvas');
                tempCanvas.width = video.videoWidth;
                tempCanvas.height = video.videoHeight;
                let tctx = tempCanvas.getContext('2d');
                tctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
                // Send jpeg frame to backend for processing & visualization box drawing
                tempCanvas.toBlob(blob => {
                    let reader = new FileReader();
                    reader.onloadend = function() {
                        let base64data = reader.result;
                        fetch('/frame', {
                            method: 'POST',
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({frame: base64data})
                        }).then(response => response.json())
                        .then(data => {
                            if(data.visualization){
                                drawImageToCanvas(data.visualization);
                            }
                            if(data.kp){
                                keypointsBuffer.push(data.kp);
                                if (keypointsBuffer.length > maxFrames) keypointsBuffer.shift();
                            }
                        });
                    }
                    reader.readAsDataURL(blob);
                }, 'image/jpeg', 0.8);
            }

            startBtn.onclick = () => {
                startBtn.disabled = true;
                stopBtn.disabled = false;
                predictionDiv.textContent = 'Capturing... Please sign.';
                audioContainer.innerHTML = '';
                keypointsBuffer = [];
                startWebcam().then(() => {
                    frameCaptureInterval = setInterval(() => {
                        captureFrameAndSend();
                    }, 200);
                });
            };

            stopBtn.onclick = () => {
                stopBtn.disabled = true;
                startBtn.disabled = false;
                predictionDiv.innerHTML = 'Sending for recognition...';
                clearInterval(frameCaptureInterval);
                stopWebcam();
                fetch('/predict', {
                    method: 'POST',
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({keypoints: keypointsBuffer})
                })
                .then(resp => resp.json())
                .then(data => {
                    if(data.error){
                        predictionDiv.textContent = 'Error: ' + data.error;
                        audioContainer.innerHTML = '';
                    } else {
                        predictionDiv.innerHTML = `Predicted Gloss: <b>${data.predicted_gloss}</b><br>Confidence: <b>${(data.confidence*100).toFixed(2)}%</b>`;
                        if(data.audio_file){
                            const audioElem = document.createElement('audio');
                            audioElem.controls = true;
                            audioElem.src = '/outputs/' + encodeURIComponent(data.audio_file);
                            audioContainer.innerHTML = '';
                            audioContainer.appendChild(audioElem);
                            audioElem.play();
                        } else {
                            audioContainer.innerHTML = 'No audio available.';
                        }
                    }
                })
                .catch(err => {
                    predictionDiv.textContent = 'Fetch error: ' + err;
                });
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/frame")
async def process_frame(request: Request):
    data = await request.json()
    frame_data = data.get('frame')
    if not frame_data:
        return JSONResponse({"error": "No frame received."})

    # Decode base64 JPEG to numpy
    content = frame_data.split(',')[1]
    raw = base64.b64decode(content)
    np_img = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    height, width, _ = img.shape

    with mp_holistic.Holistic(
        static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        results = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Extract keypoints:
        def extract_landmarks(landmarks, count, dims):
            if landmarks:
                return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
            else:
                return np.zeros(count * dims)
        pose = extract_landmarks(results.pose_landmarks, 33, 3)
        face = extract_landmarks(results.face_landmarks, 468, 3)
        lh = extract_landmarks(results.left_hand_landmarks, 21, 3)
        rh = extract_landmarks(results.right_hand_landmarks, 21, 3)
        keypoints = np.concatenate([pose, face, lh, rh]).tolist()

        # Draw boxes
        def get_bbox(landmarks):
            if not landmarks:
                return None
            xs = [lm.x for lm in landmarks.landmark]
            ys = [lm.y for lm in landmarks.landmark]
            xmin = int(min(xs) * width)
            xmax = int(max(xs) * width)
            ymin = int(min(ys) * height)
            ymax = int(max(ys) * height)
            pad = 20
            return max(xmin - pad, 0), max(ymin - pad, 0), min(xmax + pad, width), min(ymax + pad, height)

        boxes = []
        if results.left_hand_landmarks:
            box = get_bbox(results.left_hand_landmarks)
            if box: boxes.append(("Left Hand", box))

        if results.right_hand_landmarks:
            box = get_bbox(results.right_hand_landmarks)
            if box: boxes.append(("Right Hand", box))

        if results.pose_landmarks:
            box = get_bbox(results.pose_landmarks)
            if box: boxes.append(("Pose", box))

        for label, (xmin, ymin, xmax, ymax) in boxes:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img, label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Re-encode for browser
        _, buffer = cv2.imencode('.jpg', img)
        jpeg_base64 = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')

    return JSONResponse({"visualization": jpeg_base64, "kp": keypoints})

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    kp_list = data.get('keypoints', [])
    if not kp_list or len(kp_list) < config.max_seq_length:
        return JSONResponse({"error": f"Need at least {config.max_seq_length} frames, got {len(kp_list)}"})

    kp_np = np.array(kp_list[-config.max_seq_length:])  # (seq, features)
    try:
        tensor_input = torch.from_numpy(kp_np).unsqueeze(0).float().to(device)  # (1, seq_len, features)
        with torch.no_grad():
            outputs = model(tensor_input)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, dim=1)
            predicted_gloss = config.idx_to_gloss[pred.item()]
            confidence = conf.item()
        audio_path = text_to_speech(predicted_gloss)
        audio_file_name = os.path.basename(audio_path) if audio_path and os.path.exists(audio_path) else None
        return {
            "predicted_gloss": predicted_gloss,
            "confidence": confidence,
            "audio_file": audio_file_name
        }
    except Exception as e:
        return JSONResponse({"error": f"Inference failed: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    print("Starting Sign2Speak API on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
