/* ============================================================
   app.js — Sign2Speak main recognition page logic

   Requires window.S2S_CONFIG to be defined in the template:
     window.S2S_CONFIG = { maxSeqLength: <int> };
   ============================================================ */

const MAX_FRAMES        = window.S2S_CONFIG.maxSeqLength;
const CAPTURE_BUFFER_MULTIPLIER = 3;
const CAPTURE_INTERVAL_MS = 200;
const MAX_HISTORY       = 10;

let stream          = null;
let captureInterval = null;
let keypointBuffer  = [];
let isRunning       = false;
let historyItems    = [];

const video       = document.getElementById('webcam');
const canvas      = document.getElementById('canvas-overlay');
const ctx         = canvas.getContext('2d');
const statusBadge = document.getElementById('status-badge');
const btnStart    = document.getElementById('btn-start');
const btnStop     = document.getElementById('btn-stop');
const errorBanner = document.getElementById('error-banner');
const glossEl     = document.getElementById('prediction-gloss');
const subEl       = document.getElementById('prediction-sub');
const confBar     = document.getElementById('confidence-bar');
const confValue   = document.getElementById('confidence-value');
const audioEnWrap = document.getElementById('audio-en-wrap');
const audioHiWrap = document.getElementById('audio-hi-wrap');
const historyList = document.getElementById('history-list');
const historyEmpty= document.getElementById('history-empty');

// ---- Health check ----
async function checkHealth() {  try {
    const res = await fetch('/health');
    const dot = document.getElementById('health-dot');
    const lbl = document.getElementById('health-label');
    if (res.ok) {
      dot.className = 'ok';
      lbl.textContent = 'Service online';
    } else {
      dot.className = 'error';
      lbl.textContent = 'Service degraded';
    }
  } catch {
    document.getElementById('health-dot').className = 'error';
    document.getElementById('health-label').textContent = 'Unreachable';
  }
}
checkHealth();
setInterval(checkHealth, 30000);

// ---- Model selector + supported-gloss chips ----
async function loadModels() {
  const sel = document.getElementById('model-select');
  const hint = document.getElementById('model-hint');
  try {
    const res = await fetch('/inference/models');
    if (!res.ok) throw new Error('models unavailable');
    const data = await res.json();
    sel.innerHTML = '';
    (data.models || []).forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.textContent = `${m.display_name || m.name} (${(m.labels || []).length} signs)`;
      if (m.is_active) opt.selected = true;
      sel.appendChild(opt);
    });
    const active = (data.models || []).find(m => m.is_active);
    if (active) hint.textContent = `${(active.labels || []).length} signs · ${active.framework || ''}`;
    sel.onchange = switchModel;
  } catch (e) {
    hint.textContent = 'Model list unavailable';
  }
}

async function switchModel() {
  const sel = document.getElementById('model-select');
  const hint = document.getElementById('model-hint');
  try {
    const res = await fetch('/inference/models/active', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_name: sel.value })
    });
    if (!res.ok) throw new Error('switch failed');
    await loadLabels();
    const active = sel.options[sel.selectedIndex].text;
    hint.textContent = active;
  } catch (e) {
    showError('Model switch failed: ' + e.message);
  }
}

async function loadLabels() {
  const wrap = document.getElementById('gloss-chips');
  const count = document.getElementById('gloss-count');
  try {
    const res = await fetch('/inference/labels');
    if (!res.ok) throw new Error('labels unavailable');
    const data = await res.json();
    wrap.innerHTML = '';
    (data.labels || []).forEach(g => {
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.textContent = g;
      wrap.appendChild(chip);
    });
    count.textContent = data.count;
  } catch {
    wrap.innerHTML = '<p class="audio-placeholder">Labels unavailable</p>';
  }
}
loadModels();
loadLabels();

// ---- UI helpers ----
function setStatus(state) {
  statusBadge.className = state;
  const labels = { idle: 'Idle', capturing: 'Capturing…', processing: 'Processing…', result: 'Result' };
  statusBadge.textContent = labels[state] || state;
}

function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.classList.add('visible');
}

function clearError() {
  errorBanner.classList.remove('visible');
  errorBanner.textContent = '';
}

function setConfidence(pct) {
  const clamped = Math.max(0, Math.min(100, pct));
  confBar.style.width = clamped + '%';
  confBar.style.background = clamped >= 70
    ? 'var(--success)'
    : clamped >= 50
      ? 'var(--warn)'
      : 'var(--danger)';
  confValue.textContent = clamped.toFixed(1) + '%';
}

function clearAudio(wrap) {
  wrap.innerHTML = '<p class="audio-placeholder">No audio yet</p>';
}

function setAudio(wrap, fileUrl) {
  const audio = document.createElement('audio');
  audio.controls = true;
  audio.src = fileUrl;
  wrap.innerHTML = '';
  wrap.appendChild(audio);
  audio.play().catch(() => {});
}

function addToHistory(gloss, confidencePct) {
  historyItems.unshift({ gloss, confidencePct });
  if (historyItems.length > MAX_HISTORY) historyItems.pop();

  historyList.innerHTML = '';
  historyItems.forEach(item => {
    const li = document.createElement('li');
    li.innerHTML = `<span class="gloss">${item.gloss}</span><span class="conf">${item.confidencePct.toFixed(1)}%</span>`;
    historyList.appendChild(li);
  });

  historyEmpty.style.display = historyItems.length ? 'none' : '';
}

function meanAbsDiff(a, b) {
  const n = Math.min(a.length, b.length);
  if (!n) return 0;
  let sum = 0;
  for (let i = 0; i < n; i += 1) {
    sum += Math.abs(a[i] - b[i]);
  }
  return sum / n;
}

function selectBestMotionWindow(frames, windowSize) {
  if (frames.length <= windowSize) return frames.slice(-windowSize);

  let bestStart = frames.length - windowSize;
  let bestScore = -1;

  for (let start = 0; start <= frames.length - windowSize; start += 1) {
    let score = 0;
    for (let i = start + 1; i < start + windowSize; i += 1) {
      score += meanAbsDiff(frames[i], frames[i - 1]);
    }
    if (score > bestScore) {
      bestScore = score;
      bestStart = start;
    }
  }

  return frames.slice(bestStart, bestStart + windowSize);
}

function resetPredictionDisplay(subText = 'Perform a sign to begin') {
  glossEl.textContent = '-';
  subEl.textContent = subText;
  setConfidence(0);
}

// ---- Webcam ----
async function startWebcam() {
  stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = stream;
  await new Promise(resolve => { video.onloadedmetadata = resolve; });
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
}

function stopWebcam() {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  video.srcObject = null;
}

// ---- Frame capture → /media/frame ----
async function captureAndSendFrame() {
  if (!isRunning) return;
  if (!video.videoWidth || !video.videoHeight) return;

  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width  = video.videoWidth;
  tmpCanvas.height = video.videoHeight;
  tmpCanvas.getContext('2d').drawImage(video, 0, 0);

  const dataUrl = tmpCanvas.toDataURL('image/jpeg', 0.8);

  try {
    const res = await fetch('/media/frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame: dataUrl })
    });

    let data = {};
    try {
      data = await res.json();
    } catch {
      data = {};
    }

    if (!res.ok) {
      const errorMsg = data.error || `Frame processing failed (${res.status})`;
      showError(errorMsg);
      return;
    }

    if (data.error) { showError(data.error); return; }

    if (data.visualization) {
      const img = new Image();
      img.onload = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);
      };
      img.src = data.visualization;
    }

    if (data.keypoints) {
      keypointBuffer.push(data.keypoints);
      const maxBufferedFrames = MAX_FRAMES * CAPTURE_BUFFER_MULTIPLIER;
      if (keypointBuffer.length > maxBufferedFrames) keypointBuffer.shift();
    }
  } catch (e) {
    showError('Frame processing error: ' + e.message);
  }
}

// ---- Predict → /inference/predict ----
async function runPrediction() {
  if (keypointBuffer.length < MAX_FRAMES) {
    showError(`Need ${MAX_FRAMES} frames; captured ${keypointBuffer.length}. Keep signing!`);
    return;
  }

  const predictionFrames = selectBestMotionWindow(keypointBuffer, MAX_FRAMES);

  setStatus('processing');
  clearError();

  try {
    const res = await fetch('/inference/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ keypoints: predictionFrames, generate_audio: true })
    });

    let data = {};
    try {
      data = await res.json();
    } catch {
      data = {};
    }

    if (!res.ok) {
      let errorMsg = data.error || `Prediction failed (${res.status})`;
      if (data.quality && typeof data.quality.temporal_delta === 'number') {
        errorMsg += ` (motion=${data.quality.temporal_delta}, required=${data.quality.min_temporal_delta})`;
      }
      showError(errorMsg);
      resetPredictionDisplay('No new prediction');
      setStatus('idle');
      return;
    }

    if (data.error) {
      showError(data.error);
      resetPredictionDisplay('No new prediction');
      setStatus('idle');
      return;
    }

    glossEl.textContent = data.predicted_gloss;
    subEl.textContent = data.accepted ? 'Accepted prediction' : 'Low confidence — try again';
    setConfidence(data.confidence_percent);

    if (data.audio && data.audio.en) {
      setAudio(audioEnWrap, '/inference/audio/' + data.audio.en);
    } else { clearAudio(audioEnWrap); }

    if (data.audio && data.audio.hi) {
      setAudio(audioHiWrap, '/inference/audio/' + data.audio.hi);
    } else { clearAudio(audioHiWrap); }

    addToHistory(data.predicted_gloss, data.confidence_percent);
    setStatus('result');

  } catch (e) {
    showError('Prediction error: ' + e.message);
    resetPredictionDisplay('No new prediction');
    setStatus('idle');
  }
}

// ---- Button handlers ----
btnStart.addEventListener('click', async () => {
  clearError();
  resetPredictionDisplay('Capturing...');
  try {
    await startWebcam();
  } catch (e) {
    showError('Camera access denied: ' + e.message);
    return;
  }

  keypointBuffer = [];
  isRunning = true;
  btnStart.disabled = true;
  btnStop.disabled  = false;
  setStatus('capturing');

  captureInterval = setInterval(captureAndSendFrame, CAPTURE_INTERVAL_MS);
});

btnStop.addEventListener('click', async () => {
  isRunning = false;
  clearInterval(captureInterval);
  stopWebcam();
  btnStart.disabled = false;
  btnStop.disabled  = true;
  await runPrediction();
});
