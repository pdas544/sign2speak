/* ============================================================
   probe.js — handedness diagnostic probe

   Same capture flow as app.js, plus true-label logging and JSON export.
   Each logged row: {ts, model, true_gloss, used_hand, predicted_gloss,
   confidence, accepted, hand_l, hand_r, feature_dim}
   ============================================================ */

const MAX_FRAMES = window.S2S_CONFIG.maxSeqLength;
const CAPTURE_INTERVAL_MS = 200;

let stream = null;
let captureInterval = null;
let keypointBuffer = [];
let isRunning = false;
let usedHand = 'unknown';
let probeRows = [];

const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas-overlay');
const ctx = canvas.getContext('2d');
const statusBadge = document.getElementById('status-badge');
const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const errorBanner = document.getElementById('error-banner');
const glossSel = document.getElementById('true-gloss');
const probeList = document.getElementById('probe-list');
const probeCount = document.getElementById('probe-count');
const lastReading = document.getElementById('last-reading');

function setStatus(s) {
  statusBadge.className = s;
  statusBadge.textContent = { idle: 'Idle', capturing: 'Capturing…', processing: 'Processing…' }[s] || s;
}
function showError(m) { errorBanner.textContent = m; errorBanner.classList.add('visible'); }
function clearError() { errorBanner.classList.remove('visible'); errorBanner.textContent = ''; }

async function loadGlosses() {
  const res = await fetch('/inference/labels');
  const data = await res.json();
  glossSel.innerHTML = '';
  (data.labels || []).forEach(g => {
    const o = document.createElement('option');
    o.value = g; o.textContent = g;
    glossSel.appendChild(o);
  });
  const m = await (await fetch('/inference/models/active')).json();
  document.getElementById('probe-model').textContent =
    `${m.display_name || m.name} (${m.labels_count} signs)`;
}
loadGlosses();

async function startWebcam() {
  stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = stream;
  await new Promise(r => { video.onloadedmetadata = r; });
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}
function stopWebcam() {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  video.srcObject = null;
}

async function captureFrame() {
  if (!isRunning || !video.videoWidth) return;
  const c = document.createElement('canvas');
  c.width = video.videoWidth; c.height = video.videoHeight;
  c.getContext('2d').drawImage(video, 0, 0);
  try {
    const res = await fetch('/media/frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame: c.toDataURL('image/jpeg', 0.8) })
    });
    const data = await res.json();
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
      if (keypointBuffer.length > MAX_FRAMES * 3) keypointBuffer.shift();
    }
  } catch (e) { showError('Frame error: ' + e.message); }
}

function bestWindow(frames, n) {
  if (frames.length <= n) return frames.slice(-n);
  let best = frames.length - n, bestScore = -1;
  const diff = (a, b) => {
    let s = 0; const m = Math.min(a.length, b.length);
    for (let i = 0; i < m; i++) s += Math.abs(a[i] - b[i]);
    return s / m;
  };
  for (let s = 0; s <= frames.length - n; s++) {
    let score = 0;
    for (let i = s + 1; i < s + n; i++) score += diff(frames[i], frames[i - 1]);
    if (score > bestScore) { bestScore = score; best = s; }
  }
  return frames.slice(best, best + n);
}

async function stopAndLog() {
  if (keypointBuffer.length < MAX_FRAMES) {
    showError(`Need ${MAX_FRAMES} frames; got ${keypointBuffer.length}.`);
    return;
  }
  setStatus('processing');
  clearError();
  try {
    const res = await fetch('/inference/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ keypoints: bestWindow(keypointBuffer, MAX_FRAMES), generate_audio: false })
    });
    const data = await res.json();
    const row = {
      ts: new Date().toISOString(),
      model: data.model || null,
      true_gloss: glossSel.value,
      used_hand: usedHand,
      predicted_gloss: data.predicted_gloss || null,
      confidence: data.confidence ?? null,
      accepted: data.accepted ?? null,
      hand_l: (data.hands && data.hands.l) ?? null,
      hand_r: (data.hands && data.hands.r) ?? null,
      feature_dim: data.feature_dim ?? null,
      error: data.error || null
    };
    probeRows.unshift(row);
    renderRows();
    lastReading.textContent =
      `${row.true_gloss} (${row.used_hand}) → ${row.predicted_gloss} @ ${(row.confidence * 100 || 0).toFixed(1)}% [L=${row.hand_l} R=${row.hand_r}]`;
  } catch (e) { showError('Predict error: ' + e.message); }
  setStatus('idle');
}

function renderRows() {
  probeCount.textContent = probeRows.length;
  probeList.innerHTML = '';
  probeRows.forEach(r => {
    const li = document.createElement('li');
    const ok = r.true_gloss === r.predicted_gloss;
    li.innerHTML = `<span class="gloss">${r.true_gloss}(${r.used_hand}) → ${r.predicted_gloss || r.error}</span>` +
      `<span class="conf">L=${r.hand_l} R=${r.hand_r}</span>`;
    li.style.borderColor = ok ? 'var(--success)' : 'var(--danger)';
    probeList.appendChild(li);
  });
}

btnStart.addEventListener('click', async () => {
  clearError();
  try { await startWebcam(); }
  catch (e) { showError('Camera denied: ' + e.message); return; }
  keypointBuffer = [];
  isRunning = true;
  btnStart.disabled = true; btnStop.disabled = false;
  setStatus('capturing');
  captureInterval = setInterval(captureFrame, CAPTURE_INTERVAL_MS);
});

btnStop.addEventListener('click', async () => {
  isRunning = false;
  clearInterval(captureInterval);
  stopWebcam();
  btnStart.disabled = false; btnStop.disabled = true;
  await stopAndLog();
});

for (const [id, val] of [['btn-hand-r', 'right'], ['btn-hand-l', 'left'], ['btn-hand-both', 'both']]) {
  document.getElementById(id).addEventListener('click', () => {
    usedHand = val;
    showError('');
    errorBanner.textContent = `Hand marked: ${val}`;
    errorBanner.classList.add('visible');
    setTimeout(clearError, 1200);
  });
}

document.getElementById('btn-export').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(probeRows, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'probe_' + new Date().toISOString().replace(/[:.]/g, '-') + '.json';
  a.click();
});
document.getElementById('btn-clear').addEventListener('click', () => {
  probeRows = []; renderRows(); lastReading.textContent = 'No readings yet';
});
