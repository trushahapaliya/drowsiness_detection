/**
 * SafeDrive AI — Web-based Drowsiness Detection System
 * Uses MediaPipe FaceMesh to compute Eye Aspect Ratio (EAR) in real-time.
 */

// Global State
let cameraInstance = null;
let isCameraRunning = false;
let isFacingUser = true;
let score = 0;
let alarmThreshold = 15;
let earCutoff = 0.21;
let soundEnabled = true;
let alarmPlaying = false;
let lastFrameTime = performance.now();
let fpsCount = 0;
let audioContext = null;
let alarmOscillator = null;

// DOM Element References
const videoEl = document.getElementById('webcam');
const canvasEl = document.getElementById('output-canvas');
const ctx = canvasEl.getContext('2d');

const btnToggleCam = document.getElementById('btn-toggle-cam');
const btnSwitchCam = document.getElementById('btn-switch-cam');
const btnStartPrompt = document.getElementById('btn-start-prompt');
const btnToggleSound = document.getElementById('btn-toggle-sound');
const btnResetScore = document.getElementById('btn-reset-score');
const btnTestAlarm = document.getElementById('btn-test-alarm');

const startPrompt = document.getElementById('start-prompt');
const modelLoader = document.getElementById('model-loader');
const alarmBanner = document.getElementById('alarm-banner');

const statusBox = document.getElementById('status-box');
const statusEmoji = document.getElementById('status-emoji');
const statusTitle = document.getElementById('status-title');
const statusDesc = document.getElementById('status-desc');
const scoreText = document.getElementById('score-text');
const scoreBar = document.getElementById('score-bar');
const systemStatusBadge = document.getElementById('system-status-badge');

const valEarLeft = document.getElementById('val-ear');
const valEarRight = document.getElementById('val-ear-right');
const valFps = document.getElementById('val-fps');
const valFaces = document.getElementById('val-faces');

const inputThreshold = document.getElementById('input-threshold');
const thresholdValDisplay = document.getElementById('threshold-val');
const inputEarCutoff = document.getElementById('input-ear-cutoff');
const earCutoffValDisplay = document.getElementById('ear-cutoff-val');
const soundStatusText = document.getElementById('sound-status');
const soundIconText = document.getElementById('sound-icon');

// ── Web Audio API Siren Generator ──────────────────────────────────────────
function initAudio() {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioContext.state === 'suspended') {
    audioContext.resume();
  }
}

function startAlarmSound() {
  if (!soundEnabled || alarmPlaying) return;
  initAudio();

  try {
    alarmOscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();

    alarmOscillator.type = 'sawtooth';
    alarmOscillator.frequency.setValueAtTime(800, audioContext.currentTime);

    // Modulate pitch like a siren (800Hz to 1200Hz)
    const now = audioContext.currentTime;
    alarmOscillator.frequency.linearRampToValueAtTime(1200, now + 0.25);
    alarmOscillator.frequency.linearRampToValueAtTime(800, now + 0.5);

    // Loop siren sweep pattern
    setInterval(() => {
      if (alarmOscillator && alarmPlaying) {
        const t = audioContext.currentTime;
        alarmOscillator.frequency.linearRampToValueAtTime(1200, t + 0.25);
        alarmOscillator.frequency.linearRampToValueAtTime(800, t + 0.5);
      }
    }, 500);

    gainNode.gain.setValueAtTime(0.5, audioContext.currentTime);

    alarmOscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    alarmOscillator.start();
    alarmPlaying = true;
  } catch (err) {
    console.error("Audio error:", err);
  }
}

function stopAlarmSound() {
  if (alarmOscillator) {
    try {
      alarmOscillator.stop();
      alarmOscillator.disconnect();
    } catch (e) {}
    alarmOscillator = null;
  }
  alarmPlaying = false;
}

// ── Euclidean Distance & EAR Calculation ──────────────────────────────────
function dist(p1, p2) {
  return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
}

function calculateEAR(landmarks, indices) {
  const p1 = landmarks[indices[0]];
  const p2 = landmarks[indices[1]];
  const p3 = landmarks[indices[2]];
  const p4 = landmarks[indices[3]];
  const p5 = landmarks[indices[4]];
  const p6 = landmarks[indices[5]];

  const v1 = dist(p2, p6);
  const v2 = dist(p3, p5);
  const h = dist(p1, p4);

  if (h === 0) return 0;
  return (v1 + v2) / (2.0 * h);
}

// MediaPipe Landmark Indices for Eye Aspect Ratio
const LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380];

// ── MediaPipe Face Mesh Setup ─────────────────────────────────────────────
const faceMesh = new FaceMesh({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
});

faceMesh.setOptions({
  maxNumFaces: 1,
  refineLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

faceMesh.onResults(onResults);

function onResults(results) {
  modelLoader.classList.add('hidden');

  // Adjust canvas size to match video aspect ratio
  if (canvasEl.width !== videoEl.videoWidth || canvasEl.height !== videoEl.videoHeight) {
    canvasEl.width = videoEl.videoWidth || 640;
    canvasEl.height = videoEl.videoHeight || 480;
  }

  // FPS calculation
  const now = performance.now();
  const delta = (now - lastFrameTime) / 1000;
  lastFrameTime = now;
  fpsCount = Math.round(1 / delta) || 0;
  valFps.textContent = fpsCount;

  // Clear canvas & draw video frame
  ctx.save();
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  ctx.drawImage(results.image, 0, 0, canvasEl.width, canvasEl.height);

  let facesCount = results.multiFaceLandmarks ? results.multiFaceLandmarks.length : 0;
  valFaces.textContent = facesCount;

  if (facesCount > 0) {
    const landmarks = results.multiFaceLandmarks[0];

    // Compute EAR for left & right eyes
    const earLeft = calculateEAR(landmarks, LEFT_EYE_INDICES);
    const earRight = calculateEAR(landmarks, RIGHT_EYE_INDICES);
    const avgEAR = (earLeft + earRight) / 2.0;

    valEarLeft.textContent = earLeft.toFixed(2);
    valEarRight.textContent = earRight.toFixed(2);

    // Draw eye landmark points & contours
    drawEyeContour(landmarks, LEFT_EYE_INDICES, earLeft < earCutoff);
    drawEyeContour(landmarks, RIGHT_EYE_INDICES, earRight < earCutoff);

    // Check if eyes are closed
    const eyesClosed = avgEAR < earCutoff;

    if (eyesClosed) {
      score += 1;
    } else {
      score = Math.max(0, score - 1);
    }

    updateUI(eyesClosed);
  } else {
    valEarLeft.textContent = '0.00';
    valEarRight.textContent = '0.00';
    updateUI(false, true); // No face detected
  }

  ctx.restore();
}

function drawEyeContour(landmarks, indices, isClosed) {
  const w = canvasEl.width;
  const h = canvasEl.height;

  ctx.beginPath();
  const firstP = landmarks[indices[0]];
  ctx.moveTo(firstP.x * w, firstP.y * h);

  for (let i = 1; i < indices.length; i++) {
    const p = landmarks[indices[i]];
    ctx.lineTo(p.x * w, p.y * h);
  }
  ctx.closePath();

  ctx.strokeStyle = isClosed ? '#ef4444' : '#10b981';
  ctx.lineWidth = 2.5;
  ctx.stroke();
  ctx.fillStyle = isClosed ? 'rgba(239, 68, 68, 0.25)' : 'rgba(16, 185, 129, 0.15)';
  ctx.fill();
}

// ── UI Updates & Alarm Triggers ───────────────────────────────────────────
function updateUI(eyesClosed, noFace = false) {
  const threshold = parseInt(inputThreshold.value, 10);
  scoreText.textContent = `${score} / ${threshold}`;

  const pct = Math.min(100, Math.round((score / threshold) * 100));
  scoreBar.style.width = `${pct}%`;

  if (noFace) {
    statusBox.className = 'status-indicator-box status-init';
    statusEmoji.textContent = '👤';
    statusTitle.textContent = 'NO FACE DETECTED';
    statusDesc.textContent = 'Position your face clearly in front of the camera.';
    stopAlarmSound();
    alarmBanner.classList.add('hidden');
    return;
  }

  if (score >= threshold) {
    // DROWSINESS ALERT!
    statusBox.className = 'status-indicator-box status-drowsy';
    statusEmoji.textContent = '⚠️';
    statusTitle.textContent = 'DROWSY DRIVER ALERT!';
    statusDesc.textContent = 'Prolonged eye closure detected! Wake up!';

    alarmBanner.classList.remove('hidden');
    startAlarmSound();
  } else if (eyesClosed) {
    statusBox.className = 'status-indicator-box status-init';
    statusEmoji.textContent = '😑';
    statusTitle.textContent = 'EYES CLOSED';
    statusDesc.textContent = 'Warning: Eyes currently closed.';
    stopAlarmSound();
    alarmBanner.classList.add('hidden');
  } else {
    statusBox.className = 'status-indicator-box status-awake';
    statusEmoji.textContent = '😃';
    statusTitle.textContent = 'AWAKE & ALERT';
    statusDesc.textContent = 'Driver eyes open. System monitoring normal.';
    stopAlarmSound();
    alarmBanner.classList.add('hidden');
  }
}

// ── Camera Control ────────────────────────────────────────────────────────
async function startCamera() {
  initAudio();
  startPrompt.classList.add('hidden');
  modelLoader.classList.remove('hidden');

  try {
    if (cameraInstance) {
      await cameraInstance.stop();
    }

    cameraInstance = new Camera(videoEl, {
      onFrame: async () => {
        await faceMesh.send({ image: videoEl });
      },
      width: 640,
      height: 480,
      facingMode: isFacingUser ? 'user' : 'environment'
    });

    await cameraInstance.start();
    isCameraRunning = true;
    btnToggleCam.innerHTML = '<span class="btn-icon">⏹️</span> Stop Camera';
    btnToggleCam.className = 'btn btn-secondary';
    btnSwitchCam.disabled = false;
    systemStatusBadge.textContent = 'Monitoring Active';
    systemStatusBadge.className = 'badge badge-status';
  } catch (err) {
    console.error("Camera access failed:", err);
    modelLoader.classList.add('hidden');
    startPrompt.classList.remove('hidden');
    alert("Camera permission denied or camera unavailable. Please allow webcam access.");
  }
}

function stopCamera() {
  if (cameraInstance) {
    cameraInstance.stop();
    cameraInstance = null;
  }
  isCameraRunning = false;
  btnToggleCam.innerHTML = '<span class="btn-icon">📷</span> Start Camera';
  btnToggleCam.className = 'btn btn-primary';
  btnSwitchCam.disabled = true;
  startPrompt.classList.remove('hidden');
  stopAlarmSound();
  alarmBanner.classList.add('hidden');
  systemStatusBadge.textContent = 'System Paused';
}

// ── Event Listeners ───────────────────────────────────────────────────────
btnToggleCam.addEventListener('click', () => {
  if (isCameraRunning) stopCamera();
  else startCamera();
});

btnStartPrompt.addEventListener('click', startCamera);

btnSwitchCam.addEventListener('click', () => {
  isFacingUser = !isFacingUser;
  startCamera();
});

inputThreshold.addEventListener('input', (e) => {
  alarmThreshold = parseInt(e.target.value, 10);
  thresholdValDisplay.textContent = `${alarmThreshold} frames (~${(alarmThreshold * 0.033).toFixed(1)}s)`;
});

inputEarCutoff.addEventListener('input', (e) => {
  earCutoff = parseFloat(e.target.value);
  earCutoffValDisplay.textContent = earCutoff.toFixed(2);
});

btnToggleSound.addEventListener('click', () => {
  soundEnabled = !soundEnabled;
  soundStatusText.textContent = soundEnabled ? 'ON' : 'OFF';
  soundIconText.textContent = soundEnabled ? '🔊' : '🔇';
  if (!soundEnabled) stopAlarmSound();
});

btnResetScore.addEventListener('click', () => {
  score = 0;
  updateUI(false);
});

btnTestAlarm.addEventListener('click', () => {
  initAudio();
  if (alarmPlaying) {
    stopAlarmSound();
    alarmBanner.classList.add('hidden');
  } else {
    alarmBanner.classList.remove('hidden');
    startAlarmSound();
    setTimeout(() => {
      stopAlarmSound();
      alarmBanner.classList.add('hidden');
    }, 2500);
  }
});
