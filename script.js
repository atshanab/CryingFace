// Realistic Crying Face Filter (Emotion-Triggered)
/* eslint-disable no-undef */

const video = document.getElementById("cam");
const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d", { alpha: true });
const snapBtn = document.getElementById("snap");
const flipBtn = document.getElementById("flip");
const statusEl = document.getElementById("status");

let facingMode = "user";
let stream;
let tearsImg = new Image();
tearsImg.src = "assets/tears.png";

let sadScore = 0;
let blinkEMA = 0;
const ALPHA = 0.2;
let tearOffset = 0;

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    resizeCanvas();
    statusEl.textContent = "Camera ready";
  } catch (e) {
    console.error(e);
    statusEl.textContent = "Camera permission needed";
  }
}

function resizeCanvas() {
  canvas.width = video.videoWidth || 1280;
  canvas.height = video.videoHeight || 720;
}

window.addEventListener("resize", resizeCanvas);

const faceMesh = new FaceMesh({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${file}`,
});
faceMesh.setOptions({
  maxNumFaces: 1,
  refineLandmarks: true,
  minDetectionConfidence: 0.6,
  minTrackingConfidence: 0.6,
});

faceMesh.onResults(onResults);

let mpCamera;

async function startMP() {
  if (mpCamera) mpCamera.stop();
  mpCamera = new Camera(video, {
    onFrame: async () => {
      await faceMesh.send({ image: video });
    },
    width: 1280,
    height: 720,
  });
  mpCamera.start();
}

function landmarkXY(lm) {
  return [lm.x * canvas.width, lm.y * canvas.height];
}

function computeSadness(lms) {
  const L_INNER_BROW = 65, R_INNER_BROW = 295;
  const L_EYE_UP = 159, L_EYE_LOW = 145;
  const R_EYE_UP = 386, R_EYE_LOW = 374;
  const M_L = 61, M_R = 291, U_LIP = 13, L_LIP = 14;

  const lb = lms[L_INNER_BROW], rb = lms[R_INNER_BROW];
  const leu = lms[L_EYE_UP], lel = lms[L_EYE_LOW];
  const reu = lms[R_EYE_UP], rel = lms[R_EYE_LOW];
  const ml = lms[M_L], mr = lms[M_R], ul = lms[U_LIP];

  const ipd = Math.hypot(lms[33].x - lms[263].x, lms[33].y - lms[263].y) + 1e-6;

  const browEyeL = Math.abs(lb.y - leu.y) / ipd;
  const browEyeR = Math.abs(rb.y - reu.y) / ipd;
  const browLower = Math.max(0, 0.18 - (browEyeL + browEyeR) * 0.5) / 0.18;

  const mouthDown = Math.max(0, ((ml.y + mr.y) * 0.5 - ul.y) / ipd - 0.05) * 3.0;

  const earL = Math.abs(leu.y - lel.y) / ipd;
  const earR = Math.abs(reu.y - rel.y) / ipd;
  const ear = (earL + earR) * 0.5;
  const blink = Math.max(0, 0.06 - ear) / 0.06;

  let s = 0.6 * browLower + 0.4 * Math.min(1, mouthDown);
  sadScore = (1 - ALPHA) * sadScore + ALPHA * s;
  blinkEMA = (1 - ALPHA) * blinkEMA + ALPHA * blink;

  return { sad: sadScore, blink: blinkEMA };
}

function drawTearStreak(x0, y0, lengthPx, t) {
  const w = Math.max(6, lengthPx * 0.08);
  const steps = 16;
  ctx.save();
  ctx.globalAlpha = 0.7;
  ctx.lineWidth = w;
  ctx.lineCap = "round";
  ctx.strokeStyle = "rgba(240,245,255,0.55)";
  ctx.beginPath();
  for (let i = 0; i <= steps; i++) {
    const k = i / steps;
    const y = y0 + k * lengthPx + (Math.sin((k * 6 + t)) * 4);
    const x = x0 + Math.sin(k * 2 + t * 0.5) * 2;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.lineWidth = w * 0.4;
  ctx.strokeStyle = "rgba(255,255,255,0.6)";
  ctx.stroke();
  ctx.restore();
}

function onResults(results) {
  if (!results.image) return;
  if (canvas.width !== results.image.width) {
    canvas.width = results.image.width;
    canvas.height = results.image.height;
  }

  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(results.image, -canvas.width, 0, canvas.width, canvas.height);
  ctx.restore();

  const faces = results.multiFaceLandmarks || [];
  if (faces.length === 0) {
    statusEl.textContent = "No face detected";
    return;
  }
  statusEl.textContent = "Face detected";

  const lms = faces[0];
  const { sad, blink } = computeSadness(lms);

  const triggerOn = sad > 0.55;
  const maxLen = canvas.height * 0.22;
  tearOffset += 1 + sad * 2;

  if (triggerOn) {
    const L_LOWER = lms[145], R_LOWER = lms[374];
    const [lx, ly] = landmarkXY(L_LOWER);
    const [rx, ry] = landmarkXY(R_LOWER);

    const length = maxLen * Math.min(1, 0.4 + sad * 0.8 + blink * 0.5);
    const t = tearOffset * 0.05;

    drawTearStreak(lx, ly, length, t);
    drawTearStreak(rx, ry, length, t + 0.7);
  }
}

async function captureAndSave() {
  const blob = await new Promise((res) => canvas.toBlob(res, "image/png", 0.95));
  const file = new File([blob], "crying-face.png", { type: "image/png" });
  if (navigator.canShare && navigator.canShare({ files: [file] })) {
    try {
      await navigator.share({ files: [file], title: "Crying Face" });
      return;
    } catch (e) {}
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "crying-face.png";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

snapBtn.addEventListener("click", captureAndSave);
flipBtn.addEventListener("click", async () => {
  facingMode = (facingMode === "user") ? "environment" : "user";
  if (stream) { stream.getTracks().forEach(t => t.stop()); }
  await startCamera();
});

(async () => {
  await startCamera();
  await startMP();
})();
