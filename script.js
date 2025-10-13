// script.js — Realistic Sad Morph + Emotion Trigger + Tears (optional)
/* global FaceMesh, Camera */
const video = document.getElementById("cam");
const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d");
const snapBtn = document.getElementById("snap");
const statusEl = document.getElementById("status");
const flipBtn = document.getElementById("flip");

let facingMode = "user";
let stream;

// ===== Camera =====
async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false
  });
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth || 1280;
  canvas.height = video.videoHeight || 720;
  statusEl.textContent = "Camera ready";
}

function resizeCanvas() {
  if (!video.videoWidth) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}
window.addEventListener("resize", resizeCanvas);

// ===== FaceMesh =====
const faceMesh = new FaceMesh({
  locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${f}`
});
faceMesh.setOptions({
  maxNumFaces: 1,
  refineLandmarks: true,
  minDetectionConfidence: 0.6,
  minTrackingConfidence: 0.6
});
faceMesh.onResults(onResults);

let mpCamera;
async function startMP() {
  if (mpCamera) mpCamera.stop();
  mpCamera = new Camera(video, {
    width: 1280, height: 720,
    onFrame: async () => { await faceMesh.send({ image: video }); }
  });
  mpCamera.start();
}

// ===== Emotion heuristics =====
let sadEMA = 0, blinkEMA = 0;
const ALPHA = 0.18;

function computeSadness(lm) {
  // Indices
  const L_INNER_BROW = 65, R_INNER_BROW = 295;
  const L_EYE_UP = 159, L_EYE_LOW = 145;
  const R_EYE_UP = 386, R_EYE_LOW = 374;
  const M_L = 61, M_R = 291, U_LIP = 13;

  const ipd = Math.hypot(lm[33].x - lm[263].x, lm[33].y - lm[263].y) + 1e-6;

  // Brow lowered
  const browEyeL = Math.abs(lm[L_INNER_BROW].y - lm[L_EYE_UP].y) / ipd;
  const browEyeR = Math.abs(lm[R_INNER_BROW].y - lm[R_EYE_UP].y) / ipd;
  const browLower = Math.max(0, 0.18 - (browEyeL + browEyeR) * 0.5) / 0.18;

  // Mouth downturn
  const mouthDown = Math.max(0, ((lm[M_L].y + lm[M_R].y) * 0.5 - lm[U_LIP].y) / ipd - 0.05) * 3.0;

  // Blink
  const earL = Math.abs(lm[L_EYE_UP].y - lm[L_EYE_LOW].y) / ipd;
  const earR = Math.abs(lm[R_EYE_UP].y - lm[R_EYE_LOW].y) / ipd;
  const blink = Math.max(0, 0.06 - (earL + earR) * 0.5) / 0.06;

  const s = 0.6 * browLower + 0.4 * Math.min(1, mouthDown);
  sadEMA   = (1 - ALPHA) * sadEMA + ALPHA * s;
  blinkEMA = (1 - ALPHA) * blinkEMA + ALPHA * blink;
  return { sad: sadEMA, blink: blinkEMA, ipd };
}

// ===== Morph engine (grid + piecewise-affine warp) =====

// Build a coarse grid over the face box so we only warp the face region.
function faceBBox(lm) {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const p of lm) {
    minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
  }
  // pad a bit
  const pad = 0.05;
  minX = Math.max(0, minX - pad); minY = Math.max(0, minY - pad);
  maxX = Math.min(1, maxX + pad); maxY = Math.min(1, maxY + pad);
  return [
    minX * canvas.width, minY * canvas.height,
    (maxX - minX) * canvas.width, (maxY - minY) * canvas.height
  ];
}

// Controls that will be pushed to a “sad” target
const CTRL = {
  L_INNER_BROW: 65, L_MID_BROW: 70,
  R_INNER_BROW: 295, R_MID_BROW: 300,
  M_L: 61, M_R: 291, U_LIP: 13, L_LIP: 14,
  L_EYE_UP: 159, R_EYE_UP: 386
};

// Move control points to sad targets (returned in pixels)
function sadTargets(lmPx, ipdPx, strength) {
  const t = new Map();

  // Lower inner brows, slight inward pinch
  const browDrop = 0.12 * ipdPx * strength;
  const pinch    = 0.03 * ipdPx * strength;

  function add(id, dx, dy) { t.set(id, { x: lmPx[id].x + dx, y: lmPx[id].y + dy }); }

  add(CTRL.L_INNER_BROW, -pinch,  browDrop);
  add(CTRL.R_INNER_BROW,  pinch,  browDrop);
  add(CTRL.L_MID_BROW,    -pinch*0.6, browDrop*0.6);
  add(CTRL.R_MID_BROW,     pinch*0.6, browDrop*0.6);

  // Mouth corners down, center slightly up to make frown curve
  const mouthDown = 0.18 * ipdPx * strength;
  add(CTRL.M_L,  0,  mouthDown);
  add(CTRL.M_R,  0,  mouthDown);
  add(CTRL.U_LIP, 0, -mouthDown*0.25);

  // Upper eyelids a bit lower
  const lid = 0.06 * ipdPx * strength;
  add(CTRL.L_EYE_UP, 0, lid);
  add(CTRL.R_EYE_UP, 0, lid);

  return t;
}

// RBF displacement from control deltas (Gaussian weights)
function displacementField(lmPx, targets, sigma) {
  const ctrls = [...targets.keys()];
  const deltas = ctrls.map(id => {
    const s = lmPx[id], d = targets.get(id);
    return { id, dx: d.x - s.x, dy: d.y - s.y, sx: s.x, sy: s.y };
  });
  const twoSigma2 = 2 * sigma * sigma;

  return function disp(x, y) {
    let wx = 0, wy = 0, wsum = 0;
    for (const c of deltas) {
      const dx = x - c.sx, dy = y - c.sy;
      const w = Math.exp(-(dx*dx + dy*dy) / twoSigma2);
      wx += w * c.dx; wy += w * c.dy; wsum += w;
    }
    if (wsum < 1e-6) return { x, y };
    return { x: x + wx/wsum, y: y + wy/wsum };
  };
}

// Map one triangle (src → dst) using Canvas 2D affine transform
function mapTriangle(img, s0, s1, s2, d0, d1, d2) {
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(d0.x, d0.y); ctx.lineTo(d1.x, d1.y); ctx.lineTo(d2.x, d2.y); ctx.closePath();
  ctx.clip();

  // Solve affine:  d = M*s + t
  const den = (s1.x - s0.x)*(s2.y - s0.y) - (s2.x - s0.x)*(s1.y - s0.y);
  if (Math.abs(den) < 1e-5) { ctx.restore(); return; }
  const m11 = ((d1.x - d0.x)*(s2.y - s0.y) - (d2.x - d0.x)*(s1.y - s0.y)) / den;
  const m12 = ((d1.x - d0.x)*(s2.x - s0.x) - (d2.x - d0.x)*(s1.x - s0.x)) / den * -1;
  const m21 = ((d1.y - d0.y)*(s2.y - s0.y) - (d2.y - d0.y)*(s1.y - s0.y)) / den;
  const m22 = ((d1.y - d0.y)*(s2.x - s0.x) - (d2.y - d0.y)*(s1.x - s0.x)) / den * -1;
  const dx  = d0.x - m11*s0.x - m12*s0.y;
  const dy  = d0.y - m21*s0.x - m22*s0.y;

  ctx.setTransform(m11, m21, m12, m22, dx, dy);
  // mirror the video so selfie is correct
  ctx.scale(-1, 1);
  ctx.drawImage(img, -img.width, 0, img.width, img.height);
  ctx.restore();
}

// Warp a grid within bbox using displacement field
function warpFace(img, bbox, disp) {
  const [bx, by, bw, bh] = bbox;
  const cols = 22, rows = 22; // adjust for quality/perf
  const dx = bw / cols, dy = bh / rows;

  // build source & dest grids
  const src = [], dst = [];
  for (let j = 0; j <= rows; j++) {
    for (let i = 0; i <= cols; i++) {
      const x = bx + i*dx, y = by + j*dy;
      const p = { x, y };
      const q = disp(x, y);
      src.push(p); dst.push(q);
    }
  }
  // triangles
  function idx(i, j) { return j*(cols+1) + i; }
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const a = idx(i, j), b = idx(i+1, j), c = idx(i, j+1), d = idx(i+1, j+1);
      // two triangles: (a,b,d) and (a,d,c)
      mapTriangle(img, src[a], src[b], src[d], dst[a], dst[b], dst[d]);
      mapTriangle(img, src[a], src[d], src[c], dst[a], dst[d], dst[c]);
    }
  }
}

// ===== Main render =====
const off = document.createElement("canvas");
const offCtx = off.getContext("2d");

function onResults(res) {
  if (!res.image) return;

  // Prepare source frame (mirrored for selfie)
  off.width = res.image.width; off.height = res.image.height;
  offCtx.save();
  offCtx.scale(-1, 1);
  offCtx.drawImage(res.image, -off.width, 0, off.width, off.height);
  offCtx.restore();

  // Draw base (unwarped) first
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(off, 0, 0);

  const faces = res.multiFaceLandmarks || [];
  if (faces.length === 0) { statusEl.textContent = "No face detected"; return; }
  statusEl.textContent = "Face detected";

  const lm = faces[0];

  // Landmarks in pixels
  const lmPx = lm.map(p => ({ x: p.x * canvas.width, y: p.y * canvas.height }));
  const { sad, ipd } = computeSadness(lm);
  const trigger = sad > 0.52;                 // <-- emotion trigger
  const strength = Math.min(1, (sad - 0.52) * 2.2);  // ramp up morph intensity

  if (!trigger) return;

  // Build targets + displacement
  const ipdPx = ipd * canvas.width;          // pixel IPD for magnitude scaling
  const targets = sadTargets(lmPx, ipdPx, strength);
  const sigma = ipdPx * 0.55;                // falloff radius
  const disp = displacementField(lmPx, targets, sigma);

  // Warp only face bbox for speed
  const bbox = faceBBox(lm);
  warpFace(off, bbox, disp);
}

// ===== Capture (Web Share or download) =====
async function captureAndSave() {
  const blob = await new Promise(r => canvas.toBlob(r, "image/png", 0.95));
  const file = new File([blob], "crying-face.png", { type: "image/png" });
  if (navigator.canShare && navigator.canShare({ files: [file] })) {
    try { await navigator.share({ files: [file], title: "Crying Face" }); return; } catch {}
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "crying-face.png";
  document.body.appendChild(a); a.click(); a.remove();
}

// ===== UI =====
snapBtn?.addEventListener("click", captureAndSave);
flipBtn?.addEventListener("click", async () => {
  facingMode = (facingMode === "user") ? "environment" : "user";
  if (stream) stream.getTracks().forEach(t => t.stop());
  await startCamera(); await startMP();
});

// ===== Boot =====
(async () => {
  await startCamera();
  await startMP();
})();
