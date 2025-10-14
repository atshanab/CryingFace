/* global FaceMesh, Camera */
const video = document.getElementById("cam");
const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("start");
const snapBtn  = document.getElementById("snap");
const flipBtn  = document.getElementById("flip");
const statusEl = document.getElementById("status");
const intensityInput = document.getElementById("intensity"); // keep if you have it; else set fixed strength

let facingMode = "user";
let stream;

async function startCamera() {
  const constraints = { video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false };
  stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth || 1280;
  canvas.height = video.videoHeight || 720;
  statusEl && (statusEl.textContent = "Camera ready");
}

// ---- FaceMesh ----
const faceMesh = new FaceMesh({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${f}` });
faceMesh.setOptions({ maxNumFaces: 1, refineLandmarks: true, minDetectionConfidence: 0.6, minTrackingConfidence: 0.6 });
faceMesh.onResults(onResults);

let mpCamera;
async function startMP() {
  if (mpCamera) mpCamera.stop();
  mpCamera = new Camera(video, { width: 1280, height: 720, onFrame: async () => { await faceMesh.send({ image: video }); } });
  mpCamera.start();
}

// ---- Utility ----
const off = document.createElement("canvas");
const offCtx = off.getContext("2d");
const patch = document.createElement("canvas");   // where we compose deformed pieces
const pctx  = patch.getContext("2d");

function px(p) { return { x: p.x * canvas.width, y: p.y * canvas.height }; }

function rectFromPoints(points, pad = 12) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of points) { minX = Math.min(minX, p.x); minY = Math.min(minY, p.y); maxX = Math.max(maxX, p.x); maxY = Math.max(maxY, p.y); }
  return { x: Math.max(0, minX - pad), y: Math.max(0, minY - pad),
           w: Math.min(canvas.width,  maxX + pad) - Math.max(0, minX - pad),
           h: Math.min(canvas.height, maxY + pad) - Math.max(0, minY - pad) };
}

function featherMask(ctx, w, h, r = 22) {
  // radial/rectangular feather mask (white center, transparent edges)
  const g = ctx.createLinearGradient(0, 0, 0, h);
  ctx.clearRect(0,0,w,h);
  ctx.save();
  // Outer transparent
  ctx.globalCompositeOperation = "source-over";
  ctx.fillStyle = "black";
  ctx.fillRect(0,0,w,h);
  // Inner white with feather
  ctx.globalCompositeOperation = "lighter";
  ctx.save();
  ctx.fillStyle = "white";
  ctx.beginPath();
  // rounded rect
  const rr = Math.min(r, w*0.25, h*0.25);
  ctx.moveTo(rr, 0);
  ctx.arcTo(w,0,w,rr, rr);
  ctx.arcTo(w,h,w-rr,h, rr);
  ctx.arcTo(0,h,0,h-rr, rr);
  ctx.arcTo(0,0,rr,0, rr);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
  // blur via drawImage trick
  try { ctx.filter = "blur(10px)"; ctx.drawImage(ctx.canvas, 0, 0); ctx.filter = "none"; } catch {}
  ctx.restore();
}

function drawAffinePiece(src, srcQuad, dstQuad) {
  // srcQuad/dstQuad: [{x,y}x3] triangles — we do two per quad (split)
  function tri(a,b,c){ return [a,b,c]; }
  const S1 = tri(srcQuad[0], srcQuad[1], srcQuad[2]), S2 = tri(srcQuad[0], srcQuad[2], srcQuad[3]);
  const D1 = tri(dstQuad[0], dstQuad[1], dstQuad[2]), D2 = tri(dstQuad[0], dstQuad[2], dstQuad[3]);

  function mapTriangle(s0,s1,s2,d0,d1,d2){
    // build affine mapping from src to dst
    function inv3(m){const[a,b,c,d,e,f,g,h,i]=m;const A=e*i-f*h,B=-(d*i-f*g),C=d*h-e*g,D=-(b*i-c*h),E=a*i-c*g,F=-(a*h-b*g),G=b*f-c*e,H=-(a*f-b*d),I=a*e-b*d;const det=a*A+b*B+c*C;if(Math.abs(det)<1e-8)return null;return[A/det,D/det,G/det,B/det,E/det,H/det,C/det,F/det,I/det];}
    function mul(m,v){return[m[0]*v[0]+m[1]*v[1]+m[2]*v[2],m[3]*v[0]+m[4]*v[1]+m[5]*v[2],m[6]*v[0]+m[7]*v[1]+m[8]*v[2]];}
    const S=[s0.x,s0.y,1, s1.x,s1.y,1, s2.x,s2.y,1]; const invS=inv3(S); if(!invS) return;
    const Dx=mul(invS,[d0.x,d1.x,d2.x]); const Dy=mul(invS,[d0.y,d1.y,d2.y]);
    const a=Dx[0], c=Dx[1], e=Dx[2], b=Dy[0], d=Dy[1], f=Dy[2];

    pctx.save();
    pctx.beginPath();
    pctx.moveTo(d0.x, d0.y); pctx.lineTo(d1.x, d1.y); pctx.lineTo(d2.x, d2.y); pctx.closePath();
    pctx.clip();
    pctx.setTransform(a,b,c,d,e,f);
    pctx.drawImage(src, 0, 0);
    pctx.restore();
  }

  mapTriangle(S1[0],S1[1],S1[2], D1[0],D1[1],D1[2]);
  mapTriangle(S2[0],S2[1],S2[2], D2[0],D2[1],D2[2]);
}

// ---- Landmarks groups we’ll deform ----
const IDX = {
  // Brows (small quads that we nudge downward/inward)
  LBROW_IN: 65, LBROW_MID:70, RBROW_IN:295, RBROW_MID:300,
  // Eyelids (upper only; slight droop)
  L_UP:159, L_OUT:33, L_IN:133, R_UP:386, R_OUT:263, R_IN:362,
  // Mouth corners + top lip
  M_L:61, M_R:291, U_LIP:13, L_LIP:14
};

function onResults(res) {
  if (!res.image) return;

  // base frame
  off.width = res.image.width; off.height = res.image.height;
  offCtx.setTransform(1,0,0,1,0,0);
  offCtx.drawImage(res.image, 0, 0);

  // show base on screen first
  ctx.setTransform(1,0,0,1,0,0);
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);

  const faces = res.multiFaceLandmarks || [];
  if (!faces.length) { statusEl && (statusEl.textContent = "No face detected"); return; }
  statusEl && (statusEl.textContent = "Face detected");

  const lm = faces[0].map(px);

  // patch canvas same size as screen canvas
  patch.width = canvas.width; patch.height = canvas.height;
  pctx.setTransform(1,0,0,1,0,0);
  pctx.clearRect(0,0,patch.width,patch.height);

  const strength = intensityInput ? parseFloat(intensityInput.value || "0.7") : 0.7;
  const ipd = Math.hypot(lm[33].x - lm[263].x, lm[33].y - lm[263].y) + 1e-6;

  // ---- 1) Mouth patch (downturned corners, slight top lip raise) ----
  {
    const ml = lm[IDX.M_L], mr = lm[IDX.M_R], ul = lm[IDX.U_LIP], ll = lm[IDX.L_LIP];
    const pad = Math.max(20, ipd * 0.45);
    const mouthRect = rectFromPoints([ml, mr, ul, ll], pad);

    // source quad corners (rect)
    const SQ = [
      {x: mouthRect.x,                 y: mouthRect.y},
      {x: mouthRect.x + mouthRect.w,  y: mouthRect.y},
      {x: mouthRect.x + mouthRect.w,  y: mouthRect.y + mouthRect.h},
      {x: mouthRect.x,                 y: mouthRect.y + mouthRect.h},
    ];

    // destination quad (corners down, center slightly up)
    const down = ipd * 0.22 * strength;
    const up   = ipd * 0.06 * strength;

    const DQ = [
      {x: SQ[0].x, y: SQ[0].y + up},                    // top-left slightly down (for curvature)
      {x: SQ[1].x, y: SQ[1].y + up},                    // top-right slightly down
      {x: SQ[2].x, y: SQ[2].y + down},                  // bottom-right more down (corner)
      {x: SQ[3].x, y: SQ[3].y + down},                  // bottom-left  more down
    ];

    // Compose patch with feather mask
    const mouthPatch = document.createElement("canvas");
    mouthPatch.width = patch.width; mouthPatch.height = patch.height;
    const mctx = mouthPatch.getContext("2d");

    // copy the rectangular area
    mctx.drawImage(off, 0,0,off.width,off.height, 0,0,patch.width,patch.height);

    // feather mask
    const mask = document.createElement("canvas");
    mask.width = mouthPatch.width; mask.height = mouthPatch.height;
    const msk = mask.getContext("2d");
    featherMask(msk, mask.width, mask.height);

    // draw warped quad onto pctx
    drawAffinePiece(mouthPatch, SQ, DQ);

    // alpha-feather into final patch
    pctx.save();
    pctx.globalCompositeOperation = "destination-in"; // keep only masked region of what we just drew
    pctx.drawImage(mask, 0, 0);
    pctx.restore();

    // reset comp op for next patches
    pctx.globalCompositeOperation = "source-over";
  }

  // ---- 2) Brows patches (lower + slight inward pinch) ----
  function browQuad(center, width, height, angle = 0) {
    const hw = width * 0.5, hh = height * 0.5;
    const pts = [ {x:-hw,y:-hh}, {x:hw,y:-hh}, {x:hw,y:hh}, {x:-hw,y:hh} ];
    const s = Math.sin(angle), c = Math.cos(angle);
    return pts.map(p => ({ x: center.x + p.x*c - p.y*s, y: center.y + p.x*s + p.y*c }));
  }

  function drawBrow(center, isLeft) {
    const w = ipd * 0.55, h = ipd * 0.22;
    const SQ = browQuad(center, w, h, 0);
    const drop = ipd * 0.14 * strength, pinch = (isLeft ? -1 : 1) * ipd * 0.04 * strength;
    const DQ = SQ.map(p => ({ x: p.x + pinch, y: p.y + drop }));

    const browPatch = document.createElement("canvas");
    browPatch.width = patch.width; browPatch.height = patch.height;
    const bctx = browPatch.getContext("2d");
    bctx.drawImage(off, 0,0,off.width,off.height, 0,0,patch.width,patch.height);

    // mask
    const mask = document.createElement("canvas");
    mask.width = browPatch.width; mask.height = browPatch.height;
    featherMask(mask.getContext("2d"), mask.width, mask.height);

    drawAffinePiece(browPatch, SQ, DQ);
    pctx.save(); pctx.globalCompositeOperation = "destination-in"; pctx.drawImage(mask, 0, 0); pctx.restore();
    pctx.globalCompositeOperation = "source-over";
  }

  drawBrow(lm[IDX.LBROW_MID], true);
  drawBrow(lm[IDX.RBROW_MID], false);

  // ---- 3) Upper eyelid slight droop (very gentle) ----
  function eyelidQuad(up, inn, out) {
    const cx = (inn.x + out.x) * 0.5, cy = (inn.y + out.y) * 0.5;
    const w = Math.hypot(out.x - inn.x, out.y - inn.y) * 0.9;
    const h = w * 0.55;
    return browQuad({x:cx,y:cy}, w, h, 0);
  }
  function drawLid(up, inn, out, isLeft) {
    const SQ = eyelidQuad(up, inn, out);
    const droop = ipd * 0.06 * strength;
    const DQ = SQ.map(p => ({ x: p.x, y: p.y + droop }));

    const lid = document.createElement("canvas");
    lid.width = patch.width; lid.height = patch.height;
    const lctx = lid.getContext("2d");
    lctx.drawImage(off, 0,0,off.width,off.height, 0,0,patch.width,patch.height);

    const mask = document.createElement("canvas");
    mask.width = lid.width; mask.height = lid.height;
    featherMask(mask.getContext("2d"), mask.width, mask.height);

    drawAffinePiece(lid, SQ, DQ);
    pctx.save(); pctx.globalCompositeOperation = "destination-in"; pctx.drawImage(mask, 0, 0); pctx.restore();
    pctx.globalCompositeOperation = "source-over";
  }
  drawLid(lm[IDX.L_UP], lm[IDX.L_IN], lm[IDX.L_OUT], true);
  drawLid(lm[IDX.R_UP], lm[IDX.R_IN], lm[IDX.R_OUT], false);

  // ---- Composite deformed patches back onto base frame ----
  ctx.globalAlpha = 1.0;
  ctx.drawImage(patch, 0, 0);
}

// ---- Capture / UI ----
async function captureAndSave() {
  const blob = await new Promise(r => canvas.toBlob(r, "image/png", 0.95));
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "crying-face.png";
  document.body.appendChild(a); a.click(); a.remove();
}
startBtn && startBtn.addEventListener("click", async () => { try { await startCamera(); await startMP(); } catch(e){ statusEl && (statusEl.textContent='Camera blocked'); } });
flipBtn  && flipBtn.addEventListener("click", async () => {
  facingMode = (facingMode === "user") ? "environment" : "user";
  if (stream) stream.getTracks().forEach(t => t.stop());
  await startCamera(); await startMP();
});
snapBtn  && snapBtn.addEventListener("click", captureAndSave);

(async () => { try { await startCamera(); await startMP(); } catch(e){} })();
