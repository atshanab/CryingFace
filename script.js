/* global FaceMesh, Camera */
const video = document.getElementById("cam");
const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("start");
const snapBtn = document.getElementById("snap");
const flipBtn = document.getElementById("flip");
const statusEl = document.getElementById("status");
const forceCB = document.getElementById("force");

let facingMode = "user";
let stream;

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false
  });
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth || 1280;
  canvas.height = video.videoHeight || 720;
  statusEl.textContent = "Camera ready – look sad 😢";
}

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

let sadEMA = 0, blinkEMA = 0;
const ALPHA = 0.18;
function computeSadness(lm) {
  const L_INNER_BROW = 65, R_INNER_BROW = 295;
  const L_EYE_UP = 159, L_EYE_LOW = 145;
  const R_EYE_UP = 386, R_EYE_LOW = 374;
  const M_L = 61, M_R = 291, U_LIP = 13;
  const ipd = Math.hypot(lm[33].x - lm[263].x, lm[33].y - lm[263].y) + 1e-6;
  const browEyeL = Math.abs(lm[L_INNER_BROW].y - lm[L_EYE_UP].y) / ipd;
  const browEyeR = Math.abs(lm[R_INNER_BROW].y - lm[R_EYE_UP].y) / ipd;
  const browLower = Math.max(0, 0.18 - (browEyeL + browEyeR) * 0.5) / 0.18;
  const mouthDown = Math.max(0, ((lm[M_L].y + lm[M_R].y) * 0.5 - lm[U_LIP].y) / ipd - 0.05) * 3.0;
  const earL = Math.abs(lm[L_EYE_UP].y - lm[L_EYE_LOW].y) / ipd;
  const earR = Math.abs(lm[R_EYE_UP].y - lm[R_EYE_LOW].y) / ipd;
  const blink = Math.max(0, 0.06 - (earL + earR) * 0.5) / 0.06;
  const s = 0.6 * browLower + 0.4 * Math.min(1, mouthDown);
  sadEMA   = (1 - ALPHA) * sadEMA + ALPHA * s;
  blinkEMA = (1 - ALPHA) * blinkEMA + ALPHA * blink;
  return { sad: sadEMA, blink: blinkEMA, ipd };
}

const CTRL = { L_INNER_BROW:65, L_MID_BROW:70, R_INNER_BROW:295, R_MID_BROW:300, M_L:61, M_R:291, U_LIP:13, L_LIP:14, L_EYE_UP:159, R_EYE_UP:386 };

function faceBBox(lm) {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const p of lm) { minX=Math.min(minX,p.x); maxX=Math.max(maxX,p.x); minY=Math.min(minY,p.y); maxY=Math.max(maxY,p.y); }
  const pad = 0.05;
  minX = Math.max(0, minX - pad); minY = Math.max(0, minY - pad);
  maxX = Math.min(1, maxX + pad); maxY = Math.min(1, maxY + pad);
  return [minX*canvas.width, minY*canvas.height, (maxX-minX)*canvas.width, (maxY-minY)*canvas.height];
}

function sadTargets(lmPx, ipdPx, strength) {
  const t = new Map(), add=(id,dx,dy)=>t.set(id,{x:lmPx[id].x+dx,y:lmPx[id].y+dy});
  const browDrop = 0.12 * ipdPx * strength, pinch = 0.03 * ipdPx * strength, mouthDown=0.18 * ipdPx * strength, lid=0.06 * ipdPx * strength;
  add(CTRL.L_INNER_BROW,-pinch,browDrop); add(CTRL.R_INNER_BROW,pinch,browDrop);
  add(CTRL.L_MID_BROW,-pinch*0.6,browDrop*0.6); add(CTRL.R_MID_BROW,pinch*0.6,browDrop*0.6);
  add(CTRL.M_L,0,mouthDown); add(CTRL.M_R,0,mouthDown); add(CTRL.U_LIP,0,-mouthDown*0.25);
  add(CTRL.L_EYE_UP,0,lid); add(CTRL.R_EYE_UP,0,lid);
  return t;
}

function displacementField(lmPx, targets, sigma) {
  const ctrls=[...targets.keys()];
  const deltas = ctrls.map(id=>{const s=lmPx[id],d=targets.get(id);return {sx:s.x,sy:s.y,dx:d.x-s.x,dy:d.y-s.y};});
  const twoSigma2=2*sigma*sigma;
  return function disp(x,y){let wx=0,wy=0,ws=0;for(const c of deltas){const dx=x-c.sx,dy=y-c.sy;const w=Math.exp(-(dx*dx+dy*dy)/twoSigma2);wx+=w*c.dx;wy+=w*c.dy;ws+=w;}if(ws<1e-6)return{x,y};return{x:x+wx/ws,y:y+wy/ws};};
}

function affineFromTriangles(s0,s1,s2,d0,d1,d2){
  function inv3(m){const[a,b,c,d,e,f,g,h,i]=m;const A=e*i-f*h,B=-(d*i-f*g),C=d*h-e*g,D=-(b*i-c*h),E=a*i-c*g,F=-(a*h-b*g),G=b*f-c*e,H=-(a*f-b*d),I=a*e-b*d;const det=a*A+b*B+c*C;if(Math.abs(det)<1e-8)return null;return[A/det,D/det,G/det,B/det,E/det,H/det,C/det,F/det,I/det];}
  function mul(m,v){return[m[0]*v[0]+m[1]*v[1]+m[2]*v[2],m[3]*v[0]+m[4]*v[1]+m[5]*v[2],m[6]*v[0]+m[7]*v[1]+m[8]*v[2]];}
  const S=[s0.x,s0.y,1, s1.x,s1.y,1, s2.x,s2.y,1]; const invS=inv3(S); if(!invS) return null;
  const Dx=mul(invS,[d0.x,d1.x,d2.x]); const Dy=mul(invS,[d0.y,d1.y,d2.y]);
  return {a:Dx[0], c:Dx[1], e:Dx[2], b:Dy[0], d:Dy[1], f:Dy[2]};
}

function mapTriangle(img, s0,s1,s2, d0,d1,d2){
  const A=affineFromTriangles(s0,s1,s2,d0,d1,d2); if(!A) return;
  ctx.save(); ctx.beginPath(); ctx.moveTo(d0.x,d0.y); ctx.lineTo(d1.x,d1.y); ctx.lineTo(d2.x,d2.y); ctx.closePath(); ctx.clip();
  ctx.setTransform(A.a,A.b,A.c,A.d,A.e,A.f); ctx.drawImage(img,0,0); ctx.restore();
}

function warpFace(srcImg, bbox, disp){
  const [bx,by,bw,bh]=bbox; const cols=20, rows=20; const dx=bw/cols, dy=bh/rows;
  const src=[], dst=[];
  for(let j=0;j<=rows;j++){for(let i=0;i<=cols;i++){const x=bx+i*dx,y=by+j*dy;const p={x,y},q=disp(x,y);src.push(p);dst.push(q);}}
  const idx=(i,j)=> j*(cols+1)+i;
  for(let j=0;j<rows;j++){for(let i=0;i<cols;i++){const a=idx(i,j),b=idx(i+1,j),c=idx(i,j+1),d=idx(i+1,j+1);mapTriangle(srcImg,src[a],src[b],src[d],dst[a],dst[b],dst[d]);mapTriangle(srcImg,src[a],src[d],src[c],dst[a],dst[d],dst[c]);}}
}

const off = document.createElement("canvas"); const offCtx = off.getContext("2d");

function onResults(res){
  if(!res.image) return;
  off.width = res.image.width; off.height = res.image.height;
  offCtx.clearRect(0,0,off.width,off.height);
  offCtx.drawImage(res.image, 0, 0);

  ctx.setTransform(1,0,0,1,0,0);
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);

  const faces = res.multiFaceLandmarks || [];
  if (faces.length === 0){ statusEl.textContent = "No face detected"; return; }
  statusEl.textContent = "Face detected";

  const lm = faces[0];
  const { sad, ipd } = computeSadness(lm);
  const trigger = forceCB.checked || (sad > 0.52);
  const strength = forceCB.checked ? 0.9 : Math.min(1, (sad - 0.52) * 2.2);
  if(!trigger) return;

  const lmPx = lm.map(p => ({ x: p.x * canvas.width, y: p.y * canvas.height }));
  const ipdPx = ipd * canvas.width;
  const targets = sadTargets(lmPx, ipdPx, Math.max(0.3, strength));
  const sigma = ipdPx * 0.55;
  const disp = displacementField(lmPx, targets, sigma);
  const bbox = faceBBox(lm);
  warpFace(off, bbox, disp);
}

async function captureAndSave() {
  const blob = await new Promise(r => canvas.toBlob(r, "image/png", 0.95));
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "crying-face.png";
  document.body.appendChild(a); a.click(); a.remove();
}

startBtn.addEventListener("click", async () => { try{ await startCamera(); await startMP(); }catch(e){ statusEl.textContent='Camera blocked'; } });
flipBtn.addEventListener("click", async () => {
  facingMode = (facingMode === "user") ? "environment" : "user";
  if (stream) stream.getTracks().forEach(t => t.stop());
  await startCamera(); await startMP();
});
snapBtn.addEventListener("click", captureAndSave);

(async () => { try{ await startCamera(); await startMP(); } catch(e){} })();
