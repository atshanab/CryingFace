/* global clm, pModel, faceDeformer */
(function(){
  const video = document.getElementById('vid');
  const statusEl = document.getElementById('status');
  const startBtn = document.getElementById('start');
  const snapBtn = document.getElementById('snap');
  const intensityInput = document.getElementById('intensity');
  const glCanvas = document.getElementById('webgl');

  let ctrack, deformer, gl;

  async function startCamera(){
    const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode:'user', width:{ideal:1280}, height:{ideal:720} }, audio:false });
    video.srcObject = s;
    await video.play();
    statusEl.textContent = 'Camera ready';
    resize();
  }

  function resize(){
    glCanvas.width = video.videoWidth || glCanvas.clientWidth;
    glCanvas.height = video.videoHeight || glCanvas.clientHeight;
  }
  window.addEventListener('resize', resize);

  function initTracker(){
    ctrack = new clm.tracker({ useWebGL: true });
    ctrack.init(pModel);
    ctrack.start(video);
  }

  function initDeformer(){
    gl = glCanvas.getContext('webgl') || glCanvas.getContext('experimental-webgl');
    deformer = new faceDeformer(gl);
    deformer.init(video);
    deformer.load(document.createElement('canvas'), pModel.path.normalizedDelaunay);
  }

  function makeSadPositions(pts, k){
    const out = pts.map(p => [p[0], p[1]]);
    const L_BROW_IN = 19, R_BROW_IN = 15;
    const L_BROW_MID = 20, R_BROW_MID = 14;
    const MOUTH_L = 44, MOUTH_R = 50;
    const U_LIP_TOP = 47;

    const LEFT_EYE = 27, RIGHT_EYE = 32;
    const ipd = Math.hypot(pts[LEFT_EYE][0]-pts[RIGHT_EYE][0], pts[LEFT_EYE][1]-pts[RIGHT_EYE][1]) || 1;

    const browDrop = 0.25 * ipd * k;
    const pinch =  0.08 * ipd * k;
    const mouthDown = 0.28 * ipd * k;
    const lipUp = 0.06 * ipd * k;

    out[L_BROW_IN][1] += browDrop; out[R_BROW_IN][1] += browDrop;
    out[L_BROW_IN][0] -= pinch;    out[R_BROW_IN][0] += pinch;
    out[L_BROW_MID][1] += browDrop * 0.7; out[R_BROW_MID][1] += browDrop * 0.7;
    out[L_BROW_MID][0] -= pinch * 0.5;    out[R_BROW_MID][0] += pinch * 0.5;

    out[MOUTH_L][1] += mouthDown;  out[MOUTH_R][1] += mouthDown;
    out[MOUTH_L][0] += pinch * 0.2; out[MOUTH_R][0] -= pinch * 0.2;

    out[U_LIP_TOP][1] -= lipUp;

    return out;
  }

  function animate(){
    requestAnimationFrame(animate);
    if (!ctrack) return;
    const positions = ctrack.getCurrentPosition();
    if (!positions) { statusEl.textContent = 'No face detected'; return; }
    statusEl.textContent = 'Face detected';
    const k = parseFloat(intensityInput.value || '0.85');
    const target = makeSadPositions(positions, k);
    deformer.draw(video, positions, target);
  }

  snapBtn.addEventListener('click', () => {
    const w = glCanvas.width, h = glCanvas.height;
    const shot = document.createElement('canvas'); shot.width = w; shot.height = h;
    const sctx = shot.getContext('2d');
    sctx.translate(w,0); sctx.scale(-1,1);
    sctx.drawImage(video,0,0,w,h);
    sctx.setTransform(1,0,0,1,0,0);
    sctx.drawImage(glCanvas,0,0,w,h);
    const a = document.createElement('a'); a.href = shot.toDataURL('image/png'); a.download = 'crying-face.png';
    document.body.appendChild(a); a.click(); a.remove();
  });

  document.getElementById('start').addEventListener('click', async () => {
    try{
      await startCamera();
      initTracker();
      initDeformer();
      animate();
    }catch(e){ statusEl.textContent = 'Camera blocked'; }
  });

  (async ()=>{
    try{
      await startCamera();
      initTracker();
      initDeformer();
      animate();
    }catch(e){}
  })();
})();