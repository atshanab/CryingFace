/* global PIXI, FaceMesh, Camera */
(function(){
  const video = document.getElementById("cam");
  const statusEl = document.getElementById("status");
  const startBtn = document.getElementById("start");
  const snapBtn  = document.getElementById("snap");
  const intensityInput = document.getElementById("intensity");

  async function startCamera() {
    const s = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: {ideal: 1280}, height: {ideal: 720} },
      audio: false
    });
    video.srcObject = s;
    await video.play();
    statusEl.textContent = "Camera ready";
  }

  const app = new PIXI.Application({ resizeTo: window, backgroundAlpha: 1, antialias: true });
  document.body.appendChild(app.view);
  const texture = PIXI.Texture.from(video);
  texture.baseTexture.autoUpdate = true;

  const container = new PIXI.Container();
  app.stage.addChild(container);

  const GRID_X = 22, GRID_Y = 22;
  let mesh, verts, uvs, indices;

  function buildMesh(){
    const w = app.renderer.width, h = app.renderer.height;
    const dx = 1/(GRID_X-1), dy = 1/(GRID_Y-1);
    verts = new Float32Array(GRID_X*GRID_Y*2);
    uvs   = new Float32Array(GRID_X*GRID_Y*2);
    const inds = [];
    let p=0,q=0;
    for(let j=0;j<GRID_Y;j++){
      for(let i=0;i<GRID_X;i++){
        const x=i*dx, y=j*dy;
        verts[p++]=x*w; verts[p++]=y*h;
        uvs[q++]=1-x;  uvs[q++]=y;
      }
    }
    for(let j=0;j<GRID_Y-1;j++){
      for(let i=0;i<GRID_X-1;i++){
        const a=j*GRID_X+i, b=a+1, c=a+GRID_X, d=c+1;
        inds.push(a,b,d, a,d,c);
      }
    }
    indices = new Uint16Array(inds);
    mesh = new PIXI.Mesh({
      geometry: new PIXI.Geometry()
        .addAttribute("aVertexPosition", verts, 2)
        .addAttribute("aTextureCoord", uvs, 2)
        .addIndex(indices),
      texture,
      drawMode: PIXI.DRAW_MODES.TRIANGLES
    });
    container.removeChildren();
    container.addChild(mesh);
  }
  buildMesh();
  window.addEventListener("resize", buildMesh);

  const faceMesh = new FaceMesh({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${f}` });
  faceMesh.setOptions({ maxNumFaces: 1, refineLandmarks: true, minDetectionConfidence: 0.6, minTrackingConfidence: 0.6 });
  faceMesh.onResults(onResults);

  let mpCamera;
  async function startMP() {
    if (mpCamera) mpCamera.stop();
    mpCamera = new Camera(video, { width: 1280, height: 720, onFrame: async () => { await faceMesh.send({ image: video }); } });
    mpCamera.start();
  }

  const CTRL = { L_INNER_BROW:65, L_MID_BROW:70, R_INNER_BROW:295, R_MID_BROW:300, L_UP:159, R_UP:386, M_L:61, M_R:291, U_LIP:13 };

  function rbfDeform(lmPx, intensity){
    const w = app.renderer.width, h = app.renderer.height;
    const ipd = Math.hypot(lmPx[33].x - lmPx[263].x, lmPx[33].y - lmPx[263].y) || 1;
    const px = ipd;
    const t = new Map();
    const add=(id,dx,dy)=>t.set(id,{x:lmPx[id].x+dx,y:lmPx[id].y+dy});
    const brow=0.14*px*intensity, pinch=0.04*px*intensity, mouth=0.22*px*intensity, lid=0.06*px*intensity;
    add(CTRL.L_INNER_BROW,-pinch,brow); add(CTRL.R_INNER_BROW,pinch,brow);
    add(CTRL.L_MID_BROW,-pinch*0.6,brow*0.6); add(CTRL.R_MID_BROW,pinch*0.6,brow*0.6);
    add(CTRL.M_L,0,mouth); add(CTRL.M_R,0,mouth); add(CTRL.U_LIP,0,-mouth*0.25);
    add(CTRL.L_UP,0,lid); add(CTRL.R_UP,0,lid);
    const centers=[...t.keys()].map(k=>({sx:lmPx[k].x,sy:lmPx[k].y,dx:t.get(k).x-lmPx[k].x,dy:t.get(k).y-lmPx[k].y}));
    const sigma=px*0.6, twoSig2=2*sigma*sigma;
    for(let i=0;i<verts.length;i+=2){
      const x0=verts[i], y0=verts[i+1];
      let wx=0, wy=0, ws=0;
      for(const c of centers){
        const dx=x0-c.sx, dy=y0-c.sy;
        const wgt=Math.exp(-(dx*dx+dy*dy)/twoSig2);
        wx+=wgt*c.dx; wy+=wgt*c.dy; ws+=wgt;
      }
      if(ws>1e-6){ verts[i]=x0+wx/ws; verts[i+1]=y0+wy/ws; }
    }
    mesh.geometry.getBuffer("aVertexPosition").update(verts);
  }

  function onResults(res){
    if(!res.image || !res.multiFaceLandmarks || !res.multiFaceLandmarks.length){
      statusEl.textContent = "No face detected";
      return;
    }
    statusEl.textContent = "Face detected";
    buildMesh(); // reset
    const w = app.renderer.width, h = app.renderer.height;
    const lmPx = res.multiFaceLandmarks[0].map(p => ({ x:(1-p.x)*w, y:p.y*h })); // mirrored
    const k = parseFloat(intensityInput.value || "0.8");
    rbfDeform(lmPx, k);
  }

  snapBtn.addEventListener("click", () => {
    const data = app.renderer.extract.base64(app.stage);
    const a = document.createElement("a"); a.href=data; a.download="crying-face.png"; document.body.appendChild(a); a.click(); a.remove();
  });

  startBtn.addEventListener("click", async () => {
    try{ await startCamera(); await startMP(); } catch{ statusEl.textContent="Camera blocked"; }
  });
  (async () => { try{ await startCamera(); await startMP(); } catch(e){} })();
})();