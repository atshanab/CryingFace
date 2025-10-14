/* global FaceMesh, Camera */
(function(){
  const video = document.getElementById("cam");
  const overlay = document.getElementById("overlay");
  const ctx = overlay.getContext("2d");
  const statusEl = document.getElementById("status");
  const startBtn = document.getElementById("start");
  const snapBtn  = document.getElementById("snap");
  const intensityInput = document.getElementById("intensity");

  // Camera
  async function startCamera() {
    const s = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: {ideal: 1280}, height: {ideal: 720} },
      audio: false
    });
    video.srcObject = s;
    await video.play();
    resize();
    statusEl.textContent = "Camera ready";
  }
  function resize(){
    overlay.width = video.videoWidth || overlay.clientWidth;
    overlay.height = video.videoHeight || overlay.clientHeight;
  }
  window.addEventListener("resize", resize);

  // Mediapipe
  const faceMesh = new FaceMesh({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${f}` });
  faceMesh.setOptions({ maxNumFaces:1, refineLandmarks:true, minDetectionConfidence:0.6, minTrackingConfidence:0.6 });
  faceMesh.onResults(onResults);

  let mpCamera;
  async function startMP() {
    if (mpCamera) mpCamera.stop();
    mpCamera = new Camera(video, { width: 1280, height: 720, onFrame: async () => { await faceMesh.send({ image: video }); } });
    mpCamera.start();
  }

  // Utilities
  function px(p){ return { x:(1-p.x)*overlay.width, y:p.y*overlay.height }; } // mirror x
  function rectFromPts(pts, pad){
    let minX=1e9,minY=1e9,maxX=-1e9,maxY=-1e9;
    for(const p of pts){ minX=Math.min(minX,p.x); minY=Math.min(minY,p.y); maxX=Math.max(maxX,p.x); maxY=Math.max(maxY,p.y); }
    return { x: Math.max(0,minX-pad), y: Math.max(0,minY-pad),
             w: Math.min(overlay.width, maxX+pad) - Math.max(0,minX-pad),
             h: Math.min(overlay.height, maxY+pad) - Math.max(0,minY-pad) };
  }
  function featherMask(w,h,r=26){
    const c=document.createElement("canvas"); c.width=w; c.height=h;
    const k=c.getContext("2d");
    const rr=Math.min(r,w*0.25,h*0.25);
    k.fillStyle="#fff";
    k.beginPath();
    k.moveTo(rr,0);
    k.arcTo(w,0,w,rr,rr);
    k.arcTo(w,h,w-rr,h,rr);
    k.arcTo(0,h,0,h-rr,rr);
    k.arcTo(0,0,rr,0,rr);
    k.closePath(); k.fill();
    // blur soften
    try{ k.filter="blur(12px)"; k.drawImage(c,0,0); k.filter="none"; }catch{}
    return c;
  }
  function affineFromTriangles(s0,s1,s2,d0,d1,d2){
    function inv3(m){const[a,b,c,d,e,f,g,h,i]=m;const A=e*i-f*h,B=-(d*i-f*g),C=d*h-e*g,D=-(b*i-c*h),E=a*i-c*g,F=-(a*h-b*g),G=b*f-c*e,H=-(a*f-b*d),I=a*e-b*d;const det=a*A+b*B+c*C;if(Math.abs(det)<1e-8)return null;return[A/det,D/det,G/det,B/det,E/det,H/det,C/det,F/det,I/det];}
    function mul(m,v){return[m[0]*v[0]+m[1]*v[1]+m[2]*v[2],m[3]*v[0]+m[4]*v[1]+m[5]*v[2],m[6]*v[0]+m[7]*v[1]+m[8]*v[2]];}
    const S=[s0.x,s0.y,1, s1.x,s1.y,1, s2.x,s2.y,1]; const invS=inv3(S); if(!invS)return null;
    const Dx=mul(invS,[d0.x,d1.x,d2.x]); const Dy=mul(invS,[d0.y,d1.y,d2.y]);
    return {a:Dx[0], c:Dx[1], e:Dx[2], b:Dy[0], d:Dy[1], f:Dy[2]};
  }
  function drawTriWarp(targetCtx, src, s0,s1,s2, d0,d1,d2){
    const A=affineFromTriangles(s0,s1,s2,d0,d1,d2); if(!A) return;
    targetCtx.save();
    targetCtx.beginPath(); targetCtx.moveTo(d0.x,d0.y); targetCtx.lineTo(d1.x,d1.y); targetCtx.lineTo(d2.x,d2.y); targetCtx.closePath();
    targetCtx.clip();
    targetCtx.setTransform(A.a,A.b,A.c,A.d,A.e,A.f);
    targetCtx.drawImage(src,0,0);
    targetCtx.restore();
    targetCtx.setTransform(1,0,0,1,0,0);
  }
  function drawQuadWarp(targetCtx, src, SQ, DQ){
    // two triangles
    drawTriWarp(targetCtx, src, SQ[0],SQ[1],SQ[2], DQ[0],DQ[1],DQ[2]);
    drawTriWarp(targetCtx, src, SQ[0],SQ[2],SQ[3], DQ[0],DQ[2],DQ[3]);
  }

  const IDX = { LBROW_IN:65, LBROW_MID:70, RBROW_IN:295, RBROW_MID:300, L_UP:159, R_UP:386, M_L:61, M_R:291, U_LIP:13, L_LIP:14 };

  function onResults(res){
    if(!res.image){ return; }
    ctx.clearRect(0,0,overlay.width,overlay.height);

    const faces = res.multiFaceLandmarks || [];
    if(!faces.length){ statusEl.textContent="No face detected"; return; }
    statusEl.textContent="Face detected";

    const lm = faces[0].map(px);
    const ipd = Math.hypot(lm[33].x-lm[263].x, lm[33].y-lm[263].y) || 1;
    const k = parseFloat(intensityInput.value || "0.8");

    // Prepare a snapshot of the current video frame into an offscreen (same size as overlay)
    const src = document.createElement("canvas");
    src.width = overlay.width; src.height = overlay.height;
    const sctx = src.getContext("2d");
    // draw the mirrored video into the offscreen
    sctx.save();
    sctx.translate(src.width, 0); sctx.scale(-1,1);
    sctx.drawImage(video, 0, 0, src.width, src.height);
    sctx.restore();

    // Helper to render one rectangular patch with feathered mask
    function renderPatch(rect, deform){
      // Source/dest quads
      const SQ=[
        {x:rect.x, y:rect.y},
        {x:rect.x+rect.w, y:rect.y},
        {x:rect.x+rect.w, y:rect.y+rect.h},
        {x:rect.x, y:rect.y+rect.h},
      ];
      const DQ=deform(SQ);

      // Draw warped into temp canvas
      const temp = document.createElement("canvas");
      temp.width=overlay.width; temp.height=overlay.height;
      const tctx = temp.getContext("2d");
      drawQuadWarp(tctx, src, SQ, DQ);

      // Mask
      const mask = featherMask(rect.w, rect.h, Math.max(16, ipd*0.2));
      ctx.save();
      ctx.globalCompositeOperation="source-over";
      // Use mask by drawing temp through clipped rect + mask alpha
      // Draw mask into alpha channel via destination-in on a local buffer
      // Simpler: set globalAlpha via mask by using pattern — instead, draw mask to an offscreen then use it as clip
      const maskCanvas = document.createElement("canvas");
      maskCanvas.width=overlay.width; maskCanvas.height=overlay.height;
      const mctx = maskCanvas.getContext("2d");
      mctx.drawImage(mask, 0,0,mask.width,mask.height, rect.x, rect.y, rect.w, rect.h);

      // Apply mask: put mask as global alpha using 'destination-in'
      const comp = document.createElement("canvas");
      comp.width=overlay.width; comp.height=overlay.height;
      const cctx = comp.getContext("2d");
      cctx.drawImage(temp,0,0);
      cctx.globalCompositeOperation="destination-in";
      cctx.drawImage(maskCanvas,0,0);

      // Now paint to final
      ctx.drawImage(comp,0,0);
      ctx.restore();
    }

    // 1) Mouth
    {
      const ml = lm[IDX.M_L], mr = lm[IDX.M_R], ul = lm[IDX.U_LIP], ll=lm[IDX.L_LIP];
      const rect = rectFromPts([ml,mr,ul,ll], Math.max(20, ipd*0.5));
      const down = 0.22*ipd*k, up = 0.06*ipd*k;
      renderPatch(rect, (SQ)=>[
        {x:SQ[0].x, y:SQ[0].y + up},
        {x:SQ[1].x, y:SQ[1].y + up},
        {x:SQ[2].x, y:SQ[2].y + down},
        {x:SQ[3].x, y:SQ[3].y + down},
      ]);
    }
    // 2) Brows
    function browRect(center){
      const w = ipd*0.55, h=ipd*0.22;
      return { x:center.x-w/2, y:center.y-h/2, w, h };
    }
    const bd = 0.14*ipd*k, pinch = 0.04*ipd*k;
    renderPatch(browRect(lm[IDX.LBROW_MID]), (SQ)=>SQ.map(p=>({x:p.x - pinch*0.6, y:p.y + bd})));
    renderPatch(browRect(lm[IDX.RBROW_MID]), (SQ)=>SQ.map(p=>({x:p.x + pinch*0.6, y:p.y + bd})));

    // 3) Upper eyelids (gentle droop)
    function eyeRect(up, inn, out){
      const w=Math.hypot(out.x-inn.x, out.y-inn.y)*0.95, h=w*0.5;
      return { x: (inn.x+out.x)/2 - w/2, y: (inn.y+out.y)/2 - h/2, w, h };
    }
    const L_IN=133, L_OUT=33, R_IN=362, R_OUT=263;
    const lid=0.06*ipd*k;
    renderPatch(eyeRect(lm[IDX.L_UP], lm[L_IN], lm[L_OUT]), (SQ)=>SQ.map(p=>({x:p.x, y:p.y + lid})));
    renderPatch(eyeRect(lm[IDX.R_UP], lm[R_IN], lm[R_OUT]), (SQ)=>SQ.map(p=>({x:p.x, y:p.y + lid})));

    // Done; overlay contains warped regions over the live mirrored video.
  }

  // Capture
  snapBtn.addEventListener("click", async () => {
    const shot = document.createElement("canvas");
    shot.width = overlay.width; shot.height = overlay.height;
    const sctx = shot.getContext("2d");
    // base (mirrored video)
    sctx.translate(shot.width,0); sctx.scale(-1,1);
    sctx.drawImage(video, 0,0, shot.width, shot.height);
    sctx.setTransform(1,0,0,1,0,0);
    // overlay deforms
    sctx.drawImage(overlay, 0,0);
    const a = document.createElement("a"); a.href = shot.toDataURL("image/png"); a.download="crying-face.png";
    document.body.appendChild(a); a.click(); a.remove();
  });

  // Boot
  startBtn.addEventListener("click", async ()=>{ try{ await startCamera(); await startMP(); }catch{ statusEl.textContent="Camera blocked"; } });
  (async ()=>{ try{ await startCamera(); await startMP(); }catch(e){} })();
})();